import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from absl import logging
logging.set_verbosity(logging.ERROR)
import numpy as np
import tensorflow as tf
import sys
import math
import argparse
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# Configure TensorFlow logging
import logging
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from keras import mixed_precision
from AI.cpp.reversi_bitboard_cpp import ReversiBitboard
from AI.models.model_selector import try_load_model
from AI.config_loader import load_config

# Set mixed precision
try:
    mixed_precision.set_global_policy('mixed_float16')
except:
    pass

# User observation: High PUCT is better for low SimsN
DEFAULT_PUCT = 2.5
DEFAULT_SIMS = 5
GAMES_NUM = 20

def board_to_input_planes_tf(board_1d_tf, current_player_tf):
    board_2d_tf = tf.reshape(board_1d_tf, (8, 8))
    me_mask = tf.cast(tf.equal(board_2d_tf, current_player_tf), tf.float32)
    opp_mask = tf.cast(tf.equal(board_2d_tf, 3 - current_player_tf), tf.float32)
    return tf.stack([me_mask, opp_mask], axis=-1)

class MCTSNode:
    def __init__(self, prior_p=0.0):
        self.children = {}
        self.n_visits = 0
        self.q_value = 0.0
        self.prior_p = prior_p

    def ucb_score(self, parent_visits, c_puct):
        # UCB = Q + U
        # Q is the exploitation term, U is the exploration term
        q = self.q_value if self.n_visits > 0 else 0.0
        u = c_puct * self.prior_p * math.sqrt(parent_visits) / (1 + self.n_visits)
        return q + u

    def update(self, v):
        self.n_visits += 1
        self.q_value += (v - self.q_value) / self.n_visits

class MCTS:
    def __init__(self, model):
        self.model = model
        self._predict_graph = tf.function(
            self._predict_internal,
            input_signature=[tf.TensorSpec(shape=[64], dtype=tf.int8), tf.TensorSpec(shape=(), dtype=tf.int8)]
        )

    def _predict_internal(self, board_tensor, player_tensor):
        input_planes = board_to_input_planes_tf(board_tensor, player_tensor)
        input_tensor_batch = tf.expand_dims(input_planes, axis=0)
        outputs = self.model(input_tensor_batch, training=False)
        return outputs[0][0], outputs[1][0][0]

    def search(self, game_board: ReversiBitboard, player, num_simulations, c_puct):
        root = MCTSNode()
        
        # Expansion (Root)
        p_raw, _ = self._predict_graph(tf.convert_to_tensor(game_board.board_to_numpy(), dtype=tf.int8), 
                                        tf.convert_to_tensor(player, dtype=tf.int8))
        p_raw = p_raw.numpy()
        valid_moves = game_board.get_legal_moves()
        if not valid_moves: return None
        
        # Softmax-like normalization for valid moves only? 
        # Usually AlphaZero uses the raw policy output for valid moves and normalizes them.
        p_sum = sum(p_raw[m] for m in valid_moves) + 1e-9
        for m in valid_moves:
            root.children[m] = MCTSNode(prior_p=p_raw[m] / p_sum)

        for _ in range(num_simulations):
            node = root
            sim_board = ReversiBitboard()
            sim_board.black_board, sim_board.white_board = game_board.black_board, game_board.white_board
            sim_board.current_player = game_board.current_player
            sim_board.passed_last_turn = game_board.passed_last_turn
            
            path = []
            
            # Selection
            while node.children and not sim_board.is_game_over():
                parent_visits = node.n_visits if node.n_visits > 0 else 1
                move, node = max(node.children.items(), key=lambda it: it[1].ucb_score(parent_visits, c_puct))
                sim_board.apply_move(move)
                path.append(node)
            
            # Expansion & Evaluation
            if not sim_board.is_game_over():
                p, v_raw = self._predict_graph(tf.convert_to_tensor(sim_board.board_to_numpy(), dtype=tf.int8), 
                                               tf.convert_to_tensor(sim_board.current_player, dtype=tf.int8))
                p, v_raw = p.numpy(), v_raw.numpy()
                v_moves = sim_board.get_legal_moves()
                if v_moves:
                    s = sum(p[m] for m in v_moves) + 1e-9
                    for m in v_moves:
                        node.children[m] = MCTSNode(prior_p=p[m] / s)
                    v = -float(v_raw) # Value is from perspective of current player
                else:
                    # Pass turn logic: if I must pass, predict value for opponent? 
                    # ReversiBitboard handles pass internally if apply_move(-1) is called, 
                    # but here we are at a leaf node that needs expansion.
                    # If no legal moves, it's effectively a pass for the next turn, but we stop expansion here.
                    # We can just evaluate the board.
                     _, v_pass = self._predict_graph(tf.convert_to_tensor(sim_board.board_to_numpy(), dtype=tf.int8), 
                                                    tf.convert_to_tensor(3 - sim_board.current_player, dtype=tf.int8))
                     v = float(v_pass) # If I pass, the value returned is for the opponent, so it's good for me? No.
                     # Actually simpler: just evaluate the board. The model predicts value for 'current_player'.
                     # If I have no moves, I am still current player until I pass.
                     # Let's simplify: if no moves, we consider it terminal for this expansion step and just use the value.
                     v = -float(v_raw)
            else:
                # Terminal
                win = sim_board.get_winner()
                if win == 0: v = 0.0
                else:
                    v = -1.0 if win == sim_board.current_player else 1.0

            # Backpropagation
            for n in reversed(path):
                n.update(v)
                v = -v
            root.n_visits += 1

        if not root.children: return None
        # Return children visits for temperature sampling
        return {m: child.n_visits for m, child in root.children.items()}

def play_game(ai1, ai2, sims, c_puct, ai1_is_black):
    board = ReversiBitboard()
    cur = 1 # Black
    moves_count = 0
    
    while not board.is_game_over():
        valid = board.get_legal_moves()
        if not valid:
            board.apply_move(-1)
        else:
            # Determine who moves
            if (cur == 1 and ai1_is_black) or (cur == 2 and not ai1_is_black):
                ai = ai1
            else:
                ai = ai2
            
            visits_map = ai.search(board, cur, sims, c_puct)
            
            if not visits_map:
                 move = valid[0]
            else:
                moves = list(visits_map.keys())
                counts = np.array([visits_map[m] for m in moves], dtype=np.float32)
                
                # Temperature scheduling
                if moves_count < 20: # First 20 moves: probabilistic
                    if np.sum(counts) > 0:
                        probs = counts / np.sum(counts)
                        move = np.random.choice(moves, p=probs)
                    else:
                        move = np.random.choice(moves)
                else: # Deterministic
                    move = moves[np.argmax(counts)]
            
            board.apply_move(move)
            moves_count += 1
        cur = 3 - cur
        
    b = board.count_set_bits(board.black_board)
    w = board.count_set_bits(board.white_board)
    
    if ai1_is_black:
        return b, w # AI1 (Black), AI2 (White)
    else:
        return w, b # AI1 (White), AI2 (Black)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new', type=str, required=True, help='Path to new model')
    parser.add_argument('--best', type=str, required=True, help='Path to best model (backup)')
    parser.add_argument('--sims', type=int, default=DEFAULT_SIMS, help='MCTS simulations')
    parser.add_argument('--puct', type=float, default=DEFAULT_PUCT, help='PUCT constant')
    parser.add_argument('--games', type=int, default=GAMES_NUM, help='Number of games')
    args = parser.parse_args()

    print(f"--- Evaluation: New vs Best (Sims={args.sims}, PUCT={args.puct}) ---")
    
    config_new = load_config(type('Args', (), {'model': 'moe-2'}))
    model_new = try_load_model(args.new, config_new)
    ai_new = MCTS(model_new)
    
    config_best = load_config(type('Args', (), {'model': 'moe-2'}))
    model_best = try_load_model(args.best, config_best)
    ai_best = MCTS(model_best)
    
    wins_new = 0
    wins_best = 0
    draws = 0
    total_stones_new = 0
    
    for i in range(args.games):
        new_is_black = (i % 2 == 0)
        p1_stones, p2_stones = play_game(ai_new, ai_best, args.sims, args.puct, new_is_black)
        
        total_stones_new += p1_stones
        
        if p1_stones > p2_stones:
            wins_new += 1
            res = "Win "
        elif p2_stones > p1_stones:
            wins_best += 1
            res = "Loss"
        else:
            draws += 1
            res = "Draw"
            
        color_str = "Black" if new_is_black else "White"
        print(f"G{i+1:2}: {res} (New:{p1_stones:2} Best:{p2_stones:2}) {color_str}", flush=True)

    avg_stones = total_stones_new / args.games
    win_rate = (wins_new / args.games) * 100
    
    print("\n[RESULT]")
    print(f"New Model Wins: {wins_new}")
    print(f"Best Model Wins: {wins_best}")
    print(f"Draws: {draws}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Stones (New): {avg_stones:.2f}")
    
    # Simple output for parser
    print(f"FINAL_METRICS|{win_rate}|{avg_stones}")

if __name__ == "__main__":
    main()
