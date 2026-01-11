import numpy as np
import random
import os
import tensorflow as tf
from AI.config import Play_Games_Num
from AI.models.model_selector import try_load_model
import math
from AI.cpp.reversi_bitboard_cpp import ReversiBitboard
from keras import mixed_precision
from AI.config_loader import load_config

config = load_config()
print(f"Loaded config for model: {config['model_name']}")

try:
    mixed_precision.set_global_policy('mixed_float16')
except:
    pass

MCTS_SIMS_PER_MOVE = 5
NUM_GAMES_TO_PLAY = Play_Games_Num
MODEL_PATH = config.get('model_save_path', './models/TF/model.h5')
C_PUCT = 4.0

def board_to_input_planes_tf(board_1d_tf, current_player_tf):
    board_2d_tf = tf.reshape(board_1d_tf, (8, 8))
    me_mask = tf.cast(tf.equal(board_2d_tf, current_player_tf), tf.float32)
    opp_mask = tf.cast(tf.equal(board_2d_tf, 3 - current_player_tf), tf.float32)
    return tf.stack([me_mask, opp_mask], axis=-1)

class MCTSNode:
    __slots__ = ['children', 'n_visits', 'q_value', 'prior_p']
    def __init__(self, prior_p=0.0):
        self.children = {}
        self.n_visits = 0
        self.q_value = 0.0
        self.prior_p = prior_p

    def ucb_score(self, parent_visits, c_puct):
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

    def search(self, game_board: ReversiBitboard, player, num_simulations):
        root = MCTSNode()

        p_raw, _ = self._predict_graph(tf.convert_to_tensor(game_board.board_to_numpy(), dtype=tf.int8), 
                                        tf.convert_to_tensor(player, dtype=tf.int8))
        p_raw = p_raw.numpy()
        valid_moves = game_board.get_legal_moves()
        if not valid_moves: return None
        p_sum = sum(p_raw[m] for m in valid_moves) + 1e-9
        for m in valid_moves:
            root.children[m] = MCTSNode(prior_p=p_raw[m] / p_sum)

        for _ in range(num_simulations):
            node = root
            sim_board = ReversiBitboard()
            sim_board.black_board, sim_board.white_board = game_board.black_board, game_board.white_board
            sim_board.current_player = game_board.current_player
            
            path = []
            while node.children and not sim_board.is_game_over():
                parent_visits = node.n_visits if node.n_visits > 0 else 1
                move, node = max(node.children.items(), key=lambda it: it[1].ucb_score(parent_visits, C_PUCT))
                sim_board.apply_move(move)
                path.append(node)
            
            if not sim_board.is_game_over():
                p, v_raw = self._predict_graph(tf.convert_to_tensor(sim_board.board_to_numpy(), dtype=tf.int8), 
                                               tf.convert_to_tensor(sim_board.current_player, dtype=tf.int8))
                p, v_raw = p.numpy(), v_raw.numpy()
                v_moves = sim_board.get_legal_moves()
                if v_moves:
                    s = sum(p[m] for m in v_moves) + 1e-9
                    for m in v_moves:
                        node.children[m] = MCTSNode(prior_p=p[m] / s)
                    v = -float(v_raw)
                else:
                    _, v_pass = self._predict_graph(tf.convert_to_tensor(sim_board.board_to_numpy(), dtype=tf.int8), 
                                                    tf.convert_to_tensor(3 - sim_board.current_player, dtype=tf.int8))
                    v = float(v_pass)
            else:
                win = sim_board.get_winner()
                if win == 0: v = 0.0
                else:
                    v = -1.0 if win == sim_board.current_player else 1.0

            for n in reversed(path):
                n.update(v)
                v = -v
            root.n_visits += 1

        return max(root.children.items(), key=lambda it: it[1].n_visits)[0]

def main():
    print(f"--- MoE Competition (Sims={MCTS_SIMS_PER_MOVE}) ---")
    model = try_load_model(MODEL_PATH, config)
    mcts_ai = MCTS(model)
    
    wins, draws, total_stones = 0, 0, 0
    for i in range(NUM_GAMES_TO_PLAY):
        mcts_is_black = (i % 2 == 0)
        board = ReversiBitboard()
        cur = 1
        
        while not board.is_game_over():
            valid = board.get_legal_moves()
            if not valid:
                board.apply_move(-1)
            else:
                if (cur == 1 and mcts_is_black) or (cur == 2 and not mcts_is_black):
                    move = mcts_ai.search(board, cur, MCTS_SIMS_PER_MOVE)
                else:
                    move = random.choice(valid)
                board.apply_move(move if move is not None else -1)
            cur = 3 - cur
        
        b, w = board.count_set_bits(board.black_board), board.count_set_bits(board.white_board)
        ai_stones = b if mcts_is_black else w
        total_stones += ai_stones
        
        if ai_stones > 32:
            res = "Win "; wins += 1
        elif ai_stones == 32:
            res = "Draw"; draws += 1
        else:
            res = "Loss"
            
        print(f"G{i+1:2}: {res} (AI:{ai_stones:2} Bot:{64-ai_stones:2}) {'Black' if mcts_is_black else 'White'}")

    win_rate = (wins / NUM_GAMES_TO_PLAY) * 100
    avg_stones = total_stones / NUM_GAMES_TO_PLAY
    print(f"\n[RESULT] WinRate: {win_rate:.1f}% | AvgStones: {avg_stones:.2f}")

if __name__ == "__main__":
    main()
