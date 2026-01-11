import numpy as np
import tensorflow as tf
import os
import sys
import math
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from keras import mixed_precision
from AI.cpp.reversi_bitboard_cpp import ReversiBitboard
from AI.cpp.reversi_mcts_cpp import MCTS  # Use C++ MCTS
from AI.models.model_selector import try_load_model
from AI.config_loader import load_config
from AI.config import (
    NUM_GAMES_COMPARE,
    COMPARE_SIMS_N,
    Model1_Path,
    Model2_Path,
    Model1_Name,
    Model2_Name,
    MCTS_PREDICT_BATCH_SIZE
)

mixed_precision.set_global_policy('mixed_float16')

def _print_numpy_board(board_1d):
    print("  0 1 2 3 4 5 6 7")
    for r in range(8):
        row_str = f"{r}"
        for c in range(8):
            piece = board_1d[r * 8 + c]
            if piece == 1: row_str += "🔴"
            elif piece == 2: row_str += "⚪️"
            else: row_str += "🟩"
        print(row_str)

# Wrapper to bridge C++ MCTS and Keras Model (Same as in reviewHuman.py)
class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.predict_func = tf.function(
            self._predict_tf,
            input_signature=[
                tf.TensorSpec(shape=[None, 64], dtype=tf.int8),
                tf.TensorSpec(shape=[None], dtype=tf.int32) 
            ]
        )

    def board_to_input_planes_tf(self, board_1d_tf, current_player_tf):
        batch_size = tf.shape(board_1d_tf)[0]
        board_2d = tf.reshape(board_1d_tf, (batch_size, 8, 8))
        
        player_broadcast = tf.reshape(current_player_tf, (batch_size, 1, 1))
        player_broadcast = tf.tile(player_broadcast, [1, 8, 8])
        
        player_mask = tf.cast(tf.equal(board_2d, tf.cast(player_broadcast, tf.int8)), tf.float32)
        opponent_mask = tf.cast(tf.equal(board_2d, tf.cast(3 - player_broadcast, tf.int8)), tf.float32)
        
        return tf.stack([player_mask, opponent_mask], axis=-1)

    def _predict_tf(self, board_batch, player_batch):
        input_planes = self.board_to_input_planes_tf(board_batch, player_batch)
        policy, value = self.model(input_planes, training=False)
        return policy, value

    def _predict_internal_cpp(self, board_batch_np, player_batch_np):
        board_tensor = tf.convert_to_tensor(board_batch_np, dtype=tf.int8)
        player_tensor = tf.convert_to_tensor(player_batch_np, dtype=tf.int32)
        policy, value = self.predict_func(board_tensor, player_tensor)
        return policy.numpy(), value.numpy()

class MCTS_AIPlayer:
    def __init__(self, model_path, name, sims_per_move, config=None):
        self.name = name
        self.model_path = model_path
        if model_path == "Random":
            self.keras_model = None
            self.mcts = None
            print(f"Initialized Random Player '{name}'.")
        else:
            self.keras_model = try_load_model(model_path, config=config)
            if self.keras_model is None:
                 raise ValueError(f"Failed to load model: {model_path}")
            
            self.wrapped_model = ModelWrapper(self.keras_model)
            # For comparison (AI vs AI), we must use batch_size=1 because our C++ MCTS 
            # does not implement Virtual Loss for tree parallelization.
            # Using large batch_size within a single game results in redundant searches of the same path.
            batch_size = 1
            self.mcts = MCTS(self.wrapped_model, 2.3, batch_size) # c_puct=2.3 from config
            print(f"Initialized AI '{name}'. Model: {model_path}, SimsN: {sims_per_move}")
            
        self.sims_per_move = sims_per_move

    def choose_move(self, game_board: ReversiBitboard, player, verbose=False, temperature=1.0):
        if self.model_path == "Random":
            legal_moves = game_board.get_legal_moves()
            if not legal_moves:
                return None, 0, 0.0
            chosen_move = np.random.choice(legal_moves)
            return chosen_move, 1, 0.0

        # C++ search: (board, player, num_simulations, add_noise)
        # Enable noise to prevent identical games during comparison
        root_node = self.mcts.search(game_board, player, self.sims_per_move, True)

        children = root_node.children
        if not children:
            return None, 0, 0.0

        moves = list(children.keys())
        visit_counts = np.array([children[move].n_visits for move in moves])

        if temperature == 0:
            best_move_idx = np.argmax(visit_counts)
            chosen_move = moves[best_move_idx]
        else:
            # Add small epsilon to avoid div by zero if temperature is very small but not 0
            if temperature < 1e-3: temperature = 1e-3
            scaled_visits = visit_counts**(1.0 / temperature)
            if np.sum(scaled_visits) == 0:
                 probabilities = np.ones_like(scaled_visits) / len(scaled_visits)
            else:
                 probabilities = scaled_visits / np.sum(scaled_visits)
            chosen_move = np.random.choice(moves, p=probabilities)

        chosen_node = children[chosen_move]
        visits = chosen_node.n_visits
        
        q_value = -chosen_node.q_value

        return chosen_move, visits, q_value

def simulate_game(player1_ai, player2_ai, verbose=False, black_thinks_like_white=False):
    game_board = ReversiBitboard()
    current_player = 1

    player1_q_values = []
    player2_q_values = []

    if verbose:
        print("\n--- New game ---")
    while not game_board.is_game_over():
        current_ai = player1_ai if current_player == 1 else player2_ai

        if not game_board.get_legal_moves():
            if verbose: print(f"Player {current_player} ({current_ai.name}) has no legal move -> Passed")
            game_board.apply_move(-1)
            current_player = 3 - current_player
            continue

        temp_game_board = ReversiBitboard()
        temp_game_board.black_board = game_board.black_board
        temp_game_board.white_board = game_board.white_board
        temp_game_board.current_player = game_board.current_player
        temp_game_board.passed_last_turn = game_board.passed_last_turn

        search_player = current_player

        if black_thinks_like_white and current_player == 1:
            # Swap board perspective for black
            temp_black_board = temp_game_board.black_board
            temp_white_board = temp_game_board.white_board
            temp_game_board.black_board = temp_white_board
            temp_game_board.white_board = temp_black_board
            # current_player stays 1, but we treat it as white (2) logic? 
            # Actually if we swap boards, player 1 sees board as if they are white (but mapped to black bits).
            # This is complex logic, assuming standard play for now.
            pass

        num_turns = len(game_board.history)
        temperature = 1.0 if num_turns < 10 else 0.0 # Reduce temperature faster for evaluation
        
        # Note: choose_move expects integer player (1 or 2)
        move, visits, q_value = current_ai.choose_move(temp_game_board, search_player, verbose=verbose, temperature=temperature)

        if current_ai == player1_ai:
            player1_q_values.append(q_value)
        else:
            player2_q_values.append(q_value)

        if move is None: # Should not happen if legal moves exist, handled above
            if verbose: print(f"AI {current_ai.name} passed")
            game_board.apply_move(-1)
            current_player = 3 - current_player
            continue

        if verbose:
            print(f"{current_player} ({current_ai.name}) chose: {move} (Visits: {visits}, Q num: {q_value:.4f})")
        game_board.apply_move(move)
        current_player = 3 - current_player

    winner = game_board.get_winner()
    black_count = game_board.count_set_bits(game_board.black_board)
    white_count = game_board.count_set_bits(game_board.white_board)
    final_board_np = game_board.board_to_numpy()

    if verbose:
        print(f"\nGame finish. Scores: Black = {black_count}, White = {white_count}")
        _print_numpy_board(final_board_np)
        if winner == 1: print(f"Win: Black ({player1_ai.name})")
        elif winner == 2: print(f"Win: White ({player2_ai.name})")
        else: print("Draw")

    return winner, black_count, white_count, player1_q_values, player2_q_values, final_board_np

def run_comparison(model1_path, model1_name, model2_path, model2_name, num_games, SIMS_N, game_verbose=False, black_thinks_like_white=False):
# ...
    print(f"--- Model compare: {model1_name} vs {model2_name} ---")
    
    # Avoid parsing command line args when loading config for files
    class DummyArgs:
        def __init__(self):
            self.model = 'moe-2' # Default fallback

    # Helper to safely load config or default
    def get_config_for_path(path):
        try:
            # If path indicates a specific known config in yaml, try to map it?
            # Or just rely on fallback.
            # load_config(DummyArgs()) will load 'moe-2' config.
            # But try_load_model handles most logic.
            # If it is .h5, we might want None to let model_selector handle it.
            if path.endswith('.h5'):
                 return None
            return load_config(DummyArgs())
        except:
            return None

    ai1 = MCTS_AIPlayer(model1_path, model1_name, SIMS_N, config=get_config_for_path(model1_path))
    ai2 = MCTS_AIPlayer(model2_path, model2_name, SIMS_N, config=get_config_for_path(model2_path))

    wins = {ai1.name: 0, ai2.name: 0, "Draw": 0}
    total_stones = {ai1.name: 0, ai2.name: 0}
    all_q_values_ai1 = []
    all_q_values_ai2 = []
    start_time = time.time()

    print(f"\nStart simulation -> {num_games}")
    for i in range(num_games):
        if i % 2 == 0:
            p1, p2 = ai1, ai2
            p1_name, p2_name = ai1.name, ai2.name
        else:
            p1, p2 = ai2, ai1
            p1_name, p2_name = ai2.name, ai1.name

        winner, black_count, white_count, p1_q_values, p2_q_values, final_board_np = simulate_game(p1, p2, verbose=game_verbose)

        # Print final board and score for each game
        print(f"\nGame {i+1} Result: {p1_name} (Black) {black_count} - {white_count} {p2_name} (White)")
        _print_numpy_board(final_board_np)

        if p1 == ai1:
            all_q_values_ai1.extend(p1_q_values)
            all_q_values_ai2.extend(p2_q_values)
        else:
            all_q_values_ai1.extend(p2_q_values)
            all_q_values_ai2.extend(p1_q_values)

        # Winner 1 is always Black (p1), 2 is White (p2)
        if winner == 1: # p1 won
            wins[p1_name] += 1
            total_stones[p1_name] += black_count
            total_stones[p2_name] += white_count
        elif winner == 2: # p2 won
            wins[p2_name] += 1
            total_stones[p1_name] += black_count
            total_stones[p2_name] += white_count
        else:
            wins["Draw"] += 1
            total_stones[p1_name] += black_count
            total_stones[p2_name] += white_count

        progress = (i + 1) / num_games
        bar = '#' * int(progress * 20)
        print(f"[{bar:<20}] {int(progress*100)}% | Wins: {ai1.name} {wins[ai1.name]} - {wins[ai2.name]} {ai2.name} (Draw: {wins['Draw']})", end='\r')

    end_time = time.time()
    print("\n\nSimulation finish")
    print(f"Total: {end_time - start_time:.2f} sec")
    print("\n--- Results ---")
    print(f"SimsN : {SIMS_N}")
    print(f"{ai1.name} wins: {wins[ai1.name]}")
    print(f"{ai2.name} wins: {wins[ai2.name]}")
    print(f"Draw: {wins['Draw']}")
    print("-----------------")
    avg_stones_1 = total_stones[ai1.name] / num_games
    avg_stones_2 = total_stones[ai2.name] / num_games
    print(f"{ai1.name} ave scores: {avg_stones_1:.1f}")
    print(f"{ai2.name} ave scores: {avg_stones_2:.1f}")

    if all_q_values_ai1:
        print(f"{ai1.name} ave Q num: {np.mean(all_q_values_ai1):.4f}")
    else:
        print(f"No Q num data of {ai1.name}")
    if all_q_values_ai2:
        print(f"{ai2.name} ave Q num: {np.mean(all_q_values_ai2):.4f}")
    else:
        print(f"No Q num data of {ai2.name}")
    print("-----------------")

    avg_stones_1 = total_stones[ai1.name] / num_games
    avg_stones_2 = total_stones[ai2.name] / num_games
    avg_q1 = np.mean(all_q_values_ai1) if all_q_values_ai1 else 0.0
    avg_q2 = np.mean(all_q_values_ai2) if all_q_values_ai2 else 0.0

    stats = {
        ai1.name: {"wins": wins[ai1.name], "stones": avg_stones_1, "q": avg_q1},
        ai2.name: {"wins": wins[ai2.name], "stones": avg_stones_2, "q": avg_q2},
        "draws": wins["Draw"]
    }
    
    return stats

if __name__ == "__main__":
    m1_path = "./models/TF/3G.h5"
    m1_name = "TF3"
    m2_path = "./models/TF/MoE-2.keras"
    m2_name = "MoE-2"

    games = 10
    sims = 50

    stats = run_comparison(m1_path, m1_name, m2_path, m2_name, games, sims, game_verbose=False, black_thinks_like_white=False)
    print(f"Stats: {stats}")
