#!/usr/bin/env python3
import sys
import os
import argparse
import random
import numpy as np
import tensorflow as tf
import math

# Determine project root and add to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from AI.cpp.reversi_bitboard_cpp import ReversiBitboard
from AI.cpp.reversi_mcts_cpp import MCTS  # Use C++ MCTS
from AI.models.model_selector import try_load_model
from AI.config_loader import load_config
from AI.config import C_PUCT, R_SIMS_N, MCTS_PREDICT_BATCH_SIZE

def print_board(board_1d, legal_moves=None):
    print("  A B C D E F G H")
    for r in range(8):
        row_str = ""
        for c in range(8):
            idx = r * 8 + c
            piece = board_1d[idx]
            if piece == 1:
                row_str += "🔴"
            elif piece == 2:
                row_str += "⚪️"
            else:
                if legal_moves and idx in legal_moves:
                    row_str += "🟦" # Hint for legal move
                else:
                    row_str += "🟩"
        print(f"{r+1}{row_str}")

def get_human_move(legal_moves):
    while True:
        try:
            move_str = input("Your turn 🫵 (e.g. A1): ").strip()
            if move_str.lower() == "pass":
                if -1 in legal_moves:
                    return -1
                else:
                    print("❌ Cannot pass when legal moves exist.")
                    continue

            if len(move_str) < 2:
                raise ValueError("Invalid format.")

            col_char = move_str[0].upper()
            row_char = move_str[1:]
            
            if not row_char.isdigit():
                 raise ValueError("Invalid row.")

            if not ('A' <= col_char <= 'H'):
                raise ValueError("Invalid column.")

            col = ord(col_char) - ord('A')
            row = int(row_char) - 1
            
            if not (0 <= row < 8):
                 raise ValueError("Row out of bounds.")

            move = row * 8 + col

            if move not in legal_moves:
                print(f"❌ Invalid move: {move_str}")
                print(f"Valid moves: {[index_to_coord(m) for m in legal_moves if m != -1]}")
                continue
            return move
        except ValueError as e:
            print(f"Error: {e}. Please enter again (e.g. D3).")
        except Exception as e:
            print(f"Unexpected error: {e}. Try again.")

def index_to_coord(index):
    if index == -1:
        return "PASS"
    row = index // 8
    col = index % 8
    return f"{chr(ord('A') + col)}{row + 1}"

def main():
    parser = argparse.ArgumentParser(description="Othello Human vs AI")
    parser.add_argument('--model', type=str, default='models/TF/MoE-2.keras', help='Path to the model file')
    parser.add_argument('--sims', type=int, default=R_SIMS_N, help='Number of MCTS simulations per move')
    parser.add_argument('--color', type=str, choices=['black', 'white', 'random'], default='random', help='Human player color')
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)
    
    print(f"Loading model from: {model_path}")
    
    # Avoid config_loader parsing command line args automatically which causes error for file paths
    class DummyArgs:
        def __init__(self):
            self.model = 'moe-2' # Default fallback
    
    try:
        # First try to load config based on the filename stem if it matches a yaml key
        # But here we are dealing with file paths, so we just load a default config as a fallback/base
        # The model_selector.identify_architecture will likely handle the specifics for h5 files
        config = load_config(DummyArgs())
    except Exception as e:
        print(f"Warning: Could not load config (using default/none): {e}")
        config = None

    try:
        keras_model = try_load_model(model_path, config=config)
        if keras_model is None:
             raise ValueError("Model loaded as None.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Wrapper to bridge C++ MCTS and Keras Model
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
            # board_1d_tf: [Batch, 64]
            # current_player_tf: [Batch]
            
            batch_size = tf.shape(board_1d_tf)[0]
            board_2d = tf.reshape(board_1d_tf, (batch_size, 8, 8)) # [B, 8, 8]
            
            # Create masks
            # player_tf is 1 or 2. board contains 1 (black), 2 (white).
            # We need to broadcast player to [B, 8, 8]
            player_broadcast = tf.reshape(current_player_tf, (batch_size, 1, 1))
            player_broadcast = tf.tile(player_broadcast, [1, 8, 8])
            
            player_mask = tf.cast(tf.equal(board_2d, tf.cast(player_broadcast, tf.int8)), tf.float32)
            opponent_mask = tf.cast(tf.equal(board_2d, tf.cast(3 - player_broadcast, tf.int8)), tf.float32)
            
            # Stack to [B, 8, 8, 2]
            return tf.stack([player_mask, opponent_mask], axis=-1)

        def _predict_tf(self, board_batch, player_batch):
            input_planes = self.board_to_input_planes_tf(board_batch, player_batch)
            policy, value = self.model(input_planes, training=False)
            return policy, value

        def _predict_internal_cpp(self, board_batch_np, player_batch_np):
            # board_batch_np: numpy array [B, 64] (int8)
            # player_batch_np: list or numpy array [B]
            
            board_tensor = tf.convert_to_tensor(board_batch_np, dtype=tf.int8)
            # Ensure player is int32 for TF
            player_tensor = tf.convert_to_tensor(player_batch_np, dtype=tf.int32)
            
            policy, value = self.predict_func(board_tensor, player_tensor)
            
            return policy.numpy(), value.numpy()

    wrapped_model = ModelWrapper(keras_model)

    # Use C++ MCTS
    # For human review, a smaller batch size (like 1 or 4) is better for search quality 
    # than the large batch size used for parallel self-play.
    mcts_batch_size = 1 
    mcts_ai = MCTS(wrapped_model, C_PUCT, mcts_batch_size)
    
    game_board = ReversiBitboard()

    if args.color == 'random':
        human_player = random.choice([1, 2])
    elif args.color == 'black':
        human_player = 1
    else:
        human_player = 2

    ai_player = 3 - human_player

    human_color_str = "Black 🔴" if human_player == 1 else "White ⚪️"
    ai_color_str = "White ⚪️" if ai_player == 2 else "Black 🔴"
    print(f"You : {human_color_str}, AI : {ai_color_str}")

    current_player = 1

    while not game_board.is_game_over():
        turn_player_name = "You" if current_player == human_player else "AI"
        turn_color = "Black" if current_player == 1 else "White"
        print(f"\n--- Turn: {turn_color} ({turn_player_name}) ---")
        
        legal_moves = game_board.get_legal_moves()
        print_board(game_board.board_to_numpy(), legal_moves if current_player == human_player else None)

        if not legal_moves:
            print(f"{turn_color} has no legal moves -> pass")
            game_board.apply_move(-1)
            current_player = 3 - current_player
            continue

        if current_player == human_player:
            move = get_human_move(legal_moves)
        else:
            print(f"AI is thinking ({args.sims} sims)...")
            # C++ search: (board, player, num_simulations, add_noise)
            # Returns MCTSNode (the root of the search)
            root_node = mcts_ai.search(game_board, current_player, args.sims, False)
            
            # Find the best child from the root
            best_move = -1
            best_visits = -1
            best_q = 0.0
            
            children = root_node.children
            if children:
                # Select the move with the most visits
                best_move = max(children.keys(), key=lambda m: children[m].n_visits)
                best_child = children[best_move]
                best_visits = best_child.n_visits
                best_q = best_child.q_value
            
            move = best_move
            print(f"AI plays: {index_to_coord(move)}")
            if move != -1:
                 print(f"  Confidence (visits): {best_visits}")
                 print(f"  Value (Q): {best_q:.4f}")

        game_board.apply_move(move)
        current_player = 3 - current_player

    print("\n--- Game Over ---")
    print_board(game_board.board_to_numpy())

    winner = game_board.get_winner()
    black_stones = game_board.count_set_bits(game_board.black_board)
    white_stones = game_board.count_set_bits(game_board.white_board)
    print(f"Scores - Black: {black_stones}, White: {white_stones}")

    if winner == 0:
        print("Draw")
    elif winner == human_player:
        print("You won! 🎉")
    else:
        print("AI won! 🤖")

if __name__ == "__main__":
    main()