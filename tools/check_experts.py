import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AI.models.model_selector import try_load_model

model_path = './models/TF/MoE-1-optuna.h5'
model = try_load_model(model_path)

def get_expert_choice(input_planes):
    x = np.expand_dims(input_planes, axis=0)
    _ = model(x, training=False)
    for layer in model.layers:
        if 'dynamic_assembly' in layer.name:
            probs = layer.last_probs.numpy()[0]
            top2_idx = np.argsort(probs)[-2:][::-1]
            return top2_idx, probs
    return None, None

blank_board = np.zeros((8, 8, 2), dtype=np.float32)
blank_board[3, 4, 0] = 1; blank_board[4, 3, 0] = 1
blank_board[3, 3, 1] = 1; blank_board[4, 4, 1] = 1

corner_board = blank_board.copy()
corner_board[1, 1, 0] = 1
corner_board[0, 0, 1] = 1

edge_board = blank_board.copy()
edge_board[2, 0, 0] = 1; edge_board[3, 0, 0] = 1; edge_board[4, 0, 0] = 1
edge_board[5, 0, 1] = 1

middle_board = blank_board.copy()
np.random.seed(42)
random_pos = np.random.choice(64, 20, replace=False)
for i, pos in enumerate(random_pos):
    r, c = pos // 8, pos % 8
    middle_board[r, c, i % 2] = 1

late_board = np.ones((8, 8, 2), dtype=np.float32)

late_board[0, 1, :] = 0
late_board[1, 0, :] = 0
late_board[6, 7, :] = 0
late_board[7, 6, :] = 0

patterns = {
    "Initial Stage": blank_board,
    "Corner Situation": corner_board,
    "Edge Attack": edge_board,
    "Middle Game (Random)": middle_board,
    "Late Game (Almost Full)": late_board
}

print("--- Expert Diagnosis (Top-2) ---")
for name, board in patterns.items():
    idxs, probs = get_expert_choice(board)
    print(f"\nPattern: {name}")
    for i, idx in enumerate(idxs):
        print(f" {i+1} Selected: Expert {idx} ({probs[idx]:.1%})")
