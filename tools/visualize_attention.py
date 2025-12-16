import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import msgpack
import glob
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AI.cpp.reversi_bitboard_cpp import ReversiBitboard

from AI.models.transformer import build_model
from AI.config import TRANSFORMER_MODEL_PATH, TRAINING_DATA_DIR

MODEL_WEIGHTS_PATH = "./models/TF/1G.h5"
DATA_DIR = os.path.join(TRAINING_DATA_DIR, "1G")

QUERY_INDEX = 0
TARGET_BLOCK_INDEX = 3

def get_legal_moves(board_2d, current_player):
    bb = ReversiBitboard()
    
    black_mask = 0
    white_mask = 0
    
    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            if board_2d[r, c] == 1:
                black_mask |= (1 << idx)
            elif board_2d[r, c] == 2:
                white_mask |= (1 << idx)
                
    bb.black_board = black_mask
    bb.white_board = white_mask
    bb.current_player = current_player
    
    return bb.get_legal_moves()

def index_to_coord(index):
    if index < 0 or index >= 64: return "PASS"
    row = index // 8
    col = index % 8
    return f"{chr(ord('A') + col)}{row + 1}"

def load_random_sample(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.msgpack"))
    if not files:
        raise FileNotFoundError(f"No .msgpack files found in {data_dir}")
    
    file_path = random.choice(files)
    print(f"Loading sample from: {file_path}")
    
    with open(file_path, "rb") as f:
        data = msgpack.unpack(f, raw=False)
    move_records = []
    if isinstance(data[0], list):
        game = random.choice(data)
        move_records = game
    else:
        move_records = data

    if not move_records:
         raise ValueError("No move records found in file.")

    sample_record = random.choice(move_records)
    board_state = np.array(sample_record['board'], dtype=np.float32)
    input_board = np.zeros((1, 8, 8, 2), dtype=np.float32)
    current_player_color = sample_record['player']

    board_2d = board_state.reshape(8, 8)
    
    if current_player_color == 1:
        input_board[0, :, :, 0] = (board_2d == 1).astype(np.float32)
        input_board[0, :, :, 1] = (board_2d == 2).astype(np.float32)
        print("Sample turn: Black's turn")
    else:
        input_board[0, :, :, 0] = (board_2d == 2).astype(np.float32)
        input_board[0, :, :, 1] = (board_2d == 1).astype(np.float32)
        print("Sample turn: White's turn")
        
    return input_board, board_2d, current_player_color


def draw_grid_lines(ax):
    for x in range(9):
        ax.axvline(x, color='black', linewidth=1)
        ax.axhline(x, color='black', linewidth=1)

def overlay_stones(ax, board_2d, alpha=1.0):
    for y in range(8):
        for x in range(8):
            stone = board_2d[y, x]
            if stone == 1:
                circle = patches.Circle((x + 0.5, y + 0.5), 0.4, facecolor='black', edgecolor='black', alpha=alpha)
                ax.add_patch(circle)
            elif stone == 2:
                circle = patches.Circle((x + 0.5, y + 0.5), 0.4, facecolor='white', edgecolor='black', alpha=alpha)
                ax.add_patch(circle)

def draw_othello_board(ax, board_2d):
    ax.set_facecolor('green')
    draw_grid_lines(ax)
    overlay_stones(ax, board_2d, alpha=1.0)

def setup_board_axis(ax, invert=False):
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(8) + 0.5)
    ax.set_yticks(np.arange(8) + 0.5)
    ax.set_xticklabels(['A','B','C','D','E','F','G','H'])
    ax.set_yticklabels(['1','2','3','4','5','6','7','8'])
    ax.tick_params(axis='both', which='both', length=0)
    if invert:
        ax.invert_yaxis()

def plot_integrated_attention(input_board_2d, attention_heads_map, query_idx, block_idx, policy_prob=None):
    num_heads = attention_heads_map.shape[0]
    
    fig = plt.figure(figsize=(14, 6))
    
    coord = index_to_coord(query_idx)
    title_text = f"Best Move: {coord}"
    if policy_prob is not None:
        title_text += f" (Prob: {policy_prob:.4f})"
    
    ax_actual = fig.add_subplot(1, 2, 1)
    ax_actual.set_title("Actual Board")
    draw_othello_board(ax_actual, input_board_2d)
    
    qx, qy = query_idx % 8, query_idx // 8
    rect = patches.Rectangle((qx, qy), 1, 1, linewidth=3, edgecolor='red', facecolor='none')
    ax_actual.add_patch(rect)
    
    setup_board_axis(ax_actual, invert=True)

    ax_integ = fig.add_subplot(1, 2, 2)
    ax_integ.set_title(f"Integrated Attention (Top 30% per Head)\n{title_text}")
    
    draw_othello_board(ax_integ, input_board_2d)
    
    rect2 = patches.Rectangle((qx, qy), 1, 1, linewidth=3, edgecolor='red', facecolor='none', zorder=10)
    ax_integ.add_patch(rect2)
    
    cmap = plt.get_cmap('tab10')
    head_colors = [cmap(i) for i in range(num_heads)]
    
    overlay_rgb = np.zeros((8, 8, 3), dtype=np.float32)
    overlay_alpha = np.zeros((8, 8), dtype=np.float32)
    
    for h in range(num_heads):
        att_map = attention_heads_map[h, query_idx, :].reshape(8, 8)
        threshold = np.percentile(att_map, 70)
        mask = att_map >= threshold
        color = np.array(head_colors[h][:3])
        
        for r in range(8):
            for c in range(8):
                if mask[r, c]:
                    weight = 0.5
                    
                    overlay_rgb[r, c] += color * weight
                    overlay_alpha[r, c] += weight

    max_val = np.max(overlay_rgb)
    if max_val > 1.0:
        overlay_rgb /= max_val
        
    max_alpha = np.max(overlay_alpha)
    if max_alpha > 0:
        overlay_alpha = (overlay_alpha / max_alpha) * 0.7
    
    final_overlay = np.dstack((overlay_rgb, overlay_alpha))
    
    ax_integ.imshow(final_overlay, extent=[0, 8, 8, 0], origin='upper')
    
    setup_board_axis(ax_integ)
    
    legend_patches = [patches.Patch(color=head_colors[i], label=f'Head {i+1}') for i in range(num_heads)]
    ax_integ.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize='small')

    plt.tight_layout()
    print(f"\nVisualizing Integrated Attention for: {coord}")
    plt.show()

def setup_board_axis(ax, invert=False):
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(8) + 0.5)
    ax.set_yticks(np.arange(8) + 0.5)
    ax.set_xticklabels(['A','B','C','D','E','F','G','H'])
    ax.set_yticklabels(['1','2','3','4','5','6','7','8'])
    ax.tick_params(axis='both', which='both', length=0)
    if invert:
        ax.invert_yaxis()

def draw_othello_board(ax, board_2d):
    ax.set_facecolor('green')
    draw_grid_lines(ax)
    overlay_stones(ax, board_2d, alpha=1.0)

if __name__ == "__main__":
    model = build_model(visualize_mode=True) 
    try:
        print(f"Loading weights from {MODEL_WEIGHTS_PATH}...")
        model.load_weights(MODEL_WEIGHTS_PATH, by_name=True, skip_mismatch=True)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Ensure the model architecture parameters define in build_model match the saved weights.")
        sys.exit(1)

    try:
        input_board_batch, board_2d_display, current_player = load_random_sample(DATA_DIR)
    except Exception as e:
        print(f"Data loading error: {e}")
        sys.exit(1)

    print("Running inference...")
    outputs = model.predict(input_board_batch)
    
    policy_logits = outputs[0][0]
    
    legal_moves = get_legal_moves(board_2d_display, current_player)
    print(f"Legal moves: {[index_to_coord(m) for m in legal_moves]}")
    
    if not legal_moves:
        print("No legal moves (Pass). Skipping visualization.")
        sys.exit(0)

    max_prob = -1.0
    best_move = -1
    
    for move in legal_moves:
        prob = policy_logits[move]
        if prob > max_prob:
            max_prob = prob
            best_move = move
            
    print(f"Best move selected by Policy: {index_to_coord(best_move)} with prob {max_prob:.4f}")

    if TARGET_BLOCK_INDEX == -1:
        attention_map_batch = outputs[-1]
        print("Visualizing the last Transformer block.")
    else:
        attention_map_batch = outputs[2 + TARGET_BLOCK_INDEX]
        print(f"Visualizing Transformer block index: {TARGET_BLOCK_INDEX}")

    attention_map_single = attention_map_batch[0]

    plot_integrated_attention(board_2d_display, attention_map_single, best_move, TARGET_BLOCK_INDEX, policy_prob=max_prob)


if __name__ == "__main__":
    model = build_model(visualize_mode=True) 

    try:
        print(f"Loading weights from {MODEL_WEIGHTS_PATH}...")
        model.load_weights(MODEL_WEIGHTS_PATH, by_name=True, skip_mismatch=True)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    try:
        input_board_batch, board_2d_display, current_player = load_random_sample(DATA_DIR)
    except Exception as e:
        print(f"Data loading error: {e}")
        sys.exit(1)

    print("Running inference...")
    outputs = model.predict(input_board_batch)
    
    policy_logits = outputs[0][0]
    
    legal_moves = get_legal_moves(board_2d_display, current_player)
    print(f"Legal moves: {[index_to_coord(m) for m in legal_moves]}")
    
    if not legal_moves:
        print("No legal moves (Pass). Skipping visualization.")
        sys.exit(0)

    max_prob = -1.0
    best_move = -1
    
    for move in legal_moves:
        prob = policy_logits[move]
        if prob > max_prob:
            max_prob = prob
            best_move = move
            
    print(f"Best move selected by Policy: {index_to_coord(best_move)} with prob {max_prob:.4f}")
    
    if TARGET_BLOCK_INDEX == -1:
        attention_map_batch = outputs[-1]
        print("Visualizing the last Transformer block.")
    else:
        attention_map_batch = outputs[2 + TARGET_BLOCK_INDEX]
        print(f"Visualizing Transformer block index: {TARGET_BLOCK_INDEX}")

    attention_map_single = attention_map_batch[0] 

    plot_integrated_attention(board_2d_display, attention_map_single, best_move, TARGET_BLOCK_INDEX, policy_prob=max_prob)