
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Database.transformerModel import PositionalEncoding
from reversi_bitboard_cpp import ReversiBitboard

MODEL_PATH = './Database/models/Transformer/1G_trained.h5'
# You can change this to visualize different layers (e.g., 'multi_head_attention', 'multi_head_attention_1', etc.)
# Check model.summary() for the exact names.
TARGET_ATTENTION_LAYER_NAME = 'multi_head_attention_2' # Assuming the last layer
HEAD_TO_VISUALIZE = 3 # Visualize the first attention head (0 to num_heads-1)

def board_to_input_planes(board_1d, current_player):
    """Converts a 1D board numpy array to the model's input format."""
    player_plane = np.zeros((8, 8), dtype=np.float32)
    opponent_plane = np.zeros((8, 8), dtype=np.float32)
    board_2d = board_1d.reshape((8, 8))
    
    player_plane[board_2d == current_player] = 1.0
    opponent_plane[board_2d == 3 - current_player] = 1.0
    
    return np.stack([player_plane, opponent_plane], axis=-1)

def plot_attention_heatmap(board_1d, attention_scores, from_square_index, title):
    """
    Displays the board state and an attention heatmap for a specific square.
    - board_1d: 1D numpy array (64,) representing the board.
    - attention_scores: Numpy array (64, 64) of attention scores for a specific head.
    - from_square_index: The square (0-63) from which attention is being visualized.
    """
    board_2d = board_1d.reshape(8, 8)
    attention_for_square = attention_scores[from_square_index, :].reshape(8, 8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- 1. Board State ---
    board_repr = np.full((8, 8), ' ', dtype=str)
    board_repr[board_2d == 1] = '●' # Black
    board_repr[board_2d == 2] = '○' # White
    
    sns.heatmap(np.zeros((8,8)), annot=board_repr, fmt='', cbar=False, cmap=['#006400'], 
                linewidths=0.5, linecolor='black', ax=ax1, square=True)
    ax1.set_title("Current Board State")

    # Highlight the source square
    row, col = from_square_index // 8, from_square_index % 8
    ax1.add_patch(plt.Rectangle((col, row), 1, 1, fill=False, edgecolor='yellow', lw=3))

    # --- 2. Attention Heatmap ---
    sns.heatmap(attention_for_square, annot=True, fmt=".2f", cmap="viridis", 
                ax=ax2, square=True)
    ax2.set_title(title)
    
    plt.tight_layout()
    plt.show()

def main():
    print(f"--- Loading Transformer Model from {MODEL_PATH} ---")
    try:
        with tf.keras.utils.custom_object_scope({'PositionalEncoding': PositionalEncoding}):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n--- Model Summary ---")
    model.summary()

    # --- Create a new model to output attention scores ---
    try:
        target_layer = model.get_layer(TARGET_ATTENTION_LAYER_NAME)
    except ValueError:
        print(f"\nError: Layer '{TARGET_ATTENTION_LAYER_NAME}' not found in the model.")
        print("Please check the layer names in the summary above and update TARGET_ATTENTION_LAYER_NAME.")
        return

    # The .input property of a layer gives its symbolic input tensor(s).
    # For MultiHeadAttention, this is a list [query, key, value...].
    # In our self-attention case, query and key are the same tensor from the previous layer.
    if not isinstance(target_layer.input, list) or len(target_layer.input) < 1:
        print(f"Error: Could not determine input tensor for layer {TARGET_ATTENTION_LAYER_NAME}")
        return
    input_tensor = target_layer.input[0]

    # Call the layer directly on its input tensor, forcing it to return scores.
    _, attention_scores = target_layer(input_tensor, input_tensor, return_attention_scores=True)

    # Create a new model that maps the original model's input to these attention scores.
    attention_model = tf.keras.Model(inputs=model.input, outputs=attention_scores)
    print(f"\nSuccessfully created model to extract attention from '{TARGET_ATTENTION_LAYER_NAME}'.")

    # --- Prepare a sample board ---
    game = ReversiBitboard()
    # You can apply some moves to get to an interesting position
    # game.apply_move(19) 
    # game.apply_move(18)
    
    board_np = game.board_to_numpy()
    current_player = game.current_player
    
    model_input = np.expand_dims(board_to_input_planes(board_np, current_player), axis=0)

    # --- Get and Visualize Attention ---
    print("\n--- Calculating Attention Scores ---")
    scores = attention_model.predict(model_input)
    
    # scores shape is (batch, num_heads, from_seq, to_seq) -> (1, 4, 64, 64)
    print(f"Output scores shape: {scores.shape}")
    
    # Select a square to visualize attention FROM
    # (e.g., a corner, an edge, a player's piece)
    from_square = 27 # An example square (D4)
    
    attention_head_scores = scores[0, HEAD_TO_VISUALIZE, :, :]
    
    plot_attention_heatmap(
        board_np, 
        attention_head_scores, 
        from_square,
        title=f"Attention from Square {from_square} (Head {HEAD_TO_VISUALIZE})"
    )

if __name__ == '__main__':
    main()
