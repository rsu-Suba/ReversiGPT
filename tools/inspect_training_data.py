import os
import sys
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the directory path from config.py
from AI.config import TRAINING_DATA_DIR, INSPECT_GENERATION_SUBDIR, NUMS_TO_SHOW

def parse_tfrecord_fn(example_proto):
    """Same parser as in trainModel.py."""
    feature_description = {
        'input_planes': tf.io.FixedLenFeature([], tf.string),
        'policy': tf.io.FixedLenFeature([], tf.string),
        'value': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    input_planes = tf.io.parse_tensor(parsed_features['input_planes'], out_type=tf.float32)
    policy = tf.io.parse_tensor(parsed_features['policy'], out_type=tf.float32)
    value = parsed_features['value']

    input_planes = tf.reshape(input_planes, (8, 8, 2))
    policy = tf.reshape(policy, (64,))

    return {'input_planes': input_planes, 'policy': policy, 'value': value}

def display_sample(sample, index):
    """Displays a single data sample, combining policy and value for a unified heatmap."""
    board = sample['input_planes'].numpy()
    policy_np = sample['policy'].numpy()
    value = sample['value'].numpy()

    # Combine policy and value: policy_value will range from -1 to 1.
    # A high positive value means a move was favored in a winning game.
    # A high negative value means a move was favored in a losing game.
    policy_value = policy_np.reshape(8, 8) * value

    player_plane = board[:, :, 0]
    opponent_plane = board[:, :, 1]

    print(f"--- Random Sample {index + 1} ---")
    print(f"Game Outcome from this player's perspective: {value:.4f}")

    print("\nBoard State (X = Player, O = Opponent):")
    board_repr = np.full((8, 8), ' .')
    board_repr[player_plane == 1] = ' X'
    board_repr[opponent_plane == 1] = ' O'
    for row in board_repr:
        print("".join(row))

    print("\nPolicy-Value Heatmap (Red: Win-favored, Blue: Loss-favored):")
    plt.figure(figsize=(7, 6))
    # Use a diverging colormap and fix the scale from -1 to 1
    sns.heatmap(policy_value, annot=True, fmt=".2f", cmap="coolwarm", 
                vmin=-1, vmax=1, cbar=True, square=True)
    plt.title(f"Policy-Value Heatmap for Sample {index + 1} (Game Value: {value:.2f})")
    plt.show()
    print("\n" + "="*30 + "\n")

def analyze_tfrecord_data(tfrecord_dir, num_samples_to_display=5):
    """
    Analyzes TFRecord files, displays random samples, overall statistics,
    and an aggregated heatmap of policy-value.
    """
    if not os.path.isdir(tfrecord_dir):
        print(f"Error: Directory specified in config.py not found: {tfrecord_dir}")
        return

    tfrecord_files = glob.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
    if not tfrecord_files:
        print(f"Error: No .tfrecord files found in {tfrecord_dir}")
        return

    print(f"Found {len(tfrecord_files)} TFRecord files in {tfrecord_dir}")

    dataset = tf.data.TFRecordDataset(tfrecord_files).map(parse_tfrecord_fn)

    # --- Calculate Overall Statistics and Aggregate Heatmap Data ---
    print("\n--- Analyzing full dataset for statistics and aggregated heatmap ---")
    total_count = 0
    value_sum = 0
    win_count = 0
    loss_count = 0
    draw_count = 0
    aggregated_heatmap = np.zeros((8, 8), dtype=np.float32)

    for sample in tqdm(dataset, desc="Analyzing dataset"):
        policy_np = sample['policy'].numpy()
        value = sample['value'].numpy()
        
        # Aggregate heatmap data
        aggregated_heatmap += policy_np.reshape(8, 8) * value
        
        # Aggregate statistics
        value_sum += value
        total_count += 1
        if value == 1.0:
            win_count += 1
        elif value == -1.0:
            loss_count += 1
        else:
            draw_count += 1

    if total_count == 0:
        print("Dataset is empty. Cannot display anything.")
        return

    # Normalize the aggregated heatmap
    aggregated_heatmap /= total_count
    aggregated_heatmap *= 100

    print("\n--- Overall Value/Outcome Distribution ---")
    print(f"Total States: {total_count}")
    print(f"Average Value: {value_sum / total_count:.4f}")
    print(f"Wins (Value=1.0): {win_count} ({win_count / total_count:.2%})")
    print(f"Losses (Value=-1.0): {loss_count} ({loss_count / total_count:.2%})")
    print(f"Draws/Other: {draw_count} ({draw_count / total_count:.2%})")

    # --- Display Random Samples ---
    print(f"\n--- Displaying {num_samples_to_display} Random Samples ---")
    shuffled_dataset = dataset.shuffle(buffer_size=total_count)
    for i, sample in enumerate(shuffled_dataset.take(num_samples_to_display)):
        display_sample(sample, i)

    # --- Display Aggregated Heatmap ---
    print("\n--- Overall Policy-Value Heatmap ---")
    print("This heatmap shows the average promise of each square across all games.")
    print("Red = Consistently leads to wins. Blue = Consistently leads to losses (traps).")
    plt.figure(figsize=(8, 7))
    # Manually set the color range to be wider, e.g., -0.01 to 0.01
    # This makes the color differences more pronounced even for small values.
    # You can adjust vmin and vmax to best suit your data's scale.
    v_min, v_max = -0.5, 0.5
    sns.heatmap(aggregated_heatmap, annot=True, fmt=".3f", cmap="coolwarm", 
                vmin=v_min, vmax=v_max, center=0.0, square=True)
    plt.title("Aggregated Policy-Value Across All Games")
    plt.show()

if __name__ == "__main__":
    # Construct the full path from config values
    target_dir = os.path.join(TRAINING_DATA_DIR, INSPECT_GENERATION_SUBDIR, 'tfrecords', 'train')
    
    num_to_show = NUMS_TO_SHOW # Default number of samples to show
    if len(sys.argv) > 1:
        try:
            num_to_show = int(sys.argv[1])
            print(f"Will display {num_to_show} random samples.")
        except ValueError:
            print(f"Invalid argument. Using default of {num_to_show} samples.")

    analyze_tfrecord_data(target_dir, num_samples_to_display=num_to_show)