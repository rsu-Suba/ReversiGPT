import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import msgpack

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI.config import TRAINING_DATA_DIR, INSPECT_GENERATION_SUBDIR

def inspect_msgpack_structure(file_path):
    """
    Inspects the structure of a single msgpack file.
    Assumes file contains one packed object which is a list of either games or moves.
    """
    try:
        with open(file_path, 'rb') as f:
            data = msgpack.unpack(f, raw=False, use_list=True)
        
        print(f"--- Structure Inspection for: {file_path} ---")
        
        if isinstance(data, list):
            print(f"Root is a list. Length: {len(data)}")
            if data:
                first_element = data[0]
                if isinstance(first_element, list):
                    print(f"  First element is a list (likely a game history). Length: {len(first_element)}")
                    if first_element:
                        print(f"    First move of first game: {str(first_element[0])[:100]}...")
                elif isinstance(first_element, dict):
                    print(f"  First element is a dict (likely a move record). Keys: {list(first_element.keys())}")
                    print(f"  First move record: {str(first_element)[:100]}...")
                else:
                    print(f"  First element type: {type(first_element)}, Data: {str(first_element)[:100]}...")
        elif isinstance(data, dict):
            print(f"Root is a dict. Keys: {list(data.keys())}")
        else:
            print(f"Root data type: {type(data)}, Data preview: {str(data)[:200]}...")

    except Exception as e:
        print(f"Error reading or inspecting {file_path}: {e}")

def plot_heatmap(data, title, ax):
    """Helper function to plot a single heatmap."""
    if data is None or np.all(data == 0):
        ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')
        return

    v_abs_max = np.max(np.abs(data))
    if v_abs_max == 0:
        v_abs_max = 0.01

    sns.heatmap(data, annot=True, fmt=".3f", cmap="coolwarm",
                vmin=-v_abs_max, vmax=v_abs_max, center=0.0, square=True, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')

def analyze_msgpack_game_data(msgpack_dir):
    """
    Analyzes msgpack files containing game history records
    to create aggregated policy-value heatmaps for different phases of the game.
    Can handle both structured (list of games) and flat (list of moves) formats.
    """
    if not os.path.isdir(msgpack_dir):
        print(f"Error: Directory not found: {msgpack_dir}")
        return

    msgpack_files = glob.glob(os.path.join(msgpack_dir, '*.msgpack'))
    if not msgpack_files:
        print(f"Error: No .msgpack files found in {msgpack_dir}")
        return

    print(f"Found {len(msgpack_files)} msgpack files in {msgpack_dir}")

    heatmap_early, count_early = np.zeros((8, 8), dtype=np.float64), 0
    heatmap_mid, count_mid = np.zeros((8, 8), dtype=np.float64), 0
    heatmap_late, count_late = np.zeros((8, 8), dtype=np.float64), 0
    
    total_games_processed_structured = 0
    total_games_processed_reconstructed = 0
    
    initial_board_state = [0]*64
    initial_board_state[27] = 1
    initial_board_state[28] = 2
    initial_board_state[35] = 2
    initial_board_state[36] = 1

    printed_first_board_debug = False

    print("\n--- Analyzing full dataset for aggregated heatmaps by game phase ---")
    for msgpack_path in tqdm(msgpack_files, desc="Analyzing msgpack files"):
        try:
            with open(msgpack_path, 'rb') as f:
                all_data_in_file = msgpack.unpack(f, raw=False, use_list=True)

            if not all_data_in_file:
                continue

            is_structured_format = isinstance(all_data_in_file[0], list)

            if is_structured_format:
                all_games = all_data_in_file
                total_games_processed_structured += len(all_games)
                
                for game_history in all_games:
                    if not game_history: continue
                    for turn, record in enumerate(game_history):
                        if 'policy' not in record or 'value' not in record or record['policy'] is None or record['value'] is None:
                            continue
                        
                        policy = np.array(record['policy'], dtype=np.float32).reshape(8, 8)
                        value = float(record['value'])
                        policy_value = policy

                        if 0 <= turn < 20:
                            heatmap_early += policy_value
                            count_early += 1
                        elif 20 <= turn < 40:
                            heatmap_mid += policy_value
                            count_mid += 1
                        elif 40 <= turn < 64:
                            heatmap_late += policy_value
                            count_late += 1
            else:
                if not printed_first_board_debug and all_data_in_file:
                    first_record_board = all_data_in_file[0].get('board')
                    if first_record_board is not None:
                        print(f"\n[Debug] First move's board state from a flat file (should match initial_board_state): {first_record_board}")
                        printed_first_board_debug = True
                
                game_turn = -1
                
                for record in all_data_in_file:
                    if 'board' not in record or 'policy' not in record or 'value' not in record or record['policy'] is None or record['value'] is None:
                        continue

                    current_board_list = record['board']
                    
                    if current_board_list == initial_board_state:
                        game_turn = 0
                        total_games_processed_reconstructed += 1
                    
                    if game_turn == -1:
                        continue

                    policy = np.array(record['policy'], dtype=np.float32).reshape(8, 8)
                    value = float(record['value'])
                    policy_value = policy * value

                    if 0 <= game_turn < 20:
                        heatmap_early += policy_value
                        count_early += 1
                    elif 20 <= game_turn < 40:
                        heatmap_mid += policy_value
                        count_mid += 1
                    elif 40 <= game_turn < 64:
                        heatmap_late += policy_value
                        count_late += 1
                    
                    game_turn += 1

        except Exception as e:
            print(f"Error processing file {os.path.basename(msgpack_path)}: {e}")

    print(f"\n[Debug] Total games processed from structured files: {total_games_processed_structured}")
    print(f"[Debug] Total games reconstructed from flat files: {total_games_processed_reconstructed}")

    heatmaps_to_plot, titles_to_plot = [], []
    for (heatmap, count, phase_name) in [
        (heatmap_early, count_early, "Early Game (Moves 1-20)"),
        (heatmap_mid, count_mid, "Mid Game (Moves 21-40)"),
        (heatmap_late, count_late, "Late Game (Moves 41-64)")
    ]:
        if count > 0:
            heatmaps_to_plot.append(heatmap / count)
            titles_to_plot.append(f"{phase_name}\n({count} samples)")
        else:
            heatmaps_to_plot.append(None)
            titles_to_plot.append(f"{phase_name}\n(No Data)")


    print("\n--- Overall Policy-Value Heatmaps by Game Phase ---")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('Aggregated Policy-Value Across All Games by Phase', fontsize=16)

    for i, (heatmap_data, title) in enumerate(zip(heatmaps_to_plot, titles_to_plot)):
        plot_heatmap(heatmap_data, title, axes[i])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--structure":
            if len(sys.argv) < 3:
                print("Usage: python inspect_msgpack.py --structure <path_to_msgpack_file_1> [<path_to_msgpack_file_2> ...]")
                sys.exit(1)
            for path in sys.argv[2:]:
                inspect_msgpack_structure(path)
        elif sys.argv[1] == "--analyze-game-data":
            if len(sys.argv) < 3:
                print("Usage: python inspect_msgpack.py --analyze-game-data <directory_path>")
                sys.exit(1)
            target_dir_for_analysis = sys.argv[2]
            analyze_msgpack_game_data(target_dir_for_analysis)
        else:
            print("Invalid argument. Use --structure for file inspection or --analyze-game-data for directory analysis.")
            sys.exit(1)
    else:
        print("No arguments provided. Defaulting to analyze game data for the current generation.")
        target_dir = os.path.join(TRAINING_DATA_DIR, INSPECT_GENERATION_SUBDIR)
        analyze_msgpack_game_data(target_dir)