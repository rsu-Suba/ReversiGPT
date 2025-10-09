import os
import msgpack
import numpy as np
import collections
import sys

def analyze_board_similarity(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return

    msgpack_files = [f for f in os.listdir(directory_path) if f.endswith('.msgpack')]
    if not msgpack_files:
        print(f"No .msgpack files found in {directory_path}")
        return

    unique_boards = set()
    game_phase_counts = collections.Counter()
    total_states = 0

    print(f"--- Analyzing Board Similarity in {directory_path} ---")

    for filename in msgpack_files:
        file_path = os.path.join(directory_path, filename)
        try:
            with open(file_path, 'rb') as f:
                game_data = msgpack.unpack(f, raw=False)
            
            for record in game_data:
                total_states += 1
                # Create a unique representation for the board state
                board_key = (record['black_board'], record['white_board'], record['player'])
                unique_boards.add(board_key)

                # Analyze game phase based on total number of pieces
                black_pieces = bin(record['black_board']).count('1')
                white_pieces = bin(record['white_board']).count('1')
                total_pieces = black_pieces + white_pieces

                if total_pieces < 20:
                    game_phase_counts['Opening'] += 1
                elif total_pieces < 50:
                    game_phase_counts['Mid-game'] += 1
                else:
                    game_phase_counts['Endgame'] += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n--- Uniqueness Analysis ---")
    uniqueness_ratio = (len(unique_boards) / total_states) * 100 if total_states > 0 else 0
    print(f"Total Game States: {total_states}")
    print(f"Unique Game States: {len(unique_boards)}")
    print(f"Uniqueness Ratio: {uniqueness_ratio:.2f}%\n")

    print("--- Game Phase Distribution ---")
    if not game_phase_counts:
        print("No game phase data found.")
    else:
        for phase, count in sorted(game_phase_counts.items()):
            percentage = (count / total_states) * 100 if total_states > 0 else 0
            print(f"  {phase}: {count} states ({percentage:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
    else:
        target_directory = './Database/training_data/1G/'
        print(f"No directory provided. Using default: {target_directory}")
    
    analyze_board_similarity(target_directory)
