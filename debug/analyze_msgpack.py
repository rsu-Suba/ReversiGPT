import numpy as np
import msgpack
import sys

def analyze_msgpack_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            packed_data = f.read()
        data = msgpack.unpackb(packed_data, raw=False)

        if not isinstance(data, list):
            print(f"Error: Expected a list of training samples in {file_path}, but got {type(data)}")
            return

        total_legal_moves = 0
        num_samples = len(data)

        for sample in data:
            if 'policy' in sample and isinstance(sample['policy'], list):
                legal_moves = np.count_nonzero(sample['policy'])
                total_legal_moves += legal_moves

        average_legal_moves = total_legal_moves / num_samples if num_samples > 0 else 0

        print(f"--- Analysis for: {file_path} ---")
        print(f"Number of training samples: {num_samples}")
        print(f"Total legal moves found: {total_legal_moves}")
        print(f"Average legal moves per sample: {average_legal_moves:.2f}")

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for file_path in sys.argv[1:]:
            analyze_msgpack_data(file_path)
    else:
        print("Usage: python analyze_msgpack.py <path_to_msgpack_file_1> [<path_to_msgpack_file_2> ...]")