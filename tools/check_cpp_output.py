import os
import sys
import numpy as np
import msgpack

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reversi_bitboard_cpp import process_msgpack_file

def check_samples(msgpack_path):
    print(f"Checking samples from: {msgpack_path}")
    try:
        samples = process_msgpack_file(msgpack_path)
        print(f"Total samples: {len(samples)}")

        nan_found = False
        for i, sample in enumerate(samples):
            input_planes = sample['input_planes']
            policy = sample['policy']
            value = sample['value']

            if np.isnan(input_planes).any() or np.isinf(input_planes).any():
                print(f"  Sample {i}: NaN/Inf in input_planes")
                print(f"    Input Planes:\n{input_planes}")
                nan_found = True
            if np.isnan(policy).any() or np.isinf(policy).any():
                print(f"  Sample {i}: NaN/Inf in policy")
                print(f"    Policy:\n{policy}")
                nan_found = True
            if np.isnan(value) or np.isinf(value):
                print(f"  Sample {i}: NaN/Inf in value")
                print(f"    Value: {value}")
                nan_found = True
        
        if not nan_found:
            print("  No NaN/Inf found in any sample.")

    except Exception as e:
        print(f"Error processing {msgpack_path}: {e}")

if __name__ == "__main__":
    msgpack_file_to_check = "./Database/training_data/9G/mcts_tree_60.msgpack" 
    check_samples(msgpack_file_to_check)