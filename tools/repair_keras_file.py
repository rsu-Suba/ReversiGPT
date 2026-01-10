import os
import sys
import tensorflow as tf
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from AI.config_loader import load_config
from AI.models.model_selector import create_model, try_load_model

def repair(weights_path, save_path, model_name='heavy-moe-3'):
    print(f"--- Keras 3 Model Repair Tool ---")
    print(f"Input Weights: {weights_path}")
    print(f"Output Model:  {save_path}")
    print(f"Loading config for {model_name}...")
    config = load_config(type('Args', (), {'model': model_name}))
    
    print(f"Building {config['arch']} model structure...")
    model = create_model(config)
    dummy_in = tf.zeros((1, 8, 8, 2))
    model(dummy_in)
    
    print(f"Restoring weights from {weights_path}...")
    try:
        model.load_weights(weights_path, skip_mismatch=True)
        print("Restored by name.")
    except:
        print("Name mismatch detected. Falling back to positional restore...")
        model.load_weights(weights_path, by_name=False, skip_mismatch=True)
        print("Restored by position.")
    
    print(f"Saving full model to {save_path}...")
    model.save(save_path)
    print("Repair complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="heavy-moe-3")
    args = parser.parse_args()
    
    repair(args.weights, args.output, args.model)