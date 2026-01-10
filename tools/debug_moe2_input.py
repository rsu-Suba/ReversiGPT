import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AI.models.model_selector import try_load_model
from AI.config_loader import load_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_base_board():
    board = np.zeros((8, 8), dtype=np.float32)
    board[3, 3] = 2
    board[3, 4] = 1
    board[4, 3] = 1
    board[4, 4] = 2
    return board

def create_input_variations(board, current_player=1):
    variations = {}
    me = (board == current_player).astype(np.float32)
    opp = (board == (3 - current_player)).astype(np.float32)
    
    v1 = np.stack([me, opp], axis=-1)
    variations["Normal [Me, Opp]"] = v1
    v2 = np.transpose(v1, (1, 0, 2))
    variations["Transposed [Me, Opp]"] = v2
    v3 = np.stack([opp, me], axis=-1)
    variations["Ch Swap [Opp, Me]"] = v3
    v4 = np.transpose(v3, (1, 0, 2))
    variations["Swap+Trans [Opp, Me]"] = v4
    
    return variations

def main():
    print("--- Debugging MoE-2 Input Interpretation ---")
    config = load_config(type('Args', (), {'model': 'moe-2'}))
    print(f"Loaded config: {config['model_name']} (Arch: {config.get('arch')})")

    model_path = "./models/TF/MoE-2.keras"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    try:
        print(f"Loading {model_path}...")
        model = try_load_model(model_path, config=config)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    legal_moves = [19, 26, 37, 44]
    
    print("\nTest Case: Initial Board (Black to move)")
    print(f"Expected Legal Moves: {legal_moves}")
    
    board = create_base_board()
    variations = create_input_variations(board, current_player=1)
    
    best_pattern = None
    best_legal_prob = -1.0

    print(f"\n{'Pattern Name':<25} | {'Best Move':<10} | {'Legal Prob Sum':<15} | {'Val':<6} | {'Top-3 Moves'}")
    print("-" * 80)

    for name, input_np in variations.items():
        input_tensor = tf.convert_to_tensor(np.expand_dims(input_np, axis=0))
        
        out = model(input_tensor, training=False)
        p = out[0].numpy()[0]
        v = out[1].numpy()[0][0]
        
        best_move = np.argmax(p)
        legal_prob_sum = sum([p[m] for m in legal_moves])
        
        top3 = np.argsort(p)[::-1][:3]
        
        mark = "OK" if best_move in legal_moves else "NG"
        print(f"{name:<25} | {best_move:>2} ({mark})    | {legal_prob_sum:7.1%}        | {v:6.3f} | {top3}")
        
        if legal_prob_sum > best_legal_prob:
            best_legal_prob = legal_prob_sum
            best_pattern = name

    print("-" * 80)
    print(f"Most likely correct pattern: {best_pattern} (Legal Prob: {best_legal_prob:.1%})")

if __name__ == "__main__":
    main()