import numpy as np
import tensorflow as tf
from AI.models.model_selector import try_load_model
from AI.config_loader import load_config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = load_config()
print("--- AI Debugger: Input Interpretation Analysis ---")

def load_ai(model_path, name):
    print(f"Loading {name} from {model_path}...")
    try:
        model = try_load_model(model_path, config=config)
        return model
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        return None

moe1 = load_ai("./models/TF/MoE-1.h5", "MoE-1 (Teacher)")
moe2 = load_ai("./models/TF/MoE-2.keras", "MoE-2 (Student)")

def create_board_tensor(black_pos, white_pos, current_player, flip_channels=False, transpose=False):
    board = np.zeros((8, 8), dtype=np.int8)
    for p in black_pos: board[p // 8, p % 8] = 1
    for p in white_pos: board[p // 8, p % 8] = 2
    
    if transpose:
        board = board.T

    me = 1.0 if current_player == 1 else 0.0
    
    p1_mask = (board == 1).astype(np.float32)
    p2_mask = (board == 2).astype(np.float32)
    
    if current_player == 1:
        me_plane = p1_mask
        opp_plane = p2_mask
    else:
        me_plane = p2_mask
        opp_plane = p1_mask
        
    if flip_channels:
        planes = np.stack([opp_plane, me_plane], axis=-1)
    else:
        planes = np.stack([me_plane, opp_plane], axis=-1)
        
    return tf.convert_to_tensor(np.expand_dims(planes, axis=0))

def get_prediction(model, tensor):
    if model is None: return np.zeros(64), 0.0
    out = model(tensor, training=False)
    if isinstance(out, list):
        p, v = out[0], out[1]
    else:
        p, v = out[0], 0.0
    return p.numpy()[0], v.numpy()[0][0]

def analyze_case(case_name, black, white, current, legal_moves):
    print(f"\n[{case_name}] Turn: {'Black' if current==1 else 'White'}")
    print(f"  Stones: B={len(black)}, W={len(white)} | Legal: {legal_moves}")
    
    variations = [
        ("Standard", False, False),
        ("Ch Swap", True, False),
        ("Transposed", False, True),
        ("Swap+Trans", True, True)
    ]

    print(f"  {'Variation':<12} | {'Model':<6} | {'Best':<4} | {'Value':<7} | {'Legal Prob':<8} | {'Top-3'}")
    print("-" * 70)

    for v_name, flip, trans in variations:
        tensor = create_board_tensor(black, white, current, flip, trans)
        
        for name, model in [("M1", moe1), ("M2", moe2)]:
            if model is None: continue
            p, v = get_prediction(model, tensor)
            best_move = np.argmax(p)
            legal_prob_sum = sum([p[m] for m in legal_moves])
            top3 = np.argsort(p)[::-1][:3]
            top3_str = " ".join([str(x) for x in top3])
            is_legal = best_move in legal_moves
            mark = " " if is_legal else "!"
            
            print(f"  {v_name:<12} | {name:<6} | {best_move:>3}{mark} | {v:7.4f} | {legal_prob_sum:7.1%}  | {top3_str}")

case1_black = [27, 36]
case1_white = [28, 35]
case1_legal = [19, 26, 37, 44]
analyze_case("Opening", case1_black, case1_white, 1, case1_legal)

case2_black = [27]
case2_white = [28, 29, 30, 31]
case2_legal = [26]
case2_black = [27]
case2_white = [28, 29]
case2_legal = [30]
analyze_case("Line Capture", case2_black, case2_white, 1, case2_legal)

print("\n--- Diagnosis Finished ---")
