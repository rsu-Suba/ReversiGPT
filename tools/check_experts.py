import sys
import os
import numpy as np
import tensorflow as tf
from keras import models, mixed_precision
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AI.models.model_selector import try_load_model
from AI.config_loader import load_config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    mixed_precision.set_global_policy('mixed_float16')
except:
    pass

def analyze_experts(model_path):
    print(f"Loading model: {model_path}")
    is_moe1 = "MoE-1" in model_path
    if is_moe1:
        config = load_config(type('Args', (), {'model': 'moe-1'}))
    else:
        config = load_config(type('Args', (), {'model': 'moe-2'}))

    model = try_load_model(model_path, config=config)
    arch_type = "unknown"
    moe_layers = []
    moe_blocks_v2 = [l for l in model.layers if 'moe_block' in l.name]
    if moe_blocks_v2:
        arch_type = "MoE-2"
        moe_layers = moe_blocks_v2
    
    if not moe_layers:
        for layer in model.layers:
            if "DynamicAssembly" in layer.__class__.__name__:
                moe_layers.append(layer)
        if moe_layers:
            arch_type = "MoE-1"

    print(f"Detected Architecture: {arch_type} ({len(moe_layers)} blocks)")
    
    if not moe_layers:
        print("Error: No MoE layers found.")
        return

    inspectors = []
    if arch_type == "MoE-2":
        for b_idx, block in enumerate(moe_layers):
            router_outputs = []
            seen_tensors = set()
            for l in block.layers:
                if 'slice_p' in l.name:
                    inp = l.input
                    if id(inp) not in seen_tensors:
                        router_outputs.append(inp)
                        seen_tensors.add(id(inp))
            
            if not router_outputs:
                softmax_layers = [l for l in block.layers if 'activation' in l.name and 'softmax' in str(l.activation).lower()]
                router_outputs = [l.output for l in softmax_layers]

            if not router_outputs:
                print(f"Warning: Could not find router outputs in Block {b_idx}")
                inspectors.append(None)
            else:
                inspectors.append(models.Model(inputs=block.input, outputs=router_outputs))

    def create_board(desc, stones):
        b = np.zeros((8, 8), dtype=np.float32)
        for r, c, w in stones:
            if w == 1: b[r, c] = 1 
            elif w == 2: b[r, c] = 2 
        me = (b == 1).astype(np.float32)
        opp = (b == 2).astype(np.float32)
        return np.stack([me, opp], axis=-1)

    patterns = [
        ("Initial", [
            (3,3,2), (3,4,1), (4,3,1), (4,4,2)
        ]),
        ("Corner (Me)", [
            (0,0,1), (0,1,2), (1,0,2), (1,1,2), (3,3,2), (3,4,1), (4,3,1), (4,4,2)
        ]),
        ("Edge (Me)", [
            (0,2,1), (0,3,1), (0,4,1), (3,3,2), (3,4,1), (4,3,1), (4,4,2)
        ]),
        ("Full (Late)", [
            (r,c, 1 if (r+c)%2==0 else 2) for r in range(8) for c in range(8) if not (r==0 and c==0)
        ])
    ]

    emb_model = None
    if arch_type == "MoE-2":
        emb_candidates = [l for l in model.layers if 'embedding_block' in l.name]
        if emb_candidates: emb_model = emb_candidates[0]

    print("\n=== Expert Usage Analysis ===")
    
    for name, stones in patterns:
        print(f"\n>> Pattern: {name}")
        board_input = tf.convert_to_tensor(np.expand_dims(create_board(name, stones), axis=0))
        if arch_type == "MoE-2":
            x = emb_model(board_input, training=False)
            for b_idx, (block, inspector) in enumerate(zip(moe_layers, inspectors)):
                if inspector:
                    step_probs_list = inspector(x, training=False)
                    print(f"  [Block {b_idx}]")
                    if not isinstance(step_probs_list, list): step_probs_list = [step_probs_list]
                    for s_idx, probs in enumerate(step_probs_list):
                        p = probs.numpy()[0]
                        p_str = " ".join([f"{val*100:4.1f}%" for val in p])
                        best = np.argmax(p)
                        print(f"    Step {s_idx}: {p_str}  (Best: {best})")
                x = block(x, training=False)

        elif arch_type == "MoE-1":
            _ = model(board_input, training=False)
            for b_idx, layer in enumerate(moe_layers):
                print(f"  [Block {b_idx}]")
                if hasattr(layer, 'last_probs') and layer.last_probs:
                    for s_idx, probs in enumerate(layer.last_probs):
                        if hasattr(probs, 'numpy'):
                            p = probs.numpy()[0]
                        else:
                            p = probs.numpy()[0]
                            
                        p_str = " ".join([f"{val*100:4.1f}%" for val in p])
                        best = np.argmax(p)
                        print(f"    Step {s_idx}: {p_str}  (Best: {best})")
                else:
                    print("    No prob data found.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "./models/TF/MoE-2.keras"
        
    analyze_experts(target)