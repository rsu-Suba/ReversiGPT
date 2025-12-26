import os
import sys
import numpy as np
import h5py

def main():
    model_path = "models/TF/Nano-25K-B3.h5"
    output_path = "submission/weights.bin"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    weights = []
    
    try:
        with h5py.File(model_path, 'r') as f:
            all_weights_flat = []
            
            def visit_func(name, node):
                if isinstance(node, h5py.Dataset):
                    if "optimizer_weights" in name: return
                    all_weights_flat.append((name, node[()]))

            f.visititems(visit_func)
            pool = {k: v for k, v in all_weights_flat}
            
            def pop_w_pattern(pattern, shape, reshape_to=None):
                patterns = [pattern] if isinstance(pattern, str) else pattern
                candidates = []
                for k, v in pool.items():
                    if v.shape == shape:
                        for p in patterns:
                            if p in k:
                                candidates.append(k)
                                break
                
                candidates.sort()
                
                if not candidates:
                    print(f"Error: No weight found with pattern '{patterns}' and shape {shape}")
                    return np.zeros(reshape_to if reshape_to else shape)
                
                best_k = candidates[0]
                val = pool[best_k]
                del pool[best_k]
                
                if reshape_to: val = val.reshape(reshape_to)
                return val

            weights.append(pop_w_pattern('dense/dense/kernel', (2, 32)))
            weights.append(pop_w_pattern('dense/dense/bias', (32,)))
            
            weights.append(pop_w_pattern('row_emb', (8, 32)))
            weights.append(pop_w_pattern('col_emb', (8, 32)))
            
            for _ in range(3):
                # MHA
                weights.append(pop_w_pattern('query/kernel', (32, 4, 8), (32, 32)))
                weights.append(pop_w_pattern('query/bias', (4, 8), (32,)))
                
                weights.append(pop_w_pattern('key/kernel', (32, 4, 8), (32, 32)))
                weights.append(pop_w_pattern('key/bias', (4, 8), (32,)))
                
                weights.append(pop_w_pattern('value/kernel', (32, 4, 8), (32, 32)))
                weights.append(pop_w_pattern('value/bias', (4, 8), (32,)))
                
                weights.append(pop_w_pattern('attention_output/kernel', (4, 8, 32), (32, 32)))
                weights.append(pop_w_pattern('attention_output/bias', (32,), (32,)))
                
                # LN1
                weights.append(pop_w_pattern('gamma', (32,)))
                weights.append(pop_w_pattern('beta', (32,)))
                
                # FFN1
                weights.append(pop_w_pattern('sequential', (32, 128)))
                weights.append(pop_w_pattern('sequential', (128,)))
                
                # FFN2
                weights.append(pop_w_pattern('sequential', (128, 32)))
                weights.append(pop_w_pattern('sequential', (32,)))
                
                # LN2
                weights.append(pop_w_pattern('gamma', (32,)))
                weights.append(pop_w_pattern('beta', (32,)))
                
            weights.append(pop_w_pattern('policy_logits/kernel', (32, 1)))
            weights.append(pop_w_pattern('policy_logits/bias', (1,)))
            
            weights.append(pop_w_pattern('value/kernel', (32, 1)))
            weights.append(pop_w_pattern('value/bias', (1,)))
            
    except Exception as e:
        print(f"Error inspecting h5: {e}")
        import traceback
        traceback.print_exc()
        return

    with open(output_path, "wb") as f:
        for w in weights:
            f.write(w.astype(np.float16).tobytes())
            
    print(f"Exported {len(weights)} weight tensors to {output_path}")

if __name__ == "__main__":
    main()