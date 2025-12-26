import h5py
import sys

def main():
    model_path = "models/TF/MoE-1.h5"
    
    with h5py.File(model_path, 'r') as f:
        def visit_func(name, node):
            if isinstance(node, h5py.Dataset):
                print(f"{name}: {node.shape}")
        
        print("Weights in H5:")
        f.visititems(visit_func)

if __name__ == "__main__":
    main()
