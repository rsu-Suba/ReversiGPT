import h5py
import sys

def inspect_h5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print("Weights in H5:")
            def print_attrs(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"{name}: {obj.shape}")
            f.visititems(print_attrs)
    except Exception as e:
        print(f"Error inspecting {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_h5(sys.argv[1])
    else:
        inspect_h5("models/TF/Nano-25K.h5")
