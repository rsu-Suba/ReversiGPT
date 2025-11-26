
import msgpack
import sys

def inspect_msgpack(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = msgpack.unpackb(f.read())
        
        print(f"--- Inspection for: {file_path} ---")
        
        if isinstance(data, list):
            print("Root is a list.")
            if data:
                print("First element of the list:")
                print(data[0])
            return

        root_keys = data.keys()
        print(f"Root keys: {list(root_keys)}")
        
        if 'v' in root_keys:
            print(f"Root visit count (v): {data['v']}")
        if 'wi' in root_keys:
            print(f"Root wins (wi): {data['wi']}")
        if 'ch' in root_keys:
            num_children = len(data['ch'])
            print(f"Number of children (ch): {num_children}")
            
            total_child_visits = 0
            for child_key, child_node in data['ch'].items():
                if 'v' in child_node:
                    total_child_visits += child_node['v']
            print(f"Total visits in first-level children: {total_child_visits}")

    except Exception as e:
        print(f"Error reading or inspecting {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            inspect_msgpack(path)
    else:
        print("Please provide msgpack file paths as arguments.")

