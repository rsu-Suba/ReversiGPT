import sys
import os
import glob
import msgpack

def repair_msgpack_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found -> {file_path}")
        return

    print(f"--- Repairing: {file_path} ---")

    try:
        with open(file_path, 'rb') as f:
            malformed_data = msgpack.unpack(f, raw=False, use_list=True)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    if not isinstance(malformed_data, list):
        print("File does not contain a list at its root. Nothing to do.")
        return

    original_count = len(malformed_data)
    cleaned_data = []
    
    for item in malformed_data:
        if isinstance(item, dict):
            cleaned_data.append(item)

    cleaned_count = len(cleaned_data)

    if original_count == cleaned_count:
        print("File seems to be in the correct (or already repaired) flat format. No changes made.")
        return

    try:
        with open(file_path, 'wb') as f:
            msgpack.pack(cleaned_data, f)
        print("Repair successful!")
        print(f"Original item count: {original_count}")
        print(f"Cleaned item count (moves only): {cleaned_count}")
    except Exception as e:
        print(f"Error writing cleaned file {file_path}: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        target_dir = sys.argv[1]
        print(f"Processing all .msgpack files in directory: {target_dir}")
        msgpack_files = glob.glob(os.path.join(target_dir, '*.msgpack'))
        if not msgpack_files:
            print(f"No .msgpack files found in {target_dir}")
        else:
            for file_path in msgpack_files:
                repair_msgpack_file(file_path)
                print("-" * 20)
    elif len(sys.argv) > 1:
        for file_path in sys.argv[1:]:
            repair_msgpack_file(file_path)
            print("-" * 20)
    else:
        print("Usage: python tools/repair_msgpack.py <path_to_directory> or <path_to_msgpack_file_1> [<path_to_msgpack_file_2> ...]")
        sys.exit(1)
