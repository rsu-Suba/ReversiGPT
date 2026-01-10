import os
import sys
import tensorflow as tf
import numpy as np
import glob
import multiprocessing
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AI.config import TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR, NUM_PARALLEL_GAMES

MIN_TOTAL_STONES = 50
MIN_STONE_RATIO = 0.7 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

def _parse_function(example_proto):
    feature_description = {
        'input_planes': tf.io.FixedLenFeature([], tf.string),
        'policy': tf.io.FixedLenFeature([], tf.string),
        'value': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    return parsed['input_planes'], parsed['policy'], parsed['value']

def serialize_sample(input_planes_bytes, policy_bytes, value):
    feature = {
        'input_planes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_planes_bytes.numpy()])),
        'policy': tf.train.Feature(bytes_list=tf.train.BytesList(value=[policy_bytes.numpy()])),
        'value': tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def filter_file(args):
    input_path, output_path = args
    count = 0
    try:
        dataset = tf.data.TFRecordDataset(input_path)
        results = []
        for record in dataset:
            input_bytes, policy_bytes, value = _parse_function(record)
            input_tensor = tf.io.parse_tensor(input_bytes, out_type=tf.float32)
            input_tensor = tf.reshape(input_tensor, (8, 8, 2))
            my_stones = tf.reduce_sum(input_tensor[:, :, 0])
            opp_stones = tf.reduce_sum(input_tensor[:, :, 1])
            total = my_stones + opp_stones
            
            ratio = my_stones / (total + 1e-9)
            
            if value > 0.5 and total >= MIN_TOTAL_STONES and ratio >= MIN_STONE_RATIO:
                results.append((input_bytes, policy_bytes, value))
        
        if results:
            with tf.io.TFRecordWriter(output_path) as writer:
                for r in results:
                    writer.write(serialize_sample(r[0], r[1], r[2]))
                    count += 1
        return count
    except Exception as e:
        return 0

def main():
    if sys.platform != 'win32':
        multiprocessing.set_start_method('fork', force=True)
        
    print(f"--- God-Tier Game Filter (Ratio >= {MIN_STONE_RATIO:.0%}) ---")
    
    src_dir = os.path.join(TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR, 'tfrecords')
    dst_dir = os.path.join(TRAINING_DATA_DIR, '3G_god', 'tfrecords')
    
    num_workers = min(NUM_PARALLEL_GAMES, 16)
    print(f"Using {num_workers} workers for filtering.")

    for split in ['train', 'val']:
        s_path = os.path.join(src_dir, split)
        d_path = os.path.join(dst_dir, split)
        os.makedirs(d_path, exist_ok=True)
        
        for f in glob.glob(os.path.join(d_path, "*.tfrecord")):
            os.remove(f)
        
        files = glob.glob(os.path.join(s_path, '*.tfrecord'))
        if not files:
            print(f"No files in {s_path}")
            continue
        
        print(f"Processing {split}: {len(files)} files...")
        tasks = [(f, os.path.join(d_path, os.path.basename(f))) for f in files]
        
        total_kept = 0
        with multiprocessing.Pool(num_workers) as pool:
            for kept in tqdm(pool.imap_unordered(filter_file, tasks), total=len(tasks), desc=f"Filtering {split}"):
                total_kept += kept
            
        print(f"  Finished {split}: Kept {total_kept} god-tier samples.")

if __name__ == "__main__":
    main()
