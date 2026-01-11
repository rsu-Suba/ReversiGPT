import os
import sys
import glob
import random
import numpy as np
import msgpack
import tensorflow as tf
import multiprocessing
from tqdm import tqdm
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
tf.config.set_visible_devices([], 'GPU')
from config import TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR, NUM_PARALLEL_GAMES

def serialize_sample(input_planes, policy, value):
    feature = {
        'input_planes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(input_planes).numpy()])),
        'policy': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(policy).numpy()])),
        'value': tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def board_to_input_planes_tf(board_1d_tf, current_player_tf):
    board_2d_tf = tf.reshape(board_1d_tf, (8, 8))
    p_mask = tf.cast(tf.equal(board_2d_tf, tf.cast(current_player_tf, tf.int8)), tf.float32)
    o_mask = tf.cast(tf.equal(board_2d_tf, 3 - tf.cast(current_player_tf, tf.int8)), tf.float32)
    return tf.stack([p_mask, o_mask], axis=-1)

def process_and_write_file(args):
    msgpack_path, output_path = args
    sample_count = 0

    try:
        with tf.io.TFRecordWriter(output_path) as writer:
            with open(msgpack_path, 'rb') as f:
                unpacker = msgpack.Unpacker(f, raw=False, use_list=True)
                for unpacked_item in unpacker:
                    games = unpacked_item if (isinstance(unpacked_item, list) and len(unpacked_item) > 0 and isinstance(unpacked_item[0], list)) else [unpacked_item]

                    for game_history in games:
                        if not game_history: continue
                        
                        for record in game_history:
                            try:
                                board_np = np.array(record['board'], dtype=np.int8)
                                player = record['player']
                                policy_np = np.array(record['policy'], dtype=np.float32)
                                self_stones = np.sum(board_np == player)
                                opp_stones = np.sum(board_np == (3 - player))
                                if self_stones > opp_stones:
                                    value = 1.0
                                elif self_stones < opp_stones:
                                    value = -1.0
                                else:
                                    value = 0.0
                                
                                board_tf = tf.convert_to_tensor(board_np, dtype=tf.int8)
                                policy_tf = tf.convert_to_tensor(policy_np, dtype=tf.float32)
                                input_planes = board_to_input_planes_tf(board_tf, player)
                                
                                writer.write(serialize_sample(input_planes, policy_tf, value))
                                sample_count += 1
                            except: continue
    except Exception as e:
        print(f"Error processing {msgpack_path}: {e}")
        return 0
    return sample_count

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    print(f"--- Instant Value Override Sanitizer ---")
    print(f"Target: {CURRENT_GENERATION_DATA_SUBDIR}")
    print(f"Logic: Value = (Self > Opponent ? 1.0 : -1.0) | Stack order: [Self, Opponent]")

    source_dir = os.path.join(TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR)
    output_dir = os.path.join(source_dir, 'tfrecords')
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True); os.makedirs(val_dir, exist_ok=True)

    for d in [train_dir, val_dir]:
        for f in glob.glob(os.path.join(d, "*.tfrecord")): os.remove(f)

    msgpack_files = glob.glob(os.path.join(source_dir, '*.msgpack'))
    print(f"DEBUG: Searching in {source_dir}")
    print(f"DEBUG: Found {len(msgpack_files)} msgpack files.")
    if not msgpack_files: exit()
    
    random.shuffle(msgpack_files)
    val_split = max(1, int(len(msgpack_files) * 0.1))
    train_files = msgpack_files[val_split:]
    val_files = msgpack_files[:val_split]

    current_time = int(time.time())
    num_workers = NUM_PARALLEL_GAMES

    with multiprocessing.Pool(num_workers) as pool:
        t_tasks = [(fp, os.path.join(train_dir, f"{current_time + i}.tfrecord")) for i, fp in enumerate(train_files)]
        for _ in tqdm(pool.imap_unordered(process_and_write_file, t_tasks), total=len(t_tasks), desc="Converting Train"): pass
        v_tasks = [(fp, os.path.join(val_dir, f"{current_time + len(train_files) + i}.tfrecord")) for i, fp in enumerate(val_files)]
        for _ in tqdm(pool.imap_unordered(process_and_write_file, v_tasks), total=len(v_tasks), desc="Converting Val"): pass

    print("\nSanitization and Conversion Complete. All samples are now strictly normalized by stone counts.")
