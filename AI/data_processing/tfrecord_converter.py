import os
import sys
import glob
import random
import collections
import numpy as np
import msgpack
import tensorflow as tf
import multiprocessing
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

tf.config.set_visible_devices([], 'GPU')

from config import TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR, NUM_PARALLEL_GAMES

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def serialize_sample(input_planes, policy, value):
    feature = {
        'input_planes': _bytes_feature(tf.io.serialize_tensor(input_planes)),
        'policy': _bytes_feature(tf.io.serialize_tensor(policy)),
        'value': _float_feature(value),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def board_to_input_planes_tf(board_1d_tf, current_player_tf):
    player_plane = tf.zeros((8, 8), dtype=tf.float32)
    opponent_plane = tf.zeros((8, 8), dtype=tf.float32)
    board_2d_tf = tf.reshape(board_1d_tf, (8, 8))
    current_player_tf_int8 = tf.cast(current_player_tf, tf.int8)

    current_player_mask = tf.cast(tf.equal(board_2d_tf, current_player_tf_int8), tf.float32)
    opponent_player_mask = tf.cast(tf.equal(board_2d_tf, 3 - current_player_tf_int8), tf.float32)
    player_plane += current_player_mask
    opponent_plane += opponent_player_mask
    return tf.stack([player_plane, opponent_plane], axis=-1)

def process_and_write_file(args):
    msgpack_path, output_path = args
    sample_count = 0

    try:
        with tf.io.TFRecordWriter(output_path) as writer:
            with open(msgpack_path, 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(0)
                if size == 0:
                    print(f"[WARN] Empty msgpack file: {msgpack_path}")
                    return 0

                unpacker = msgpack.Unpacker(f, raw=False, use_list=True)
                loaded_games = 0
                
                for unpacked_item in unpacker:
                    if isinstance(unpacked_item, list) and len(unpacked_item) > 0 and isinstance(unpacked_item[0], list):
                        games_batch = unpacked_item
                    else:
                        games_batch = [unpacked_item]

                    for game_history in games_batch:
                        loaded_games += 1
                        if not game_history: continue

                        for i, record in enumerate(game_history):
                            try:
                                board_np = np.array(record['board'], dtype=np.int8)
                                player = record['player']
                                policy_np = np.array(record['policy'], dtype=np.float32)
                                value = record['value']
                            except TypeError:
                                print(f"[ERROR] Malformed record in {os.path.basename(msgpack_path)}. Expected dict, got {type(record)}")
                                continue

                            board_tf = tf.convert_to_tensor(board_np, dtype=tf.int8)
                            player_tf = tf.convert_to_tensor(player, dtype=tf.int32)
                            policy_tf = tf.convert_to_tensor(policy_np, dtype=tf.float32)
                            value_tf = tf.convert_to_tensor(value, dtype=tf.float32)

                            input_planes = board_to_input_planes_tf(board_tf, tf.cast(player_tf, tf.int8))

                            if np.any(np.isnan(input_planes)) or np.any(np.isinf(input_planes)):
                                print(f"[SKIP] NaN/Inf in input_planes. File: {os.path.basename(msgpack_path)}, Game: {loaded_games}, Step: {i}")
                                continue
                            if np.any(np.isnan(policy_tf)) or np.any(np.isinf(policy_tf)):
                                print(f"[SKIP] NaN/Inf in policy. File: {os.path.basename(msgpack_path)}, Game: {loaded_games}, Step: {i}")
                                continue
                            if np.isnan(value_tf) or np.isinf(value_tf):
                                print(f"[SKIP] NaN/Inf in value. File: {os.path.basename(msgpack_path)}, Game: {loaded_games}, Step: {i}")
                                continue

                            serialized_sample = serialize_sample(input_planes, policy_tf, value_tf)
                            writer.write(serialized_sample)
                            sample_count += 1
                
                if loaded_games == 0:
                    print(f"[WARN] No games found in {msgpack_path}")

    except Exception as e:
        print(f"File error {os.path.basename(msgpack_path)}: {e}")
        import traceback
        traceback.print_exc()
        return 0

    return sample_count

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    print("Start convert to TFRecord")

    source_dir = os.path.join(TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR)
    output_dir = os.path.join(source_dir, 'tfrecords')

    train_output_dir = os.path.join(output_dir, 'train')
    val_output_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)

    for old_file in glob.glob(os.path.join(train_output_dir, "*.tfrecord")): os.remove(old_file)
    for old_file in glob.glob(os.path.join(val_output_dir, "*.tfrecord")): os.remove(old_file)
    print(f"Deleted old tfrecord -> {train_output_dir}, {val_output_dir}")

    msgpack_files = glob.glob(os.path.join(source_dir, '*.msgpack'))
    if not msgpack_files:
        print(f"No msgpack ->{source_dir}")
        exit()

    random.shuffle(msgpack_files)

    val_split = int(len(msgpack_files) * 0.1)
    if len(msgpack_files) > 1 and val_split == 0: val_split = 1
    train_files = msgpack_files[val_split:]
    val_files = msgpack_files[:val_split]

    num_workers = NUM_PARALLEL_GAMES
    print(f"Parallel : {num_workers}")

    with multiprocessing.Pool(num_workers) as pool:
        print(f"\nTrained data : {len(train_files)}")
        train_tasks = [(fp, os.path.join(train_output_dir, f"part_{i:05d}.tfrecord")) for i, fp in enumerate(train_files)]

        total_train_samples = 0
        with tqdm(total=len(train_tasks), desc="Train") as pbar:
            for sample_count in pool.imap_unordered(process_and_write_file, train_tasks):
                total_train_samples += sample_count
                pbar.update(1)
        print(f"Train converted : {total_train_samples} samples")

        print(f"\nVal data : {len(val_files)}")
        val_tasks = [(fp, os.path.join(val_output_dir, f"part_{i:05d}.tfrecord")) for i, fp in enumerate(val_files)]

        total_val_samples = 0
        with tqdm(total=len(val_tasks), desc="Val") as pbar:
            for sample_count in pool.imap_unordered(process_and_write_file, val_tasks):
                total_val_samples += sample_count
                pbar.update(1)
        print(f"Val converted: {total_val_samples} samples")

    print("\nConvert successful to TFRecord.")