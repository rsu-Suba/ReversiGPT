import numpy as np
import tensorflow as tf
import math
import random
import time
import os
import msgpack
import multiprocessing
import json
import glob
from tqdm import tqdm

from reversi_bitboard_cpp import ReversiBitboard
from reversi_mcts_cpp import MCTS as MCTS_CPP
from Database.transformer_Model import TransformerBlock

def _print_numpy_board(board_1d):
    print("  0 1 2 3 4 5 6 7")
    print("-----------------")
    for r in range(8):
        row_str = f"{r}|"
        for c in range(8):
            piece = board_1d[r * 8 + c]
            if piece == 1: row_str += "🔴"
            elif piece == 2: row_str += "⚪️"
            else: row_str += "🟩"
        print(row_str)
    print("-----------------")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

from config import (
    NUM_PARALLEL_GAMES,
    SIMS_N,
    C_PUCT,
    TOTAL_GAMES,
    TRAINING_HOURS,
    TRAINING_DATA_DIR,
    CURRENT_GENERATION_DATA_SUBDIR,
    SAVE_DATA_EVERY_N_GAMES,
    SELF_PLAY_MODEL_PATH,
    MCTS_PREDICT_BATCH_SIZE
)

def board_to_input_planes_tf(board_1d_batch_tf, current_player_batch_tf):
    batch_size = tf.shape(board_1d_batch_tf)[0]
    player_plane = tf.zeros((batch_size, 8, 8), dtype=tf.float32)
    opponent_plane = tf.zeros((batch_size, 8, 8), dtype=tf.float32)
    board_2d_batch_tf = tf.reshape(board_1d_batch_tf, (batch_size, 8, 8))
    current_player_batch_expanded = tf.expand_dims(tf.expand_dims(current_player_batch_tf, -1), -1)
    current_player_mask = tf.cast(tf.equal(board_2d_batch_tf, current_player_batch_expanded), tf.float32)
    opponent_player_mask = tf.cast(tf.equal(board_2d_batch_tf, 3 - current_player_batch_expanded), tf.float32)

    player_plane += current_player_mask
    opponent_plane += opponent_player_mask

    return tf.stack([player_plane, opponent_plane], axis=-1)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super(TokenAndPositionEmbedding, self).get_config()
        config.update({
            "position": self.position,
            "d_model": self.d_model,
        })
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                                     tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                                     d_model)
        # apply sin to even indices in the array; 2i
        angle_rads_sin = tf.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads_cos = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([angle_rads_sin, angle_rads_cos], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

class ModelWrapper:
    def __init__(self, model_path):
        custom_objects = {'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'TransformerBlock': TransformerBlock}
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        self._predict_internal_cpp = tf.function(
            self._predict_for_cpp,
            input_signature=[
                tf.TensorSpec(shape=[None, 64], dtype=tf.int8),
                tf.TensorSpec(shape=[None], dtype=tf.int32)
            ]
        )

    def _predict_for_cpp(self, board_batch_tensor, player_batch_tensor):
        input_planes_batch = board_to_input_planes_tf(tf.cast(board_batch_tensor, tf.int32), tf.cast(player_batch_tensor, tf.int32))
        
        policy, value = self.model(input_planes_batch, training=False)
        return policy, tf.squeeze(value, axis=-1)

def run_self_play_game_worker(game_id, model_path, sims_n, c_puct):
    print(f"G{game_id}: Game start")
    seed = (os.getpid() + int(time.time() * 1000) + game_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        model_wrapper = ModelWrapper(model_path)
    except Exception as e:
        print(f"G{game_id}: Model load error: {e}")
        return None

    game_board = ReversiBitboard()
    game_board.history = []
    current_player = 1
    game_board.current_player = current_player

    mcts_ai = MCTS_CPP(model_wrapper, c_puct=c_puct, batch_size=MCTS_PREDICT_BATCH_SIZE)

    game_history = []

    while not game_board.is_game_over():
        legal_moves = game_board.get_legal_moves()
        if not legal_moves:
            game_board.apply_move(-1)
            current_player = game_board.current_player
            continue

        add_noise = len(game_board.history) < 30
        root_node = mcts_ai.search(game_board, current_player, sims_n, add_noise)

        policy_target = np.zeros(64, dtype=np.float32)
        if root_node.children:
            total_visits = 0
            for move, child in root_node.children.items():
                total_visits += child.n_visits
            if total_visits > 0:
                for move, child in root_node.children.items():
                    policy_target[move] = child.n_visits / total_visits

        game_history.append({
            'board': game_board.board_to_numpy().tolist(),
            'player': current_player,
            'policy': policy_target.tolist()
        })

        if len(game_board.history) < 30:
            moves = list(root_node.children.keys())
            visits = [child.n_visits for child in root_node.children.values()]
            if sum(visits) == 0:
                best_move = random.choice(legal_moves)
            else:
                probabilities = np.array(visits, dtype=np.float32) / sum(visits)
                best_move = np.random.choice(moves, p=probabilities)
        else:
            best_move = max(root_node.children.items(), key=lambda item: item[1].n_visits)[0]

        game_board.apply_move(best_move)
        current_player = game_board.current_player

    winner = game_board.get_winner()
    black_stones = game_board.count_set_bits(game_board.black_board)
    white_stones = game_board.count_set_bits(game_board.white_board)
    stonesDiff = abs(black_stones - white_stones) / (black_stones + white_stones)
    value = 2.0 + stonesDiff
    print(f"G{game_id}: Game finish, winner: {winner}")
    for record in game_history:
        if winner == 0:
            record['value'] = 0.0
        elif record['player'] == winner:
            record['value'] = value / 3
        else:
            record['value'] = -(value / 3)
            
    return game_history

def _worker_wrapper(args):
    return run_self_play_game_worker(*args)

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

def board_to_input_planes_for_tfrecord(board_1d_tf, current_player_tf):
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
                unpacker = msgpack.Unpacker(f, raw=False, use_list=True)
                for game_history in unpacker:
                    if not game_history: continue

                    for record in game_history:
                        board_np = np.array(record['board'], dtype=np.int8)
                        player = record['player']
                        policy_np = np.array(record['policy'], dtype=np.float32)
                        value = record['value']

                        board_tf = tf.convert_to_tensor(board_np, dtype=tf.int8)
                        player_tf = tf.convert_to_tensor(player, dtype=tf.int32)
                        policy_tf = tf.convert_to_tensor(policy_np, dtype=tf.float32)
                        value_tf = tf.convert_to_tensor(value, dtype=tf.float32)

                        input_planes = board_to_input_planes_for_tfrecord(board_tf, tf.cast(player_tf, tf.int8))

                        if np.any(tf.math.is_nan(input_planes)) or np.any(tf.math.is_inf(input_planes)): continue
                        if np.any(tf.math.is_nan(policy_tf)) or np.any(tf.math.is_inf(policy_tf)): continue
                        if tf.math.is_nan(value_tf) or tf.math.is_inf(value_tf): continue

                        serialized_sample = serialize_sample(input_planes, policy_tf, value_tf)
                        writer.write(serialized_sample)
                        sample_count += 1

    except Exception as e:
        print(f"File error {os.path.basename(msgpack_path)}: {e}")
        return 0

    return sample_count

def convert_msgpack_to_tfrecord():
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

    msgpack_files = glob.glob(os.path.join(source_dir, 'mcts_tree_*.msgpack'))
    if not msgpack_files:
        print(f"No msgpack ->{source_dir}")
        return

    random.shuffle(msgpack_files)

    val_split = int(len(msgpack_files) * 0.1)
    if len(msgpack_files) > 1 and val_split == 0: val_split = 1
    train_files = msgpack_files[val_split:]
    val_files = msgpack_files[:val_split]

    num_workers = NUM_PARALLEL_GAMES
    print(f"Parallel : {num_workers}")

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(num_workers) as pool:
        print(f"Trained data : {len(train_files)}")
        train_tasks = [(fp, os.path.join(train_output_dir, f"part_{i:05d}.tfrecord")) for i, fp in enumerate(train_files)]

        total_train_samples = 0
        with tqdm(total=len(train_tasks), desc="Train") as pbar:
            for sample_count in pool.imap_unordered(process_and_write_file, train_tasks):
                total_train_samples += sample_count
                pbar.update(1)
        print(f"Train converted : {total_train_samples} samples")
        print(f"Val data : {len(val_files)}")
        val_tasks = [(fp, os.path.join(val_output_dir, f"part_{i:05d}.tfrecord")) for i, fp in enumerate(val_files)]

        total_val_samples = 0
        with tqdm(total=len(val_tasks), desc="Val") as pbar:
            for sample_count in pool.imap_unordered(process_and_write_file, val_tasks):
                total_val_samples += sample_count
                pbar.update(1)
        print(f"Val converted: {total_val_samples} samples")

    print("Convert successful to TFRecord.")


def run_self_play():
    game_results_buffer = []
    training_start_time = time.time()
    games_played = 0

    generation_data_path = os.path.join(TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR)
    os.makedirs(generation_data_path, exist_ok=True)

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(NUM_PARALLEL_GAMES) as pool:
        game_args = [(i + 1, SELF_PLAY_MODEL_PATH, SIMS_N, C_PUCT) for i in range(TOTAL_GAMES)]

        for game_history_result in pool.imap_unordered(_worker_wrapper, game_args):
            if game_history_result is None:
                print(f"Main process: Skiped game due to worker error.")
                continue

            game_results_buffer.extend(game_history_result)
            games_played += 1

            if games_played > 0 and games_played % SAVE_DATA_EVERY_N_GAMES == 0:
                data_filename = f"mcts_tree_{games_played}.msgpack"
                data_filepath = os.path.join(generation_data_path, data_filename)
                with open(data_filepath, "wb") as f:
                    msgpack.pack(game_results_buffer, f)
                print(f"{len(game_results_buffer)} states from {games_played} games saved -> {data_filepath}")
                game_results_buffer.clear()

            if TRAINING_HOURS > 0 and (time.time() - training_start_time) / 3600 >= TRAINING_HOURS:
                print("Reaching finish time")
                break
            if TOTAL_GAMES > 0 and games_played >= TOTAL_GAMES:
                print("Reaching finish games")
                break

    print(f"Train finish, Games: {games_played}")

    if game_results_buffer:
        final_data_filename = f"mcts_tree_{games_played}.msgpack"
        final_data_filepath = os.path.join(generation_data_path, final_data_filename)
        with open(final_data_filepath, "wb") as f:
            msgpack.pack(game_results_buffer, f)
        print(f"Final save: {len(game_results_buffer)} states saved -> {final_data_filepath}")
    else:
        print("No final data to save")

    print("Self-play data created")

if __name__ == "__main__":
    run_self_play()
    convert_msgpack_to_tfrecord()