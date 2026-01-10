import os
import shutil
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import numpy as np
import random
import time
import msgpack
import multiprocessing
from multiprocessing import shared_memory
import traceback
from datetime import datetime

from AI.cpp.reversi_bitboard_cpp import ReversiBitboard
from AI.cpp.reversi_mcts_cpp import MCTS as MCTS_CPP
import uuid
from AI.config_loader import load_config
from AI.config import (
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
import AI.models.MoE_1 as dynamic_moe_model
from AI.training.scheduler import WarmupCosineDecay

try:
    config = load_config()
    print(f"Using model configuration: {config['model_name']}")
    print(f"Model path: {SELF_PLAY_MODEL_PATH}")
    IS_MOE = 'moe' in config['model_name'].lower()
    
except Exception as e:
    print(f"Config loading failed: {e}. Falling back to default AI.config settings.")
    IS_MOE = False

STATUS_IDLE = 0
STATUS_REQ_READY = 1
STATUS_PROCESSING = 2
STATUS_RES_READY = 3
STATUS_ERROR = -1

global_worker_id = None
global_shm_input_board = None
global_shm_input_player = None
global_shm_output_policy = None
global_shm_output_value = None
global_shm_status = None

def create_shm_array(shm, shape, dtype):
    return np.ndarray(shape, dtype=dtype, buffer=shm.buf)


def prediction_worker(model_path, shm_names, shapes, stop_event, is_moe=False):
    import tensorflow as tf
    import keras
    from keras import mixed_precision

    if is_moe:
        from AI.models.MoE_1 import DynamicAssembly, TokenAndPositionEmbedding
        print("[PredictionWorker] Using MoE architecture (AI.models.static_MoE)")
    else:
        from AI.models.transformer import TransformerBlock, TokenAndPositionEmbedding
        print("[PredictionWorker] Using Standard Transformer architecture (AI.models.transformer)")
    
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    mixed_precision.set_global_policy('mixed_float16')

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

    shm_input_board = shared_memory.SharedMemory(name=shm_names['input_board'])
    shm_input_player = shared_memory.SharedMemory(name=shm_names['input_player'])
    shm_output_policy = shared_memory.SharedMemory(name=shm_names['output_policy'])
    shm_output_value = shared_memory.SharedMemory(name=shm_names['output_value'])
    shm_status = shared_memory.SharedMemory(name=shm_names['status'])
    shm_counts = shared_memory.SharedMemory(name=shm_names['counts'])
    np_input_board = create_shm_array(shm_input_board, shapes['input_board'], np.int8)
    np_input_player = create_shm_array(shm_input_player, shapes['input_player'], np.int32)
    np_output_policy = create_shm_array(shm_output_policy, shapes['output_policy'], np.float32)
    np_output_value = create_shm_array(shm_output_value, shapes['output_value'], np.float32)
    np_status = create_shm_array(shm_status, shapes['status'], np.int8)
    np_counts = create_shm_array(shm_counts, shapes['counts'], np.int32)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"[PredictionWorker] {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(f"[PredictionWorker] GPU Config Error: {e}")

    print(f"[PredictionWorker] Loading model from {model_path}...")
    try:
        # custom_objects = {'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'TransformerBlock': TransformerBlock}
        custom_objects = {
            'TokenAndPositionEmbedding': dynamic_moe_model.TokenAndPositionEmbedding,
            'MHA': dynamic_moe_model.MHA,
            'FFN': dynamic_moe_model.FFN,
            'DynamicAssembly': dynamic_moe_model.DynamicAssembly,
            'WarmupCosineDecay': WarmupCosineDecay
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        print(f"[PredictionWorker] FATAL: Model load error: {e}")
        traceback.print_exc()
        return

    MAX_BATCH_SIZE = NUM_PARALLEL_GAMES * MCTS_PREDICT_BATCH_SIZE

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[MAX_BATCH_SIZE, 64], dtype=tf.int8),
            tf.TensorSpec(shape=[MAX_BATCH_SIZE], dtype=tf.int32)
        ],
        jit_compile=True
    )
    def predict_step(board_batch, player_batch):
        input_planes = board_to_input_planes_tf(tf.cast(board_batch, tf.int32), tf.cast(player_batch, tf.int32))
        policy, value = model(input_planes, training=False)
        return policy, tf.squeeze(value, axis=-1)

    print("[PredictionWorker] Ready. Starting shared memory polling loop.")

    last_log_time = time.time()
    batch_utilization_stats = []

    while not stop_event.is_set():
        ready_indices = np.where(np_status == STATUS_REQ_READY)[0]
        num_ready = len(ready_indices)

        if num_ready > 0:
            start_wait = time.time()
            while num_ready < NUM_PARALLEL_GAMES and (time.time() - start_wait) < 0.015:
                time.sleep(0.0005)
                ready_indices = np.where(np_status == STATUS_REQ_READY)[0]
                num_ready = len(ready_indices)

            np_status[ready_indices] = STATUS_PROCESSING
            
            batch_utilization_stats.append(num_ready)
            if time.time() - last_log_time > 30.0:
                 if batch_utilization_stats:
                    avg_batch = sum(batch_utilization_stats) / len(batch_utilization_stats)
                    print(f"[PredictionWorker] Avg Workers/Batch: {avg_batch:.2f}/{NUM_PARALLEL_GAMES}", flush=True)
                 batch_utilization_stats = []
                 last_log_time = time.time()
            
            batch_counts = np_counts[ready_indices]
            total_boards = []
            total_players = []
            worker_map = []
            
            current_idx = 0
            total_samples = 0
            
            for i, worker_idx in enumerate(ready_indices):
                count = batch_counts[i]
                if count <= 0 or count > MCTS_PREDICT_BATCH_SIZE:
                    # print(f"[PredictionWorker] Invalid batch count {count} for worker {worker_idx}. Skipping.")
                    np_status[worker_idx] = STATUS_ERROR
                    continue
                    
                b = np_input_board[worker_idx, :count, :]
                p = np_input_player[worker_idx, :count]
                
                total_boards.append(b)
                total_players.append(p)
                worker_map.append((worker_idx, current_idx, count))
                current_idx += count
                total_samples += count
            
            if not total_boards:
                 continue

            combined_boards = np.concatenate(total_boards, axis=0)
            combined_players = np.concatenate(total_players, axis=0)
            
            pad_size = MAX_BATCH_SIZE - total_samples
            if pad_size > 0:
                pad_boards = np.zeros((pad_size, 64), dtype=np.int8)
                pad_players = np.zeros((pad_size,), dtype=np.int32)
                combined_boards = np.concatenate([combined_boards, pad_boards], axis=0)
                combined_players = np.concatenate([combined_players, pad_players], axis=0)
            elif pad_size < 0:
                print(f"[PredictionWorker] FATAL: Batch size {total_samples} exceeds MAX {MAX_BATCH_SIZE}")
                for worker_idx, _, _ in worker_map: np_status[worker_idx] = STATUS_ERROR
                continue

            try:
                policy_batch, value_batch = predict_step(combined_boards, combined_players)
                
                policy_np = policy_batch.numpy()
                value_np = value_batch.numpy()
            
                for worker_idx, start_idx, count in worker_map:
                    p_slice = policy_np[start_idx : start_idx + count]
                    v_slice = value_np[start_idx : start_idx + count]
                    
                    np_output_policy[worker_idx, :count, :] = p_slice
                    np_output_value[worker_idx, :count] = v_slice
                    
                    np_status[worker_idx] = STATUS_RES_READY
                
            except Exception as e:
                print(f"[PredictionWorker] Inference Error: {e}")
                traceback.print_exc()
                for worker_idx, _, _ in worker_map:
                     np_status[worker_idx] = STATUS_ERROR
        else:
            time.sleep(0.0005)

    shm_input_board.close()
    shm_input_player.close()
    shm_output_policy.close()
    shm_output_value.close()
    shm_status.close()


class RemoteModelWrapper:
    def __init__(self):
        self.worker_id = global_worker_id
        self.shm_input_board_view = global_shm_input_board
        self.shm_input_player_view = global_shm_input_player
        self.shm_output_policy_view = global_shm_output_policy
        self.shm_output_value_view = global_shm_output_value
        self.shm_status_view = global_shm_status
        self.shm_counts_view = global_shm_counts

        if self.worker_id is None:
            raise RuntimeError("RemoteModelWrapper initialized without worker_id!")
        
        self._predict_internal_cpp = self._predict_internal_cpp_func

    def _predict_internal_cpp_func(self, board_batch, player_batch):
        board_batch_np = np.array(board_batch, dtype=np.int8)
        player_batch_np = np.array(player_batch, dtype=np.int32)

        batch_len = len(board_batch_np)
        if batch_len > MCTS_PREDICT_BATCH_SIZE:
             # print(f"WARN: Worker {self.worker_id}: Batch size {batch_len} exceeds limit {MCTS_PREDICT_BATCH_SIZE}. Truncating.")
             batch_len = MCTS_PREDICT_BATCH_SIZE
             board_batch_np = board_batch_np[:batch_len]
             player_batch_np = player_batch_np[:batch_len]

        self.shm_input_board_view[self.worker_id, :batch_len, :] = board_batch_np
        self.shm_input_player_view[self.worker_id, :batch_len] = player_batch_np
        self.shm_counts_view[self.worker_id] = batch_len
        self.shm_status_view[self.worker_id] = STATUS_REQ_READY
        start_wait = time.time()
        while True:
            status = self.shm_status_view[self.worker_id]
            
            if status == STATUS_RES_READY:
                break
            if status == STATUS_ERROR:
                self.shm_status_view[self.worker_id] = STATUS_IDLE
                raise RuntimeError(f"Worker {self.worker_id}: Prediction server reported an error.")
            
            if (time.time() - start_wait) > 60.0:
                self.shm_status_view[self.worker_id] = STATUS_IDLE
                raise RuntimeError(f"Worker {self.worker_id}: Timed out waiting for prediction server (60s).")
            
            time.sleep(0.0005)

        policy = self.shm_output_policy_view[self.worker_id, :batch_len, :].copy()
        value = self.shm_output_value_view[self.worker_id, :batch_len].copy()
        
        self.shm_status_view[self.worker_id] = STATUS_IDLE
        
        return policy, value

global_shm_objects = []

def worker_init(id_queue, shm_names, shapes):
    global global_worker_id
    global global_shm_input_board
    global global_shm_input_player
    global global_shm_output_policy
    global global_shm_output_value
    global global_shm_status
    global global_shm_counts
    global global_shm_objects

    try:
        global_worker_id = id_queue.get()
        # print(f"Worker {os.getpid()} assigned ID: {global_worker_id}", flush=True)
    except Exception as e:
        print(f"FATAL: Worker {os.getpid()} could not get worker ID: {e}", flush=True)
        raise e

    try:
        shm_input_board = shared_memory.SharedMemory(name=shm_names['input_board'])
        shm_input_player = shared_memory.SharedMemory(name=shm_names['input_player'])
        shm_output_policy = shared_memory.SharedMemory(name=shm_names['output_policy'])
        shm_output_value = shared_memory.SharedMemory(name=shm_names['output_value'])
        shm_status = shared_memory.SharedMemory(name=shm_names['status'])
        shm_counts = shared_memory.SharedMemory(name=shm_names['counts'])
        
        global_shm_objects.extend([
            shm_input_board, shm_input_player, shm_output_policy, 
            shm_output_value, shm_status, shm_counts
        ])

        global_shm_input_board = create_shm_array(shm_input_board, shapes['input_board'], np.int8)
        global_shm_input_player = create_shm_array(shm_input_player, shapes['input_player'], np.int32)
        global_shm_output_policy = create_shm_array(shm_output_policy, shapes['output_policy'], np.float32)
        global_shm_output_value = create_shm_array(shm_output_value, shapes['output_value'], np.float32)
        global_shm_status = create_shm_array(shm_status, shapes['status'], np.int8)
        global_shm_counts = create_shm_array(shm_counts, shapes['counts'], np.int32)
    except Exception as e:
        print(f"FATAL: Worker {global_worker_id} shared memory attach failed: {e}", flush=True)
        raise e


def run_self_play_game_worker(game_id, sims_n, c_puct):
    seed_int = int(uuid.uuid4().int & (2**32 - 1))
    seed = seed_int
    random.seed(seed)
    np.random.seed(seed)

    seed_lock_dir = "./tmp/seeds"
    os.makedirs(seed_lock_dir, exist_ok=True)
    seed_lock_file = os.path.join(seed_lock_dir, str(seed))
    
    if os.path.exists(seed_lock_file):
        return run_self_play_game_worker(game_id, sims_n, c_puct)
    
    try:
        with open(seed_lock_file, 'w') as f:
            f.write(str(os.getpid()))
    except OSError:
        pass
        
    print(f"G{game_id}: Game start", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        model_wrapper = RemoteModelWrapper()
    except Exception as e:
        print(f"G{game_id}: Remote Model init error: {e}")
        return None

    game_board = ReversiBitboard()
    game_board.history = []
    current_player = 1
    game_board.current_player = current_player

    try:
        mcts_ai = MCTS_CPP(model_wrapper, c_puct=c_puct, batch_size=MCTS_PREDICT_BATCH_SIZE)
    except Exception as e:
        print(f"FATAL: MCTS_CPP init failed: {e}")
        return None

    game_history = []
    start_time = time.time()

    while not game_board.is_game_over():
        legal_moves = game_board.get_legal_moves()
        if not legal_moves:
            game_board.apply_move(-1)
            current_player = game_board.current_player
            continue

        add_noise = len(game_board.history) < 12
        root_node = mcts_ai.search(game_board, current_player, sims_n, add_noise)

        policy_target = np.zeros(64, dtype=np.float32)
        if root_node.children:
            total_visits = sum(child.n_visits for child in root_node.children.values())
            if total_visits > 0:
                for move, child in root_node.children.items():
                    policy_target[move] = child.n_visits / total_visits

        game_history.append({
            'board': game_board.board_to_numpy().tolist(),
            'player': current_player,
            'policy': policy_target.tolist()
        })

        if len(game_board.history) < 12:
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
    
    end_time = time.time()
    duration = end_time - start_time
    end_time_str = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')

    print(f"G{game_id}: Game finish, winner: {winner}. Time: {duration:.2f}s. Finished at: {end_time_str}", flush=True)
    
    for record in game_history:
        if winner == 0:
            record['value'] = 0.0
        elif record['player'] == winner:
            record['value'] = 1.0
        else:
            record['value'] = -1.0
            
    return game_history

def _worker_wrapper(args):
    try:
        return run_self_play_game_worker(*args)
    except Exception as e:
        print(f"Error in game worker {args[0] if args else ''}: {e}")
        traceback.print_exc()
        return None

def run_self_play():
    shutil.rmtree(".gemini/tmp/seeds", ignore_errors=True)

    game_results_buffer = []
    training_start_time = time.time()
    games_played = 0

    generation_data_path = os.path.join(TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR)
    os.makedirs(generation_data_path, exist_ok=True)

    shapes = {
        'input_board': (NUM_PARALLEL_GAMES, MCTS_PREDICT_BATCH_SIZE, 64),
        'input_player': (NUM_PARALLEL_GAMES, MCTS_PREDICT_BATCH_SIZE),
        'output_policy': (NUM_PARALLEL_GAMES, MCTS_PREDICT_BATCH_SIZE, 64),
        'output_value': (NUM_PARALLEL_GAMES, MCTS_PREDICT_BATCH_SIZE),
        'status': (NUM_PARALLEL_GAMES,),
        'counts': (NUM_PARALLEL_GAMES,)
    }

    dtypes = {
        'input_board': np.int8,
        'input_player': np.int32,
        'output_policy': np.float32,
        'output_value': np.float32,
        'status': np.int8,
        'counts': np.int32
    }

    shm_objects = {}
    shm_names = {}

    try:
        for key, shape in shapes.items():
            dtype = dtypes[key]
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            shm = shared_memory.SharedMemory(create=True, size=size)
            shm_objects[key] = shm
            shm_names[key] = shm.name

        ctx = multiprocessing.get_context("spawn")
        id_queue = ctx.Queue() 
        for i in range(NUM_PARALLEL_GAMES):
            id_queue.put(i)

        print("Starting Prediction Server (Shared Memory Mode)...")
        stop_event = ctx.Event()
        pred_process = ctx.Process(
            target=prediction_worker, 
            args=(SELF_PLAY_MODEL_PATH, shm_names, shapes, stop_event, IS_MOE)
        )
        pred_process.start()
        
        print("Waiting for Prediction Server to initialize (10s)...")
        time.sleep(10) 

        with ctx.Pool(NUM_PARALLEL_GAMES, initializer=worker_init, initargs=(id_queue, shm_names, shapes)) as pool:
            game_args = [(i + 1, SIMS_N, C_PUCT) for i in range(TOTAL_GAMES)]

            for game_history_result in pool.imap_unordered(_worker_wrapper, game_args):
                if game_history_result is None:
                    print(f"Main process: Skipped game due to worker error.")
                    continue

                game_results_buffer.append(game_history_result)
                games_played += 1

                if games_played > 0 and games_played % SAVE_DATA_EVERY_N_GAMES == 0:
                    data_time = int(time.time())
                    data_filename = f"{data_time}.msgpack"
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

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        print("Shutting down...")
        if 'stop_event' in locals():
            stop_event.set()
        if 'pred_process' in locals():
            pred_process.join()
        
        for shm in shm_objects.values():
            try:
                shm.close()
                shm.unlink()
            except:
                pass

    print(f"Train finish, Games: {games_played}")

    if game_results_buffer:
        final_timestamp = int(time.time())
        final_data_filename = f"{final_timestamp}.msgpack"
        final_data_filepath = os.path.join(generation_data_path, final_data_filename)
        with open(final_data_filepath, "wb") as f:
            msgpack.pack(game_results_buffer, f)
        print(f"Final save: {len(game_results_buffer)} states saved -> {final_data_filepath}")

if __name__ == "__main__":
    run_self_play()
