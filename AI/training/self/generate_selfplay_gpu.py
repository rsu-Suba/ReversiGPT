import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from absl import logging
logging.set_verbosity(logging.ERROR)
import sys

import numpy as np
import tensorflow as tf
import msgpack
import time
import multiprocessing
from multiprocessing import shared_memory
import traceback
from tqdm import tqdm
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from AI.log_filter import install_log_filter
install_log_filter()
from AI.models.model_selector import try_load_model
from AI.config import TRAINING_DATA_DIR
from AI.config_loader import load_config
from AI.cpp.reversi_bitboard_cpp import ReversiBitboard
try:
    from AI.cpp.reversi_mcts_cpp import MCTS as MCTS_CPP
    USE_CPP_MCTS = True
except ImportError:
    print("FATAL: C++ MCTS module not found.")
    sys.exit(1)

GEN_SIMS_N = 1000
GEN_GAMES_PER_ITER = 200
NUM_WORKERS = 40
WORKER_BATCH_SIZE = 40
MAX_BATCH = NUM_WORKERS * WORKER_BATCH_SIZE

STATUS_IDLE = 0
STATUS_REQ_READY = 1
STATUS_PROCESSING = 2
STATUS_RES_READY = 3
STATUS_ERROR = -1

global_worker_id = None
global_shm_objects = []
global_arrays = {}

def create_shm_array(shm, shape, dtype):
    return np.ndarray(shape, dtype=dtype, buffer=shm.buf)

def prediction_worker(model_path, shm_names, shapes, stop_event):
    import tensorflow as tf
    from keras import mixed_precision
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    try:
        mixed_precision.set_global_policy('mixed_float16')
    except:
        pass

    def board_to_input_planes_tf(board_1d_batch_tf, current_player_batch_tf):
        batch_size = tf.shape(board_1d_batch_tf)[0]
        board_2d = tf.reshape(board_1d_batch_tf, (batch_size, 8, 8))
        curr = tf.reshape(current_player_batch_tf, (batch_size, 1, 1))
        me_mask = tf.cast(tf.equal(board_2d, curr), tf.float32)
        opp_mask = tf.cast(tf.equal(board_2d, 3 - curr), tf.float32)
        return tf.stack([me_mask, opp_mask], axis=-1)

    shm_objs = {}
    arrays = {}
    try:
        for name, shm_name in shm_names.items():
            shm = shared_memory.SharedMemory(name=shm_name)
            shm_objs[name] = shm
            arrays[name] = create_shm_array(shm, shapes[name], 
                                            np.int8 if 'status' in name or 'board' in name else 
                                            (np.int32 if 'player' in name or 'counts' in name else np.float32))
    except Exception as e:
        print(f"[Server] SHM Attach Failed: {e}")
        return

    print(f"[Server] Loading model: {model_path}")
    config = load_config(type('Args', (), {'model': 'moe-2'}))
    model = try_load_model(model_path, config=config)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[MAX_BATCH, 64], dtype=tf.int8),
        tf.TensorSpec(shape=[MAX_BATCH], dtype=tf.int8)
    ], jit_compile=True)
    def predict_step(board_batch, player_batch):
        input_planes = board_to_input_planes_tf(tf.cast(board_batch, tf.int32), tf.cast(player_batch, tf.int32))
        outputs = model(input_planes, training=False)
        return outputs[0], outputs[1]

    print(f"[Server] Ready. Workers: {NUM_WORKERS}, Max Batch: {MAX_BATCH}")
    
    while not stop_event.is_set():
        np_status = arrays['status']
        ready_indices = np.where(np_status == STATUS_REQ_READY)[0]
        
        if len(ready_indices) > 0:
            np_status[ready_indices] = STATUS_PROCESSING
            
            combined_boards = np.zeros((MAX_BATCH, 64), dtype=np.int8)
            combined_players = np.zeros((MAX_BATCH,), dtype=np.int8)
            map_info = []
            
            current_idx = 0
            for wid in ready_indices:
                count = arrays['counts'][wid]
                if 0 < count <= WORKER_BATCH_SIZE:
                    combined_boards[current_idx : current_idx + count] = arrays['input_board'][wid, :count, :]
                    combined_players[current_idx : current_idx + count] = arrays['input_player'][wid, :count]
                    map_info.append((wid, current_idx, count))
                    current_idx += count
            
            if not map_info:
                np_status[ready_indices] = STATUS_IDLE
                continue
                
            try:
                p_out, v_out = predict_step(combined_boards, combined_players)
                p_np = p_out.numpy()
                v_np = tf.squeeze(v_out, axis=-1).numpy()
                
                for wid, start, count in map_info:
                    arrays['output_policy'][wid, :count, :] = p_np[start : start+count]
                    arrays['output_value'][wid, :count] = v_np[start : start+count]
                    np_status[wid] = STATUS_RES_READY
                    
            except Exception as e:
                print(f"[Server] Inference Error: {e}")
                for wid, _, _ in map_info: np_status[wid] = STATUS_ERROR
        else:
            time.sleep(0.0001)

    for shm in shm_objs.values(): shm.close()

class RemoteModelWrapper:
    def __init__(self):
        self.worker_id = global_worker_id
        self._predict_internal_cpp = self.predict_batch

    def predict_batch(self, boards, players):
        n_samples = len(boards)
        if n_samples == 0: return [], []
        global_arrays['input_board'][self.worker_id, :n_samples, :] = np.array(boards, dtype=np.int8)
        global_arrays['input_player'][self.worker_id, :n_samples] = np.array(players, dtype=np.int32)
        global_arrays['counts'][self.worker_id] = n_samples
        global_arrays['status'][self.worker_id] = STATUS_REQ_READY
        start = time.time()
        while True:
            s = global_arrays['status'][self.worker_id]
            if s == STATUS_RES_READY: break
            if s == STATUS_ERROR: raise RuntimeError("Server Error")
            if time.time() - start > 600: raise TimeoutError("Timeout")
            time.sleep(0.0001)
        policies = global_arrays['output_policy'][self.worker_id, :n_samples, :].copy()
        values = global_arrays['output_value'][self.worker_id, :n_samples].copy()
        global_arrays['status'][self.worker_id] = STATUS_IDLE
        return policies, values

def worker_init(id_queue, shm_names, shapes):
    global global_worker_id, global_shm_objects, global_arrays
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    global_worker_id = id_queue.get()
    for name, shm_name in shm_names.items():
        shm = shared_memory.SharedMemory(name=shm_name)
        global_shm_objects.append(shm)
        global_arrays[name] = create_shm_array(shm, shapes[name], 
                                        np.int8 if 'status' in name or 'board' in name else 
                                        (np.int32 if 'player' in name or 'counts' in name else np.float32))

def play_one_game(_):
    try:
        wrapper = RemoteModelWrapper()
        mcts = MCTS_CPP(wrapper, c_puct=2.3, batch_size=WORKER_BATCH_SIZE)
        board = ReversiBitboard()
        history = []
        while not board.is_game_over():
            valid = board.get_legal_moves()
            if not valid:
                board.apply_move(-1)
                continue
            add_noise = (len(history) < 15)
            root = mcts.search(board, board.current_player, GEN_SIMS_N, add_noise)
            policy = np.zeros(64, dtype=np.float32)
            total_visits = 0
            if hasattr(root, 'children'):
                for move, child in root.children.items():
                    policy[move] = child.n_visits
                    total_visits += child.n_visits
            if total_visits > 0: policy /= total_visits
            else: policy[valid[0]] = 1.0
            history.append({'board': board.board_to_numpy().tolist(), 'player': board.current_player, 'policy': policy.tolist()})
            move = np.random.choice(64, p=policy) if add_noise else np.argmax(policy)
            board.apply_move(move)
        winner = board.get_winner()
        for rec in history:
            rec['value'] = 0.0 if winner == 0 else (1.0 if winner == rec['player'] else -1.0)
        return history
    except Exception as e:
        print(f"Worker Error: {e}")
        return []

def main():
    config = load_config(type('Args', (), {'model': 'moe-2'}))
    model_path = config.get('model_save_path', './models/TF/MoE-2.keras')
    output_dir = os.path.join(TRAINING_DATA_DIR, 'self_play')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Generating (GPU/CPU Optimized Scale, SimsN={GEN_SIMS_N}) ---")
    
    shapes = {
        'input_board': (NUM_WORKERS, WORKER_BATCH_SIZE, 64),
        'input_player': (NUM_WORKERS, WORKER_BATCH_SIZE),
        'output_policy': (NUM_WORKERS, WORKER_BATCH_SIZE, 64),
        'output_value': (NUM_WORKERS, WORKER_BATCH_SIZE),
        'status': (NUM_WORKERS,),
        'counts': (NUM_WORKERS,)
    }
    
    shm_list = []
    shm_names = {}
    try:
        for name, shape in shapes.items():
            dtype = np.int8 if 'status' in name or 'board' in name else (np.int32 if 'player' in name or 'counts' in name else np.float32)
            shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape) * np.dtype(dtype).itemsize))
            shm_list.append(shm)
            shm_names[name] = shm.name
            
        ctx = multiprocessing.get_context('spawn')
        id_queue = ctx.Queue()
        for i in range(NUM_WORKERS): id_queue.put(i)
        
        stop_event = ctx.Event()
        server = ctx.Process(target=prediction_worker, args=(model_path, shm_names, shapes, stop_event))
        server.start()
        time.sleep(30)
        
        start_time = time.time()
        games_collected = []
        with ctx.Pool(NUM_WORKERS, initializer=worker_init, initargs=(id_queue, shm_names, shapes)) as pool:
            for data in tqdm(pool.imap_unordered(play_one_game, [None]*GEN_GAMES_PER_ITER), total=GEN_GAMES_PER_ITER):
                if data: games_collected.append(data)
        
        stop_event.set()
        server.join()
        
        save_path = os.path.join(output_dir, f"gen_{int(time.time())}.msgpack")
        with open(save_path, 'wb') as f: msgpack.pack(games_collected, f)
        print(f"\nSaved {len(games_collected)} games. Time: {time.time() - start_time:.1f}s")
        
    finally:
        for shm in shm_list:
            try: shm.close(); shm.unlink()
            except: pass

if __name__ == "__main__":
    main()
