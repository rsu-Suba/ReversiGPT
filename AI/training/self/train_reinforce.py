import os
import sys
import subprocess
import time
import shutil
import traceback
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Enable XLA persistent cache to speed up JIT compilation across runs
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda' # Optional: helps finding cuda libs
# Set a persistent cache directory for XLA
cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.xla_cache'))
os.makedirs(cache_dir, exist_ok=True)
os.environ['TF_XLA_FLAGS'] = f"--tf_xla_persistent_cache_directory={cache_dir} --tf_xla_persistent_cache_prefix=xla_cache"

from absl import logging
logging.set_verbosity(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from AI.config import TRAINING_DATA_DIR

PYTHON_EXE = sys.executable

import re

def should_ignore_log(line):
    ignore_keywords = [
        "computation_placer.cc",
        "cuda_dnn.cc",
        "service.cc",
        "E0000",
        "I0000",
        "WARNING",
        "oneDNN",
        "successful NUMA",
        "AgPlacer",
        "tensorflow"
    ]
    return any(k in line for k in ignore_keywords)

def is_progress_bar(line):
    # TQDM: 100%|... or ...it/s...
    if ('%|' in line or 'it/s]' in line or 's/it]' in line) and '|' in line and '[' in line and ']' in line:
        return True
    # Keras (Standard): 1/10 [==>...] or 1/10 [......]
    if re.search(r'\s*\d+/\d+\s+\[', line):
        return True
    # Keras (Unicode): 54/54 ━━━━━━━━━━━━━━━━━━━━
    if re.search(r'\d+/\d+\s+━', line):
        return True
    return False

def run_command(command, description, filter_log=True):

    print(f"\n>>> [Reinforce] {description} ...")

    start = time.time()

    try:

        env = os.environ.copy()

        env['TF_CPP_MIN_LOG_LEVEL'] = '3'

        

        # Directly inherit stdout/stderr to let the subprocess handle TTY/progress bars naturally

        ret = subprocess.run(

            [PYTHON_EXE] + command.split(),

            env=env

        )

        

        if ret.returncode == 0:

            print(f">>> [Reinforce] {description} Finished ({time.time() - start:.1f}s)")

            return True

        else:

            print(f">>> [Reinforce] Error in {description}: Return code {ret.returncode}")

            return False

            

    except Exception as e:

        print(f">>> [Reinforce] Error in {description}: {e}")

        return False

def update_config_data_dir(target_dir):
    config_path = "AI/config.py"
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if "CURRENT_GENERATION_DATA_SUBDIR =" in line:
            new_lines.append(f"CURRENT_GENERATION_DATA_SUBDIR = '{target_dir}'\n")
        else:
            new_lines.append(line)
            
    with open(config_path, 'w') as f:
        f.writelines(new_lines)

def update_config_model_path(model_path):
    config_path = "AI/config.py"
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if "    'model_save_path':" in line:
            # Preserve indentation
            indent = line[:line.find("'")]
            new_lines.append(f"{indent}'model_save_path': '{model_path}',\n")
        else:
            new_lines.append(line)
            
    with open(config_path, 'w') as f:
        f.writelines(new_lines)

def main():
    print("=== MoE-2 Reinforcement Learning Loop (Challenger Mode) ===")
    
    cycle = 1
    # Use a dedicated directory for self-play models to protect backups
    self_play_model_dir = "models/TF/self_play"
    os.makedirs(self_play_model_dir, exist_ok=True)
    
    best_model_path = os.path.join(self_play_model_dir, "MoE-2-Best.keras")
    current_model_path = os.path.join(self_play_model_dir, "MoE-2.keras")
    candidate_model_path = os.path.join(self_play_model_dir, "MoE-2-Candidate.keras")
    
    # Initialize Best Model if not exists in self_play folder
    if not os.path.exists(best_model_path):
        # Fallback: Copy from initial source if available, otherwise expect user to place it
        initial_source = "models/TF/MoE-2.keras" 
        if os.path.exists(initial_source):
            print(f"Initializing self-play Best model from {initial_source}")
            shutil.copy(initial_source, best_model_path)
        else:
            print(f"FATAL: No initial model found. Please place a model at {best_model_path}")
            return

    while True:
        print(f"\n\n========== Cycle {cycle} ==========")
        
        # 0. Reset to Best Model
        print(f">>> [Reinforce] Resetting working model to Best Model...")
        shutil.copy(best_model_path, current_model_path)
        
        # Point config to current model for generation
        update_config_model_path(current_model_path)

        # 1. Generate Self-Play Data
        if not run_command("AI/training/self/generate_selfplay_gpu.py", "Generating Games", filter_log=False):
            break
            
        # 2. Convert to TFRecord
        update_config_data_dir('self_play')
        if not run_command("AI/data_processing/tfrecord_converter.py", "Converting Data", filter_log=False):
            break
            
        # 3. Train (Save to Candidate, Load from Current/Best)
        os.environ['MODEL_SAVE_PATH'] = candidate_model_path
        os.environ['MODEL_LOAD_PATH'] = current_model_path
        if not run_command("AI/training/train_loop.py", "Training", filter_log=False):
            del os.environ['MODEL_SAVE_PATH']
            del os.environ['MODEL_LOAD_PATH']
            break
        del os.environ['MODEL_SAVE_PATH']
        del os.environ['MODEL_LOAD_PATH']
            
        # 4. Promote Candidate to Best (AlphaZero style: No Gating)
        print("\n>>> [Reinforce] Promoting Candidate to Best Model (AlphaZero style - No Evaluation)...")
        shutil.copy(candidate_model_path, best_model_path)
        print(f"Cycle {cycle}: Model updated.")

        cycle += 1

if __name__ == "__main__":
    main()
