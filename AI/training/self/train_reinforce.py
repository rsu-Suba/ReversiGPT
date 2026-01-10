import os
import sys
import subprocess
import time
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from AI.config import TRAINING_DATA_DIR

PYTHON_EXE = sys.executable

def run_command(command, description):
    print(f"\n>>> [Reinforce] {description} ...")
    start = time.time()
    try:
        subprocess.check_call([PYTHON_EXE] + command.split())
        print(f">>> [Reinforce] {description} Finished ({time.time() - start:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
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

def main():
    print("=== MoE-2 Reinforcement Learning Loop ===")
    
    cycle = 1
    best_avg_stones = 0.0
    
    while True:
        print(f"\n\n========== Cycle {cycle} ==========")
        
        # 1. Generate Self-Play Data
        if not run_command("AI/training/self/generate_selfplay_gpu.py", "Generating Games"):
            break
            
        # 2. Convert to TFRecord
        update_config_data_dir('self_play')
        if not run_command("AI/data_processing/tfrecord_converter.py", "Converting Data"):
            break
            
        # 3. Train
        if not run_command("AI/training/train_loop.py", "Training"):
            break
            
        # 4. Evaluate
        print("\n>>> [Reinforce] Evaluating Performance...")
        try:
            result = subprocess.check_output([PYTHON_EXE, "review.py"], universal_newlines=True)
            print(result)
            
            for line in result.splitlines():
                if "[RESULT]" in line:
                    parts = line.split("|")
                    stones_part = parts[1].strip()
                    avg_stones = float(stones_part.split(":")[1].strip())
                    
                    print(f"Current Avg Stones: {avg_stones}")
                    
                    if avg_stones > 60.0:
                        print("\n!!!!!!!!!! GOAL ACHIEVED !!!!!!!!!!")
                        print(f"Cycle {cycle}: Average Stones {avg_stones} > 60.0")
                        shutil.copy("models/TF/MoE-2.keras", f"models/TF/MoE-2-God-{avg_stones:.1f}.keras")
                        return
                    
                    if avg_stones > best_avg_stones:
                        print(f"New Best! ({best_avg_stones} -> {avg_stones})")
                        best_avg_stones = avg_stones
                        shutil.copy("models/TF/MoE-2.keras", "models/TF/MoE-2-Best.keras")
                        
        except Exception as e:
            print(f"Evaluation failed: {e}")
            
        cycle += 1

if __name__ == "__main__":
    main()
