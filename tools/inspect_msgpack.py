import os
import sys
import msgpack
import numpy as np
import glob
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from AI.config import TRAINING_DATA_DIR

TRAINING_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

def inspect_msgpack():
    subdirs = ['3G']
    
    for subdir in subdirs:
        data_dir = os.path.join(TRAINING_DATA_DIR, subdir)
        if not os.path.exists(data_dir):
            continue
            
        print(f"\n========== Checking {subdir} ==========")
        files = glob.glob(os.path.join(data_dir, '*.msgpack'))
        
        if not files:
            print("No msgpack files found.")
            continue

        target_file = files[random.randint(0, 10)]
        print(f"Inspecting: {target_file}")

        try:
            with open(target_file, 'rb') as f:
                unpacker = msgpack.Unpacker(f, raw=False)
                games = next(unpacker)
                
            print(f"Found {len(games)} games.")

            for game_idx, game in enumerate(games[:1]):
                if not game: continue
                last_record = game[-1]
                last_board = np.array(last_record['board'])
                black_count = np.sum(last_board == 1)
                white_count = np.sum(last_board == 2)
                
                real_winner = 0
                if black_count > white_count: real_winner = 1
                elif white_count > black_count: real_winner = 2
                
                winner_str = "Draw"
                if real_winner == 1: winner_str = "Black"
                elif real_winner == 2: winner_str = "White"
                
                print(f"  Game {game_idx} Winner: {winner_str} ({real_winner}) (B:{black_count}, W:{white_count})")
                
                steps = [0, len(game)-1]
                all_ok = True
                for step in steps:
                    record = game[step]
                    player = record['player']
                    val = record['value']
                    
                    consistent = False
                    if val > 0:
                        if player == real_winner: consistent = True
                    elif val < 0:
                        if player != real_winner and real_winner != 0: consistent = True
                    else:
                        if real_winner == 0: consistent = True
                    
                    if not consistent:
                        print(f"    [FAIL] Step {step}: Player {player}, Value {val} vs Winner {real_winner}")
                        all_ok = False
                    # else:
                    #     print(f"    [OK] Step {step}: Player {player}, Value {val}")
                
                if all_ok:
                    print("    -> Game Consistency: OK")
                else:
                    print("    -> Game Consistency: FAILED (REVERSED?)")

        except Exception as e:
            print(f"Error reading {target_file}: {e}")

if __name__ == "__main__":
    inspect_msgpack()
