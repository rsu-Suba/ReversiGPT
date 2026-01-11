import re
import csv
import sys
import os

def parse_match_results(file_path, output_csv="tournament_results.csv"):
    with open(file_path, 'r') as f:
        content = f.read()

    # Split by the "--- Results ---" separator, ignoring the first split (before the first match)
    blocks = content.split('--- Results ---')[1:]
    
    rows = []
    
    for block in blocks:
        # Extract players names
        # Pattern: "Name wins: X"
        wins_matches = re.findall(r"(.+?) wins: (\d+)", block)
        if len(wins_matches) < 2:
            continue
            
        p1_name, p1_wins = wins_matches[0]
        p2_name, p2_wins = wins_matches[1]
        p1_wins = int(p1_wins)
        p2_wins = int(p2_wins)
        
        # Extract draws
        draw_match = re.search(r"Draw: (\d+)", block)
        draws = int(draw_match.group(1)) if draw_match else 0
        
        total_games = p1_wins + p2_wins + draws
        
        # Extract average scores
        # Pattern: "Name ave scores: X.X"
        scores_matches = re.findall(r"(.+?) ave scores: ([\d\.]+)", block)
        # Create a dict for easy lookup
        scores = {name.strip(): float(score) for name, score in scores_matches}
        
        # Extract average Q values
        # Pattern: "Name ave Q num: X.X" (could be negative)
        q_matches = re.findall(r"(.+?) ave Q num: ([\-\d\.]+)", block)
        q_values = {name.strip(): float(q) for name, q in q_matches}
        
        # Helper to safely get stats
        def get_stats(name):
            return scores.get(name, 0.0), q_values.get(name, 0.0)

        p1_score, p1_q = get_stats(p1_name)
        p2_score, p2_q = get_stats(p2_name)
        
        # Row 1: P1 vs P2
        rows.append({
            "Model": p1_name,
            "Opponent": p2_name,
            "Games": total_games,
            "W": p1_wins,
            "L": p2_wins,
            "D": draws,
            "WinRate": f"{(p1_wins/total_games)*100:.1f}%" if total_games > 0 else "0.0%",
            "AvgStones": f"{p1_score:.1f}",
            "AvgValue": f"{p1_q:.4f}"
        })
        
        # Row 2: P2 vs P1
        rows.append({
            "Model": p2_name,
            "Opponent": p1_name,
            "Games": total_games,
            "W": p2_wins,
            "L": p1_wins,
            "D": draws,
            "WinRate": f"{(p2_wins/total_games)*100:.1f}%" if total_games > 0 else "0.0%",
            "AvgStones": f"{p2_score:.1f}",
            "AvgValue": f"{p2_q:.4f}"
        })

    # Output to CSV
    keys = ["Model", "Opponent", "Games", "W", "L", "D", "WinRate", "AvgStones", "AvgValue"]
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Successfully converted {file_path} to {output_csv}")
    print(f"Processed {len(rows)//2} matches.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default behavior for the user request
        if os.path.exists("match_results.txt"):
            parse_match_results("match_results.txt")
        else:
            print("Please provide the log file path.")
    else:
        parse_match_results(sys.argv[1])
