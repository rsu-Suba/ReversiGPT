#!/usr/bin/env python3
import sys
import os
import itertools
import csv
# カレントディレクトリをパスに追加してモジュールをロード可能にする
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from compare_models import run_comparison

# 設定
MODELS = [
    {"name": "TF3",    "path": "./models/TF/3G.h5"},
    {"name": "MoE-1",  "path": "./models/TF/MoE-1.h5"},
    {"name": "MoE-2",  "path": "./models/TF/backup/MoE-2.keras"},
    {"name": "Random", "path": "Random"},
]

GAMES_PER_MATCH = 2 # 対局数（偶数：先後3回ずつ）
SIMS = 30            # シミュレーション回数

def main():
    print(f"=== OthelloGPT Tournament ({GAMES_PER_MATCH} games each, Sims={SIMS}) ===\n")
    
    # 詳細ログを保存するリスト
    match_logs = []
    
    # 総合成績
    summary = {m["name"]: {"points": 0, "wins": 0, "losses": 0, "draws": 0, "games": 0} for m in MODELS}
    
    for m1, m2 in itertools.combinations(MODELS, 2):
        name1 = m1["name"]
        name2 = m2["name"]
        
        print(f"Matchup: {name1} vs {name2}")
        
        stats = run_comparison(
            m1["path"], name1,
            m2["path"], name2,
            GAMES_PER_MATCH, SIMS,
            game_verbose=False
        )
        
        w1 = stats[name1]["wins"]
        w2 = stats[name2]["wins"]
        draw = stats["draws"]
        
        # 集計 (Summary用)
        summary[name1]["wins"] += w1
        summary[name1]["losses"] += w2
        summary[name1]["draws"] += draw
        summary[name1]["games"] += GAMES_PER_MATCH
        summary[name1]["points"] += w1 * 1.0 + draw * 0.5
        
        summary[name2]["wins"] += w2
        summary[name2]["losses"] += w1
        summary[name2]["draws"] += draw
        summary[name2]["games"] += GAMES_PER_MATCH
        summary[name2]["points"] += w2 * 1.0 + draw * 0.5

        # 詳細ログ (Excel用)
        match_logs.append({
            "Model": name1, "Opponent": name2, "Games": GAMES_PER_MATCH,
            "W": w1, "L": w2, "D": draw, "WinRate": f"{(w1/GAMES_PER_MATCH)*100:.1f}%",
            "AvgStones": f"{stats[name1]['stones']:.1f}", "AvgValue": f"{stats[name1]['q']:.4f}"
        })
        match_logs.append({
            "Model": name2, "Opponent": name1, "Games": GAMES_PER_MATCH,
            "W": w2, "L": w1, "D": draw, "WinRate": f"{(w2/GAMES_PER_MATCH)*100:.1f}%",
            "AvgStones": f"{stats[name2]['stones']:.1f}", "AvgValue": f"{stats[name2]['q']:.4f}"
        })
        
        print(f"Result: {name1} {w1} - {w2} {name2} (Draws: {draw})\n")

    # 1. ターミナル用順位表
    print("\n=== Final Standings ===")
    print(f"{'Rank':<5} {'Model':<10} {'Points':<8} {'Win Rate':<10} {'W-L-D':<10}")
    print("-" * 50)
    sorted_summary = sorted(summary.items(), key=lambda x: x[1]["points"], reverse=True)
    for rank, (name, s) in enumerate(sorted_summary, 1):
        wr = (s["wins"] / s["games"]) * 100 if s["games"] > 0 else 0
        wld = f"{s['wins']}-{s['losses']}-{s['draws']}"
        print(f"{rank:<5} {name:<10} {s['points']:<8.1f} {wr:>5.1f}%    {wld:<10}")

    # 2. Excel用CSV出力
    csv_file = "tournament_results.csv"
    keys = ["Model", "Opponent", "Games", "W", "L", "D", "WinRate", "AvgStones", "AvgValue"]
    with open(csv_file, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(match_logs)
    
    print(f"\n[Success] Detailed results saved to: {csv_file}")
    print("You can open this file directly in Excel.")

if __name__ == "__main__":
    main()
