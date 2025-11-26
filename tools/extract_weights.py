import tensorflow as tf
import json
import os
import numpy as np

# config.py から MODELS_DIR をインポート
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODELS_DIR, TRAINED_MODEL_SAVE_PATH

# createModel.py からモデル定義をインポート
from Database.createModel import create_dual_resnet_model

# モデルのパス (H5形式)
model_path = TRAINED_MODEL_SAVE_PATH

# モデルのインスタンスを作成し、重みをロード
model = create_dual_resnet_model()
model.load_weights(model_path)

weights_data = {}

for layer in model.layers:
    layer_name = layer.name
    weights = layer.get_weights()

    if weights:
        weights_data[layer_name] = {}
        for i, w in enumerate(weights):
            # NumPy配列をリストに変換してJSONにシリアル化可能にする
            weights_data[layer_name][f'weight_{i}'] = w.tolist()

# JSONファイルとして保存
output_file = 'model_weights.json'
with open(output_file, 'w') as f:
    json.dump(weights_data, f, indent=4)

print(f"Model weights extracted and saved to {output_file}")

# 各層の重みの形状を確認（デバッグ用）
print("\n--- Weight Shapes ---")
for layer in model.layers:
    if layer.get_weights():
        print(f"Layer: {layer.name}")
        for i, w in enumerate(layer.get_weights()):
            print(f"  Weight {i} shape: {w.shape}")
