import tensorflow as tf
import json
import os
import numpy as np

# config.py から MODELS_DIR, TRAINED_MODEL_SAVE_PATH をインポート
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODELS_DIR, TRAINED_MODEL_SAVE_PATH

# createModel.py からモデル定義をインポート
from Database.createModel import create_dual_resnet_model

# 出力ディレクトリ
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'Database', 'codingame', 'weights', 'blocks')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# モデルのパス (H5形式)
model_path = TRAINED_MODEL_SAVE_PATH

# モデルのインスタンスを作成し、重みをロード
# num_residual_blocks はモデルが学習された時の値に合わせる
# 以前のやり取りで num_residual_blocks=5 で重み抽出が成功したため、ここでは5を使用
model = create_dual_resnet_model(num_residual_blocks=5)
model.load_weights(model_path)

# C++のテンソル初期化文字列を生成するヘルパー関数
def format_cpp_tensor(data, indent_level=0):
    indent = "    " * indent_level
    if isinstance(data, list):
        if not data:
            return "{}"
        if not isinstance(data[0], list): # 1D tensor
            # Format floats with sufficient precision
            return "{" + ", ".join(f"{x:.10f}f" for x in data) + "}"
        else: # Nested list (2D, 3D, 4D)
            lines = ["{"]
            for item in data:
                lines.append(indent + "    " + format_cpp_tensor(item, indent_level + 1) + ",")
            lines.append(indent + "}")
            return "\n".join(lines)
    else:
        return f"{data:.10f}f" # Single float

# 重みとC++変数名のマッピング
# Kerasのレイヤー名とweight_idxからC++の変数名と型を導出
WEIGHT_MAP = {
    "conv2d": {0: ("conv2d_kernel", "Tensor4D"), 1: ("conv2d_bias", "Tensor1D")},
    "batch_normalization": {
        0: ("batch_normalization_gamma", "Tensor1D"),
        1: ("batch_normalization_beta", "Tensor1D"),
        2: ("batch_normalization_moving_mean", "Tensor1D"),
        3: ("batch_normalization_moving_variance", "Tensor1D"),
    },
    # Policy Head
    "conv2d_12": {0: ("policy_conv2d_kernel", "Tensor4D"), 1: ("policy_conv2d_bias", "Tensor1D")},
    "batch_normalization_12": {
        0: ("policy_batch_normalization_gamma", "Tensor1D"),
        1: ("policy_batch_normalization_beta", "Tensor1D"),
        2: ("policy_batch_normalization_moving_mean", "Tensor1D"),
        3: ("policy_batch_normalization_moving_variance", "Tensor1D"),
    },
    "policy_output": {0: ("policy_dense_kernel", "Tensor2D"), 1: ("policy_dense_bias", "Tensor1D")},
    # Value Head
    "conv2d_11": {0: ("value_conv2d_kernel", "Tensor4D"), 1: ("value_conv2d_bias", "Tensor1D")},
    "batch_normalization_11": {
        0: ("value_batch_normalization_gamma", "Tensor1D"),
        1: ("value_batch_normalization_beta", "Tensor1D"),
        2: ("value_batch_normalization_moving_mean", "Tensor1D"),
        3: ("value_batch_normalization_moving_variance", "Tensor1D"),
    },
    "dense": {0: ("value_dense1_kernel", "Tensor2D"), 1: ("value_dense1_bias", "Tensor1D")},
    "value_output": {0: ("value_dense2_kernel", "Tensor2D"), 1: ("value_dense2_bias", "Tensor1D")},
}

# 各ブロックの重みを格納する辞書
block_weights = {
    "initial_block": [],
    "policy_head": [],
    "value_head": [],
}
for i in range(5): # 5 residual blocks
    block_weights[f"residual_block_{i+1}_conv1"] = []
    block_weights[f"residual_block_{i+1}_conv2"] = []

# レイヤー名からブロックへのマッピング
layer_to_block_map = {
    "conv2d": "initial_block",
    "batch_normalization": "initial_block",
    "conv2d_12": "policy_head",
    "batch_normalization_12": "policy_head",
    "policy_output": "policy_head",
    "conv2d_11": "value_head",
    "batch_normalization_11": "value_head",
    "dense": "value_head",
    "value_output": "value_head",
}

# 残差ブロックのレイヤー名マッピング
for i in range(1, 11): # conv2d_1 to conv2d_10
    if i % 2 != 0: # Odd numbers are first conv in block
        layer_to_block_map[f"conv2d_{i}"] = f"residual_block_{(i+1)//2}_conv1"
        layer_to_block_map[f"batch_normalization_{i}"] = f"residual_block_{(i+1)//2}_conv1"
    else: # Even numbers are second conv in block
        layer_to_block_map[f"conv2d_{i}"] = f"residual_block_{i//2}_conv2"
        layer_to_block_map[f"batch_normalization_{i}"] = f"residual_block_{i//2}_conv2"


print(f"Extracting weights to {OUTPUT_DIR}...")

for layer in model.layers:
    layer_name = layer.name
    weights = layer.get_weights()

    if weights:
        if layer_name in layer_to_block_map:
            block_name = layer_to_block_map[layer_name]
            for i, w in enumerate(weights):
                cpp_var_name = None
                cpp_type = None

                if layer_name in WEIGHT_MAP and i in WEIGHT_MAP[layer_name]:
                    cpp_var_name, cpp_type = WEIGHT_MAP[layer_name][i]
                elif layer_name.startswith("conv2d_") and layer_name[7:].isdigit():
                    block_num = int(layer_name[7:])
                    if i == 0:
                        cpp_var_name = f"conv2d_{block_num}_kernel"
                        cpp_type = "Tensor4D"
                    elif i == 1:
                        cpp_var_name = f"conv2d_{block_num}_bias"
                        cpp_type = "Tensor1D"
                elif layer_name.startswith("batch_normalization_") and layer_name[20:].isdigit():
                    block_num = int(layer_name[20:])
                    if i == 0:
                        cpp_var_name = f"batch_normalization_{block_num}_gamma"
                        cpp_type = "Tensor1D"
                    elif i == 1:
                        cpp_var_name = f"batch_normalization_{block_num}_beta"
                        cpp_type = "Tensor1D"
                    elif i == 2:
                        cpp_var_name = f"batch_normalization_{block_num}_moving_mean"
                        cpp_type = "Tensor1D"
                    elif i == 3:
                        cpp_var_name = f"batch_normalization_{block_num}_moving_variance"
                        cpp_type = "Tensor1D"
                
                if cpp_var_name and cpp_type:
                    block_weights[block_name].append(f"const {cpp_type} {cpp_var_name} = {format_cpp_tensor(w.tolist(), 0)};")
                else:
                    print(f"  WARNING: Could not map layer {layer_name} weight_{i}. Skipping.")
        else:
            print(f"  WARNING: Layer {layer_name} not mapped to any block. Skipping.")

# 各ブロックの重みをファイルに書き出す
for block_name, weights_list in block_weights.items():
    if weights_list:
        output_file_path = os.path.join(OUTPUT_DIR, f"{block_name}_weights.h")
        with open(output_file_path, 'w') as f:
            for weight_str in weights_list:
                f.write(weight_str + "\n")
        print(f"  Generated {block_name}_weights.h")

print("\nWeight extraction and block generation complete.")
print(f"Please copy the contents of the .h files from {OUTPUT_DIR} into CodinGame.cpp.")