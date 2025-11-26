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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'Database', 'codingame', 'weights')
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
    # Residual blocks (conv2d_1 to conv2d_10, batch_normalization_1 to batch_normalization_10)
    # These are named sequentially in Keras, so direct mapping is fine.
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

print(f"Extracting weights to {OUTPUT_DIR}...")

for layer in model.layers:
    layer_name = layer.name
    weights = layer.get_weights()

    if weights:
        for i, w in enumerate(weights):
            cpp_var_name = None
            cpp_type = None

            # Check direct mapping
            if layer_name in WEIGHT_MAP and i in WEIGHT_MAP[layer_name]:
                cpp_var_name, cpp_type = WEIGHT_MAP[layer_name][i]
            # Handle sequential residual block layers
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
                cpp_code = f"const {cpp_type} {cpp_var_name} = {format_cpp_tensor(w.tolist(), 0)};"
                output_file_path = os.path.join(OUTPUT_DIR, f"{cpp_var_name}.h")
                with open(output_file_path, 'w') as f:
                    f.write(cpp_code)
                print(f"  Extracted {layer_name} weight_{i} to {cpp_var_name}.h")
            else:
                print(f"  WARNING: Could not map layer {layer_name} weight_{i}. Skipping.")

print("\nWeight extraction complete.")
print(f"Please copy the contents of the .h files from {OUTPUT_DIR} into CodinGame.cpp.")
