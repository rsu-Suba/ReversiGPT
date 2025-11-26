import tensorflow as tf
import os
import numpy as np
import base64
import heapq
from collections import defaultdict

# config.py から MODELS_DIR, TRAINED_MODEL_SAVE_PATH をインポート
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODELS_DIR, TRAINED_MODEL_SAVE_PATH

# simplified_model.py から簡素化されたモデル定義をインポート
from Database.codingame.simplified_model import create_simplified_dual_resnet_model

# 出力ディレクトリ
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'Database', 'codingame', 'weights', 'blocks_compressed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 元の学習済みモデルのパス (H5形式)
original_model_path = TRAINED_MODEL_SAVE_PATH

# --- ハフマン符号化ヘルパー関数 ---
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    frequency = defaultdict(int)
    for val in data:
        frequency[val] += 1

    priority_queue = [HuffmanNode(char, freq) for char, freq in frequency.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)

    return priority_queue[0] if priority_queue else None

def build_huffman_codes(node, current_code="", codes={}):
    if node is None:
        return
    if node.char is not None:
        codes[node.char] = current_code
        return
    build_huffman_codes(node.left, current_code + "0", codes)
    build_huffman_codes(node.right, current_code + "1", codes)

def huffman_encode(data, codes):
    encoded_bits = ""
    for val in data:
        encoded_bits += codes[val]
    return encoded_bits

# --- C++のテンソル初期化文字列を生成するヘルパー関数 ---
def format_cpp_tensor_shape(shape):
    return "{" + ", ".join(map(str, shape)) + "}"

# --- 重み抽出、量子化、圧縮、C++コード生成 ---
def process_weights(layer_name, weight_idx, weight_tensor, cpp_var_prefix, quantization_bits=8):
    cpp_code_lines = []

    # 1. 量子化 (int8)
    min_val = np.min(weight_tensor)
    max_val = np.max(weight_tensor)
    
    if max_val == min_val:
        # Handle constant weights (e.g., all zeros)
        quantized_w = np.zeros_like(weight_tensor, dtype=np.int8)
        scale = 0.0
        zero_point = 0.0
    else:
        # Map float range [min_val, max_val] to int8 range [-127, 127]
        scale = (max_val - min_val) / (2**quantization_bits - 1)
        zero_point = -min_val / scale
        quantized_w = np.round(weight_tensor / scale + zero_point).astype(np.int8)

    # 2. ハフマン符号化
    huffman_tree = build_huffman_tree(quantized_w.flatten().tolist())
    codes = {}
    if huffman_tree: # Handle empty or single-value data
        build_huffman_codes(huffman_tree, "", codes)
    
    encoded_bits = huffman_encode(quantized_w.flatten().tolist(), codes)
    
    # パディングしてバイトに変換
    padded_bits = encoded_bits + '0' * ((8 - len(encoded_bits) % 8) % 8)
    byte_array = bytearray()
    for j in range(0, len(padded_bits), 8):
        byte_array.append(int(padded_bits[j:j+8], 2))
    
    # 3. Base64エンコード
    base64_encoded_data = base64.b64encode(byte_array).decode('ascii')

    # 4. C++コード生成
    cpp_var_name = f"{cpp_var_prefix}_{weight_idx}"
    cpp_code_lines.append(f"// Layer: {layer_name}, Weight: {weight_idx}")
    cpp_code_lines.append(f"const std::string {cpp_var_name}_b64 = \"{base64_encoded_data}\";")
    cpp_code_lines.append(f"const float {cpp_var_name}_scale = {scale:.10f}f;")
    cpp_code_lines.append(f"const float {cpp_var_name}_zero_point = {zero_point:.10f}f;")
    
    # ハフマンテーブルをC++マップ形式で出力
    huffman_table_name = f"{cpp_var_name}_huffman_table"
    cpp_code_lines.append(f"const std::map<std::string, int> {huffman_table_name} = {{")
    for char_val, code in codes.items():
        cpp_code_lines.append(f"    {{\"{code}\", {char_val}}},")
    cpp_code_lines.append("};")
    
    # 元の形状も保存
    cpp_code_lines.append(f"const std::vector<int> {cpp_var_name}_shape = {format_cpp_tensor_shape(weight_tensor.shape)};")
    cpp_code_lines.append("") # 空行

    return cpp_code_lines

# --- メイン処理 ---
print("Loading original model...")
original_model = tf.keras.models.load_model(original_model_path)

print("Creating simplified model...")
simplified_model = create_simplified_dual_resnet_model()

print("Loading weights from original model to simplified model...")
# by_name=True と skip_mismatch=True を使用して、一致する層の重みのみをロード
simplified_model.load_weights(original_model_path, by_name=True, skip_mismatch=True)

# 重みとC++変数名のマッピング
WEIGHT_MAP = {
    "conv2d": "conv2d",
    "batch_normalization": "batch_normalization",
    "conv2d_12": "policy_conv2d",
    "batch_normalization_12": "policy_batch_normalization",
    "policy_output": "policy_dense",
    "conv2d_11": "value_conv2d",
    "batch_normalization_11": "value_batch_normalization",
    "dense": "value_dense1",
    "value_output": "value_dense2",
}

print(f"Extracting and compressing weights to {OUTPUT_DIR}...")

# 各ブロックの重みを格納する辞書
block_weights_cpp_code = {
    "initial_block": [],
    "policy_head": [],
    "value_head": [],
}
for i in range(1): # num_residual_blocks=1 を想定
    block_weights_cpp_code[f"residual_block_{i+1}_conv1"] = []
    block_weights_cpp_code[f"residual_block_{i+1}_conv2"] = []

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

# 残差ブロックのレイヤー名マッピング (簡素化されたモデルに合わせて調整)
# num_residual_blocks=1 なので、conv2d_1 と conv2d_2 のみ
layer_to_block_map["conv2d_1"] = "residual_block_1_conv1"
layer_to_block_map["batch_normalization_1"] = "residual_block_1_conv1"
layer_to_block_map["conv2d_2"] = "residual_block_1_conv2"
layer_to_block_map["batch_normalization_2"] = "residual_block_1_conv2"


for layer in simplified_model.layers:
    layer_name = layer.name
    weights = layer.get_weights()

    if weights:
        if layer_name in layer_to_block_map:
            block_name = layer_to_block_map[layer_name]
            cpp_var_prefix = WEIGHT_MAP.get(layer_name, layer_name) # Use mapped name or original
            
            for i, w in enumerate(weights):
                block_weights_cpp_code[block_name].extend(process_weights(layer_name, i, w, cpp_var_prefix))
        else:
            print(f"  WARNING: Layer {layer_name} not mapped to any block. Skipping.")

# 各ブロックの重みをファイルに書き出す
for block_name, cpp_code_lines in block_weights_cpp_code.items():
    if cpp_code_lines:
        output_file_path = os.path.join(OUTPUT_DIR, f"{block_name}_weights.h")
        with open(output_file_path, 'w') as f:
            f.write("\n".join(cpp_code_lines))
        print(f"  Generated {block_name}_weights.h")

print("\nWeight compression and block generation complete.")
print(f"Please copy the contents of the .h files from {OUTPUT_DIR} into CodinGame.cpp.")
