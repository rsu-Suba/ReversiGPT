import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import SELF_PLAY_MODEL_PATH, TRANSFORMER_MODEL_PATH

def transformer_block(x, d_model, num_heads, rate=0.2):
    # Multi-Head Attention
    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(x_norm, x_norm)
    attn_output = layers.Dropout(rate)(attn_output)
    x = layers.Add()([x, attn_output]) # Res 1

    # Feed Forward Network
    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    ffn_output = layers.Dense(d_model * 4, activation='gelu')(x_norm)
    ffn_output = layers.Dense(d_model)(ffn_output)
    ffn_output = layers.Dropout(rate)(ffn_output)
    x = layers.Add()([x, ffn_output]) # Res 2

    return x

def build_model(input_shape=(8, 8, 2), d_model=64, num_blocks=4, num_heads=4):
    inputs = layers.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Reshape((64, 2))(inputs)
    x = layers.Dense(d_model)(x)

    positions = tf.range(start=0, limit=64, delta=1)
    row_indices = layers.Lambda(lambda p: p // 8)(positions)
    col_indices = layers.Lambda(lambda p: p % 8)(positions)

    row_emb = layers.Embedding(8, d_model, name="row_emb")(row_indices)
    col_emb = layers.Embedding(8, d_model, name="col_emb")(col_indices)
    x = layers.Add(name="add_pos_emb")([x, row_emb, col_emb])
    
    for i in range(num_blocks):
        x = transformer_block(x, d_model, num_heads, rate=0.2)

    # Policy Head
    policy_x = layers.Dense(d_model, activation='relu', name="policy_hidden")(x)
    policy_logits = layers.Dense(1, name="policy_logits_pre")(policy_x)
    policy_logits = layers.Reshape((64,))(policy_logits)
    policy_head = layers.Activation('softmax', name='policy', dtype='float32')(policy_logits)

    # Value Head
    value_x = layers.Conv1D(16, 1, activation='relu')(x)
    value_x = layers.Flatten()(value_x)
    value_x = layers.Dense(64, activation='relu', name="value_hidden")(value_x)
    value_head = layers.Dense(1, activation='tanh', name='value', dtype='float32')(value_x)

    model = models.Model(inputs=inputs, outputs=[policy_head, value_head])

    return model

if __name__ == '__main__':
    transformer_model = build_model()
    print("--- Transformer model architecture (Keras Functional API) ---")
    transformer_model.summary()

    os.makedirs(os.path.dirname(TRANSFORMER_MODEL_PATH), exist_ok=True)
    
    # transformer_model.save(TRANSFORMER_MODEL_PATH)
    print(f"New model -> {TRANSFORMER_MODEL_PATH}")