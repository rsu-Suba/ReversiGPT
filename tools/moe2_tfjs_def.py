import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from AI.config_loader import load_config

# Functions
# Note: Using the same package name "MoE_2" to be compatible with weights from original model
@tf.keras.utils.register_keras_serializable(package="MoE_2")
def fast_gelu(x):
    return tf.nn.gelu(x, approximate=True)

@tf.keras.utils.register_keras_serializable(package="MoE_2")
def pos_embedding_logic(args):
    target, r, c = args
    return target + tf.cast(r, target.dtype) + tf.cast(c, target.dtype)

@tf.keras.utils.register_keras_serializable(package="MoE_2")
def stone_count_logic(b):
    return tf.reduce_sum(b, axis=[1, 2, 3])

@tf.keras.utils.register_keras_serializable(package="MoE_2")
def move_calc_logic(c):
    return tf.maximum(tf.cast(c, tf.int32) - 4, 0)

@tf.keras.utils.register_keras_serializable(package="MoE_2")
def slice_prob_logic(p, i):
    return p[:, i:i+1]

# Token Embedding
def TokenAndPositionEmbedding(d_model):
    board_input = layers.Input(shape=(8, 8, 2), name="emb_board_in")
    x_reshaped = layers.Reshape((64, 2))(board_input)
    x_proj = layers.Dense(d_model, name="input_proj")(x_reshaped)

    # Position
    positions = tf.range(64)
    row_emb = layers.Embedding(8, d_model, name="row_emb")(positions // 8)
    col_emb = layers.Embedding(8, d_model, name="col_emb")(positions % 8)
    x_with_pos = layers.Lambda(pos_embedding_logic, name='add_pos')([x_proj, row_emb, col_emb])

    # Time step
    stone_count = layers.Lambda(stone_count_logic, name="stone_count")(board_input)
    current_moves = layers.Lambda(move_calc_logic, name="move_calc")(stone_count)
    time_emb = layers.Embedding(65, d_model, name="time_emb")(current_moves)
    time_emb_bc = layers.Reshape((1, d_model))(time_emb)

    x = layers.Add(name="embedding_sum")([x_with_pos, time_emb_bc])

    return models.Model(inputs=board_input, outputs=x, name="embedding")

# Experts weights (MHA / FFN)
def create_experts(d_model, num_heads, num_mha, num_ffn, block_idx, dropout_rate):
    mha_experts = []
    for i in range(num_mha):
        mha_experts.append(
            models.Sequential([
                layers.LayerNormalization(epsilon=1e-6, name=f"b{block_idx}_mha{i}_ln"),
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, name=f"b{block_idx}_mha{i}_att"),
                layers.Dropout(dropout_rate, name=f"b{block_idx}_mha{i}_do")
            ], name=f"b{block_idx}_mha_exp_{i}")
        )

    ffn_experts = []
    for i in range(num_ffn):
        ffn_experts.append(
            models.Sequential([
                layers.LayerNormalization(epsilon=1e-6, name=f"b{block_idx}_ffn{i}_ln"),
                layers.Dense(d_model * 4, activation=fast_gelu, name=f"b{block_idx}_ffn{i}_h"),
                layers.Dense(d_model, name=f"b{block_idx}_ffn{i}_o"),
                layers.Dropout(dropout_rate, name=f"b{block_idx}_ffn{i}_do")
            ], name=f"b{block_idx}_ffn_exp_{i}")
        )
    return mha_experts, ffn_experts

# Router for Experts
def create_router(num_options, router_dim, block_idx):
    return models.Sequential([
        layers.Dense(router_dim, activation=fast_gelu, name=f"b{block_idx}_router_h"),
        layers.Dense(num_options, name=f"b{block_idx}_router_l")
    ], name=f"b{block_idx}_router")

# MoE
def apply_moe_step(x, step_emb_layer, step_idx, router, mha_experts, ffn_experts):
    step_vec = step_emb_layer(tf.constant([step_idx], dtype=tf.int32))
    x_pooled = layers.GlobalAveragePooling1D(keepdims=False)(x)
    router_input = layers.Add()([x_pooled, step_vec])
    logits = router(router_input)
    probs = layers.Activation('softmax', dtype='float32')(logits * 10.0)

    mha_outs = []
    for exp in mha_experts:
        ln, att, do = exp.layers[0], exp.layers[1], exp.layers[2]
        mha_outs.append(do(att(ln(x), ln(x))))

    ffn_outs = []
    for exp in ffn_experts:
        ffn_outs.append(exp(x))

    total_experts = mha_outs + ffn_outs
    weighted_outs = []

    for i, expert_out in enumerate(total_experts):
        prob_slice = layers.Lambda(slice_prob_logic, arguments={'i': i}, name=f"slice_p_s{step_idx}_{i}")(probs)
        prob_bc = layers.Reshape((1, 1))(prob_slice)
        weighted_out = layers.Multiply()([expert_out, prob_bc])
        weighted_outs.append(weighted_out)

    summed_update = layers.Add()(weighted_outs) if len(weighted_outs) > 1 else weighted_outs[0]
    return layers.Add()([x, summed_update])

# MoE Transformer Block
def build_block_model(block_idx, d_model, num_heads, num_mha, num_ffn, steps, router_dim, dropout_rate):
    x_in = layers.Input(shape=(64, d_model), name=f"b{block_idx}_in")
    mha_experts, ffn_experts = create_experts(d_model, num_heads, num_mha, num_ffn, block_idx, dropout_rate)
    router = create_router(num_mha + num_ffn, router_dim, block_idx)
    step_emb_layer = layers.Embedding(steps, d_model, name=f"b{block_idx}_step_emb")

    curr_x = x_in
    for s in range(steps):
        curr_x = apply_moe_step(curr_x, step_emb_layer, s, router, mha_experts, ffn_experts)

    return models.Model(inputs=x_in, outputs=curr_x, name=f"moe_block_{block_idx}")

# Policy
def build_policy_head(d_model):
    x_in = layers.Input(shape=(64, d_model), name="p_in")
    x = layers.Dense(d_model, activation=fast_gelu, name="p_h")(x_in)
    x = layers.Dense(1, name="p_l")(x)
    x = layers.Reshape((64,))(x)
    out = layers.Activation('softmax', dtype='float32')(x)
    return models.Model(inputs=x_in, outputs=out, name="p")

# Value
def build_value_head(d_model):
    x_in = layers.Input(shape=(64, d_model), name="v_in")
    x = layers.Conv1D(d_model // 4, 1, activation=fast_gelu, name="v_c")(x_in)
    x = layers.Flatten(name="v_f")(x)
    x = layers.Dense(d_model, activation=fast_gelu, name="v_h")(x)
    
    # MODIFIED: Split Dense and Activation for WebGL compatibility
    # and use sigmoid formula to simulate tanh: tanh(x) = 2*sigmoid(2x) - 1
    # This prevents 'Tanh' op fusion which causes errors in TFJS WebGL backend.
    x = layers.Dense(1, dtype='float32')(x)
    out = layers.Lambda(lambda z: 2.0 * tf.math.sigmoid(2.0 * z) - 1.0, name='value_tanh_approx')(x)
    return models.Model(inputs=x_in, outputs=out, name="v")

# Model
def build_model(config=None):
    if config is None: config = {}
    d_model = config.get('embed_dim', 128)
    num_blocks = config.get('block', 4)
    num_heads = config.get('head', 8)
    num_mha = config.get('num_mha', 2)
    num_ffn = config.get('num_ffn', 2)
    steps = config.get('steps', 4)
    router_dim = config.get('router_dim', 64)
    dropout_rate = config.get('dropout', 0.1)

    main_input = layers.Input(shape=(8, 8, 2), dtype=tf.float32, name="main_input")
    x = TokenAndPositionEmbedding(d_model)(main_input)

    for b in range(num_blocks):
        x = build_block_model(b, d_model, num_heads, num_mha, num_ffn, steps, router_dim, dropout_rate)(x)

    policy = build_policy_head(d_model)(x)
    value = build_value_head(d_model)(x)

    return models.Model(inputs=main_input, outputs=[policy, value], name="MoE_2")
