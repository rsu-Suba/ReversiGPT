import tensorflow as tf
from tensorflow.keras import layers, models

# WebGL Safe Activation Functions
def approximate_gelu(x):
    # x * sigmoid(1.5957691216057308 * (x + 0.044715 * x^3))
    inner = 1.5957691216057308 * (x + 0.044715 * tf.math.pow(x, 3))
    return x * tf.math.sigmoid(inner)

def approximate_tanh(x):
    # tanh(x) = 2 * sigmoid(2x) - 1
    return 2.0 * tf.math.sigmoid(2.0 * x) - 1.0

# Components
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, d_model, moves=64, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.moves = moves
        self.row_embedding = layers.Embedding(8, d_model, name="row_emb")
        self.col_embedding = layers.Embedding(8, d_model, name="col_emb")
        self.time_embedding = layers.Embedding(moves + 1, d_model, name="time_emb")

    def call(self, inputs):
        # inputs: [x (reshaped board), board (original)]
        # However, functional model usually passes tensor flow.
        # In original build_model: x = TokenAndPositionEmbedding(...) ([x, inputs])
        x, board = inputs

        positions = tf.range(start=0, limit=64, delta=1, dtype=tf.int32)
        r_emb = self.row_embedding(positions // 8)
        c_emb = self.col_embedding(positions % 8)
        
        # Calculate stone count for time embedding
        stone_count = tf.reduce_sum(board, axis=[1, 2, 3])
        current_moves = tf.cast(stone_count, tf.int32) - 4
        current_moves = tf.maximum(current_moves, 0)

        t_emb = self.time_embedding(current_moves)
        t_emb = tf.expand_dims(t_emb, axis=1) # [Batch, 1, d_model]
        
        return x + tf.cast(r_emb, x.dtype) + tf.cast(c_emb, x.dtype) + tf.cast(t_emb, x.dtype)

class MHA(layers.Layer):
    def __init__(self, d_model, num_heads, rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(rate)
    
    def call(self, x, training=False):
        x_f32 = tf.cast(x, tf.float32)
        normed_inputs = self.layernorm(x_f32)
        attn_output = self.att(
            query = normed_inputs,
            value = normed_inputs,
            key = normed_inputs,
            training = training
        )
        # Dropout is ignored in inference usually, but explicit training=False helps
        attn_output = self.dropout(attn_output, training=training)
        return x_f32 + tf.cast(attn_output, tf.float32)

class FFN(layers.Layer):
    def __init__(self, d_model, rate=0.2, **kwargs):
        super().__init__(**kwargs)
        ff_dim = d_model * 4
        # Use approx gelu
        self.ffn_dense1 = layers.Dense(ff_dim, activation=approximate_gelu)
        self.ffn_dense2 = layers.Dense(d_model)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(rate)
    
    def call(self, x, training=False):
        x_f32 = tf.cast(x, tf.float32)
        normed_inputs = self.layernorm(x_f32)
        
        ffn_out = self.ffn_dense1(normed_inputs)
        ffn_out = self.ffn_dense2(ffn_out)
        
        ffn_out = self.dropout(ffn_out, training=training)
        return x_f32 + tf.cast(ffn_out, tf.float32)

class DynamicAssembly(layers.Layer):
    def __init__(self, d_model, num_heads, num_mha=4, num_ffn=4, steps=8, rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.steps = steps
        self.num_options = num_mha + num_ffn
        self.layer_pool = []
        for i in range(num_mha):
            self.layer_pool.append(MHA(d_model, num_heads, rate, name=f"pool_mha_{i}"))
        for i in range(num_ffn):
            self.layer_pool.append(FFN(d_model, rate, name=f"pool_ffn_{i}"))
            
        self.router_dense = layers.Dense(self.num_options, name="router")
        self.step_embedding = layers.Embedding(steps, d_model)

    def call(self, x, training=False):
        # Unroll the loop for graph export stability
        for i in range(self.steps):
            step_vec = self.step_embedding(tf.constant([i], dtype=tf.int32)) # [1, d_model]
            x_pooled = tf.reduce_mean(x, axis=1) # [Batch, d_model]

            router_input = x_pooled + tf.cast(step_vec, x_pooled.dtype)
            logits = self.router_dense(router_input)
            probs = tf.nn.softmax(logits, axis=-1) # [Batch, num_options]
            
            # Execute all experts
            # To make it TFJS friendly, we compute all and weight them.
            # Conditional execution (like Switch) is hard in Keras layers for export.
            
            outputs = [layer(x, training=training) for layer in self.layer_pool]
            stacked_outputs = tf.stack(outputs, axis=1) # [Batch, num_options, 64, d_model]

            # Broadcast probs: [Batch, num_options, 1, 1]
            probs_bc = tf.reshape(probs, [-1, self.num_options, 1, 1])
            probs_bc = tf.cast(probs_bc, stacked_outputs.dtype)

            weighted_sum = tf.reduce_sum(stacked_outputs * probs_bc, axis=1)

            x = tf.cast(weighted_sum, x.dtype)
        return x

def build_moe1_model(config):
    d_model = config.get('embed_dim', 128)
    num_blocks = config.get('block', 4)
    num_heads = config.get('head', 4)
    num_mha = config.get('num_mha', 2)
    num_ffn = config.get('num_ffn', 2)
    steps = config.get('steps', 2)
    dropout_rate = config.get('dropout', 0.2)
    input_shape = (8, 8, 2)

    inputs = layers.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Reshape((64, 2))(inputs)
    x = layers.Dense(d_model)(x)
    x = TokenAndPositionEmbedding(d_model, 64)([x, inputs])

    for i in range(num_blocks):
        x = DynamicAssembly(d_model, num_heads, num_mha=num_mha, num_ffn=num_ffn, steps=steps, rate=dropout_rate, name=f"moe_block_{i}")(x)
    
    # Policy Head
    policy_x = layers.Dense(d_model, activation=approximate_gelu, name="policy_hidden")(x)
    policy_logits = layers.Dense(1, name="policy_logits")(policy_x)
    policy_logits = layers.Reshape((64,))(policy_logits)
    policy_head = layers.Activation('softmax', name='policy', dtype='float32')(policy_logits)

    # Value Head
    value_x = layers.Conv1D(8, 1, activation=approximate_gelu)(x)
    value_x = layers.Flatten()(value_x)
    value_x = layers.Dense(128, activation=approximate_gelu, name="value_hidden")(value_x)
    # Use approximate tanh as activation for the final Dense layer
    value_head = layers.Dense(1, activation=approximate_tanh, name='value', dtype='float32')(value_x)
    
    model = models.Model(inputs=inputs, outputs=[policy_head, value_head])

    return model
