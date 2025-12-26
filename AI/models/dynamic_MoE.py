import tensorflow as tf
from tensorflow.keras import layers, models, Input

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, d_model, moves=64, **kwargs):
        if 'position' in kwargs:
            kwargs.pop('position')
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.d_model = d_model
        self.moves = moves
        self.row_embedding = layers.Embedding(8, d_model, name="row_emb")
        self.col_embedding = layers.Embedding(8, d_model, name="col_emb")
        self.time_embedding = layers.Embedding(moves + 1, d_model, name="time_emb")

    def call(self, inputs):
        x, board = inputs

        positions = tf.range(start=0, limit=64, delta=1, dtype=tf.int32)
        r_emb = self.row_embedding(positions // 8)
        c_emb = self.col_embedding(positions % 8)
        
        stone_count = tf.reduce_sum(board, axis=[1, 2, 3])
        current_moves = tf.cast(stone_count, tf.int32) - 4
        current_moves = tf.maximum(current_moves, 0)

        t_emb = self.time_embedding(current_moves)
        t_emb = tf.expand_dims(t_emb, axis=1)
        
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
        attn_output = self.dropout(attn_output, training=training)
        return x_f32 + tf.cast(attn_output, tf.float32)

class FFN(layers.Layer):
    def __init__(self, d_model, rate=0.2, **kwargs):
        super().__init__(**kwargs)
        ff_dim = d_model * 4
        self.ffn = models.Sequential([layers.Dense(ff_dim, activation='gelu'),layers.Dense(d_model)])
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(rate)
    
    def call(self, x, training=False):
        x_f32 = tf.cast(x, tf.float32)
        normed_inputs = self.layernorm(x_f32)
        ffn_output = self.ffn(normed_inputs)
        ffn_output = self.dropout(ffn_output, training=training)
        return x_f32 + tf.cast(ffn_output, tf.float32)

class DynamicAssembly(layers.Layer):
    def __init__(self, d_model, num_heads, num_mha=2, num_ffn=4, steps=1, rate=0.1, **kwargs):
        if 'num_options' in kwargs:
            kwargs.pop('num_options')
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_mha = num_mha
        self.num_ffn = num_ffn
        self.num_heads = num_heads
        self.steps = steps
        self.rate = rate
        
        # --- MHA Expert Pool ---
        self.mha_experts = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=rate)
            for _ in range(num_mha)
        ]
        self.mha_router = layers.Dense(num_mha, dtype='float32', name="mha_router")

        # --- FFN Expert Pool ---
        self.ffn_expert_w1 = self.add_weight(shape=(num_ffn, d_model, d_model * 2), name="fw1")
        self.ffn_expert_w2 = self.add_weight(shape=(num_ffn, d_model * 2, d_model), name="fw2")
        self.ffn_router = layers.Dense(num_ffn, dtype='float32', name="ffn_router")

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=False):
        # --- MHA Routing 1/2 ---
        norm_x1 = self.layernorm1(x)
        mha_logits = self.mha_router(tf.cast(tf.reduce_mean(norm_x1, axis=1), tf.float32))
        mha_probs = tf.nn.softmax(mha_logits)
        mha_idx = tf.argmax(mha_probs, axis=-1) # Top-1
        mha_outs = [expert(norm_x1, norm_x1, training=training) for expert in self.mha_experts]

        mha_stacked = tf.stack(mha_outs, axis=1)
        mha_stacked = tf.cast(mha_stacked, x.dtype)
        mha_final = tf.reduce_sum(mha_stacked * tf.reshape(tf.cast(mha_probs, x.dtype), [-1, self.num_mha, 1, 1]), axis=1)
        mha_final = tf.cast(self.dropout1(mha_final, training=training), x.dtype)

        x = x + mha_final

        # --- FFN Routing 2/4 ---
        norm_x2 = self.layernorm2(x)
        ffn_logits = self.ffn_router(tf.cast(tf.reduce_mean(norm_x2, axis=1), tf.float32))
        ffn_probs = tf.nn.softmax(ffn_logits)
        self.last_probs = ffn_probs

        k_ffn = 2 # Top-2
        topk_probs, topk_indices = tf.math.top_k(ffn_probs, k=k_ffn)
        topk_probs = topk_probs / tf.reduce_sum(topk_probs, axis=-1, keepdims=True)
        topk_probs = tf.cast(topk_probs, x.dtype)

        sw1 = tf.cast(tf.gather(self.ffn_expert_w1, topk_indices), x.dtype)
        sw2 = tf.cast(tf.gather(self.ffn_expert_w2, topk_indices), x.dtype)

        norm_x2 = tf.cast(norm_x2, x.dtype)
        norm_x2_tiled = tf.tile(tf.expand_dims(norm_x2, axis=1), [1, k_ffn, 1, 1])
        h = tf.nn.gelu(tf.einsum('bktd,bkdf->bktf', norm_x2_tiled, sw1))
        ffn_expert_outs = tf.einsum('bktf,bkfd->bktd', h, sw2)
        
        ffn_final = tf.reduce_sum(ffn_expert_outs * tf.reshape(topk_probs, [-1, k_ffn, 1, 1]), axis=1)
        ffn_final = tf.cast(self.dropout2(ffn_final, training=training), x.dtype)

        return x + ffn_final

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "num_heads": self.num_heads,
            "num_mha": self.num_mha, "num_ffn": self.num_ffn,
            "steps": self.steps, "rate": self.rate
        })
        return config

def build_model(input_shape=(8, 8, 2), d_model=32, num_blocks=2, num_heads=4, num_mha=2, num_ffn=2, steps=3, dropout_rate=0.2):
    inputs = layers.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Reshape((64, 2))(inputs)
    x = layers.Dense(d_model)(x)
    x = TokenAndPositionEmbedding(d_model, 64)([x, inputs])

    for _ in range(num_blocks):
        x = DynamicAssembly(d_model, num_heads, num_mha=num_mha, num_ffn=num_ffn, steps=steps, rate=dropout_rate)(x)
    
    # Policy Head
    policy_x = layers.Dense(d_model, activation='relu', name="policy_hidden")(x)
    policy_logits = layers.Dense(1, name="policy_logits")(policy_x)
    policy_logits = layers.Reshape((64,))(policy_logits)
    policy_head = layers.Activation('softmax', name='policy', dtype='float32')(policy_logits)

    # Value Head
    value_x = layers.Conv1D(64, 1, activation='relu')(x)
    value_x = layers.Flatten()(value_x)
    value_x = layers.Dense(64, activation='relu', name="value_hidden")(value_x)
    value_head = layers.Dense(1, activation='tanh', name='value', dtype='float32')(value_x)
    
    model = models.Model(inputs=inputs, outputs=[policy_head, value_head])

    return model

if __name__ == '__main__':
    # model = build_model(d_model=32, num_blocks=2, num_heads=4, num_mha=2, num_ffn=2, steps=1)
    model = build_model(d_model=164, num_blocks=6, num_heads=6, num_mha=2, num_ffn=3, steps=1)
    model.summary()
    print(f"Total Params: {model.count_params()}")