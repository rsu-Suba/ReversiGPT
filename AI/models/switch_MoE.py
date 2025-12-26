import tensorflow as tf
from tensorflow.keras import layers, models, Input

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, d_model, moves=64, **kwargs):
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
        current_moves = tf.clip_by_value(current_moves, 0, self.moves)

        t_emb = self.time_embedding(current_moves)
        t_emb = tf.expand_dims(t_emb, axis=1)
        
        return x + tf.cast(r_emb, x.dtype) + tf.cast(c_emb, x.dtype) + tf.cast(t_emb, x.dtype)

class PhaseExpert(layers.Layer):
    def __init__(self, d_model, num_heads, mha_count=1, ffn_count=2, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        ff_dim = d_model * 4
        self.mha_layers = [layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads) for _ in range(mha_count)]
        self.ffn_layers = [models.Sequential([layers.Dense(ff_dim, activation='gelu'), layers.Dense(d_model)]) for _ in range(ffn_count)]
        self.layernorms = [layers.LayerNormalization(epsilon=1e-6) for _ in range(mha_count + ffn_count)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, training=False):
        ln_idx = 0
        # Multi-Head Attention Layers
        for mha in self.mha_layers:
            normed = self.layernorms[ln_idx](x)
            attn_out = mha(normed, normed, training=training)
            x = x + self.dropout(attn_out, training=training)
            ln_idx += 1
        # Feed-Forward Layers
        for ffn in self.ffn_layers:
            normed = self.layernorms[ln_idx](x)
            ffn_out = ffn(normed)
            x = x + self.dropout(ffn_out, training=training)
            ln_idx += 1
        return x

class PhaseTransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, rate, **kwargs):
        super().__init__(**kwargs)
        self.early_expert = PhaseExpert(d_model, num_heads, mha_count=3, ffn_count=1, rate=rate, name="expert_early")
        self.mid_expert   = PhaseExpert(d_model, num_heads, mha_count=2, ffn_count=2, rate=rate, name="expert_mid")
        self.late_expert  = PhaseExpert(d_model, num_heads, mha_count=1, ffn_count=4, rate=rate, name="expert_late")

    def call(self, inputs, training=False):
            x, board = inputs
            target_dtype = x.dtype

            stone_count = tf.reduce_sum(board, axis=[1, 2, 3])
            moves = tf.cast(stone_count, tf.float32) - 4.0

            w_early = tf.nn.sigmoid(20.0 - moves)
            w_late  = tf.nn.sigmoid(moves - 40.0)
            w_mid   = 1.0 - (w_early + w_late)

            out_early = self.early_expert(x, training=training)
            out_mid   = self.mid_expert(x, training=training)
            out_late  = self.late_expert(x, training=training)

            w_early = tf.cast(tf.reshape(w_early, (-1, 1, 1)), target_dtype)
            w_mid   = tf.cast(tf.reshape(w_mid,   (-1, 1, 1)), target_dtype)
            w_late  = tf.cast(tf.reshape(w_late,  (-1, 1, 1)), target_dtype)

            out = (w_early * out_early +
                   w_mid   * out_mid +
                   w_late  * out_late)
            return out

def build_model(input_shape=(8, 8, 2), d_model=100, num_blocks=4, num_heads=8, dropout_rate=0.1):
    inputs = layers.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Reshape((64, 2))(inputs)
    x = layers.Dense(d_model)(x)
    x = TokenAndPositionEmbedding(d_model, 64)([x, inputs])

    for i in range(num_blocks):
        x = PhaseTransformerBlock(d_model, num_heads, rate=dropout_rate, name=f"block_{i}")([x, inputs])

    # Policy Head
    policy_x = layers.Dense(d_model, activation='relu')(x)
    policy_logits = layers.Dense(1)(policy_x)
    policy_logits = layers.Reshape((64,))(policy_logits)
    policy_head = layers.Activation('softmax', name='policy', dtype='float32')(policy_logits)

    # Value Head
    value_x = layers.Conv1D(16, 1, activation='relu')(x)
    value_x = layers.Flatten()(value_x)
    value_x = layers.Dense(128, activation='relu')(value_x)
    value_head = layers.Dense(1, activation='tanh', name='value', dtype='float32')(value_x)

    model = models.Model(inputs=inputs, outputs=[policy_head, value_head])
    return model

if __name__ == '__main__':
    model = build_model(d_model=100, num_blocks=8, num_heads=8)
    model.summary()
    print(f"Total Params: {model.count_params() / 1e6:.2f} M")