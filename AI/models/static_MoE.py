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

# M2 : F1
class ExpertSearch(layers.Layer):
    def __init__(self, d_model, num_heads, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        ff_dim = d_model * 4
        self.att1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.att2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.ffn = models.Sequential([layers.Dense(ff_dim, activation='gelu'),layers.Dense(d_model)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(rate)

    def call(self, x, training=False):
        x_f32 = tf.cast(x, tf.float32)

        # Multi Head Attention 1
        normed_inputs1 = self.layernorm1(x_f32)
        attn1_output = self.att1(
            query = normed_inputs1,
            value = normed_inputs1,
            key = normed_inputs1,
            training = training
        )
        attn1_output = self.dropout(attn1_output, training=training)
        x_f32 = x_f32 + tf.cast(attn1_output, tf.float32) # Res 1

        # Multi Head Attention 2
        normed_inputs2 = self.layernorm2(x_f32)
        attn2_output = self.att2(
            query = normed_inputs2,
            value = normed_inputs2,
            key = normed_inputs2,
            training = training
        )
        attn2_output = self.dropout(attn2_output, training=training)
        x_f32 = x_f32 + tf.cast(attn2_output, tf.float32) # Res 2

        # Feed Forward Network
        normed_x = self.layernorm3(x_f32)
        ffn_output = self.ffn(normed_x)
        ffn_output = self.dropout(ffn_output, training=training)
        x_f32 = x_f32 + tf.cast(ffn_output, tf.float32) # Res 3

        return tf.cast(x_f32, x.dtype)

# M1 : F3
class ExpertThink(layers.Layer):
    def __init__(self, d_model, num_heads, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        ff_dim = d_model * 4
        self.layernorm_att = layers.LayerNormalization(epsilon=1e-6)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.ffn1 = models.Sequential([layers.Dense(ff_dim, activation='gelu'), layers.Dense(d_model)])
        self.ffn2 = models.Sequential([layers.Dense(ff_dim, activation='gelu'), layers.Dense(d_model)])
        self.ffn3 = models.Sequential([layers.Dense(ff_dim, activation='gelu'), layers.Dense(d_model)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm4 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(rate)

    def call(self, x, training=False):
        x_f32 = tf.cast(x, tf.float32)

        # Multi Head Attention
        normed_inputs = self.layernorm1(x_f32)
        attn_output = self.att(
            query = normed_inputs,
            value = normed_inputs,
            key = normed_inputs,
            training = training,
        )
        attn_output = self.dropout(attn_output, training=training)
        x_f32 = x_f32 + tf.cast(attn_output, tf.float32) # Res 1

        #Feed Forward Network 1
        normed_x = self.layernorm2(x_f32)
        ffn_output = self.ffn1(normed_x)
        ffn_output = self.dropout(ffn_output, training=training)
        x_f32 = x_f32 + tf.cast(ffn_output, tf.float32) # Res 2

        #Feed Forward Network 2
        normed_x = self.layernorm3(x_f32)
        ffn_output = self.ffn2(normed_x)
        ffn_output = self.dropout(ffn_output, training=training)
        x_f32 = x_f32 + tf.cast(ffn_output, tf.float32) # Res 3

        #Feed Forward Network 1
        normed_x = self.layernorm4(x_f32)
        ffn_output = self.ffn3(normed_x)
        ffn_output = self.dropout(ffn_output, training=training)
        x_f32 = x_f32 + tf.cast(ffn_output, tf.float32) # Res 4

        return tf.cast(x_f32, x.dtype)

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.expert_search = ExpertSearch(d_model, num_heads)
        self.expert_think = ExpertThink(d_model, num_heads)
        self.gate_dense = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x, board = inputs

        stone_count = tf.reduce_sum(board, axis=[1, 2, 3])
        progress = tf.cast(stone_count, tf.float32) / 64.0
        progress = tf.expand_dims(progress, -1) # (Batch, 1)

        gate_val = self.gate_dense(progress)
        gate_val_bc = tf.expand_dims(gate_val, -1) # (Batch, 1, 1)

        out_search = self.expert_search(x, training=training)
        out_think = self.expert_think(x, training=training)

        return (1.0 - gate_val_bc) * out_search + gate_val_bc * out_think


def build_model(input_shape=(8, 8, 2), d_model=32, num_blocks=3, num_heads=4):
    inputs = layers.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Reshape((64, 2))(inputs)
    x = layers.Dense(d_model)(x)
    x = TokenAndPositionEmbedding(d_model, 64)([x, inputs])

    for _ in range(num_blocks):
        x = TransformerBlock(d_model, num_heads)([x, inputs])

    # Policy Head
    policy_logits = layers.Dense(1, name="policy_logits")(x)
    policy_logits = layers.Reshape((64,))(policy_logits)
    policy_head = layers.Activation('softmax', name='policy', dtype='float32')(policy_logits)

    # Value Head
    value_pooled = layers.GlobalAveragePooling1D()(x)
    value_head = layers.Dense(1, activation='tanh', name='value', dtype='float32')(value_pooled)

    model = models.Model(inputs=inputs, outputs=[policy_head, value_head])

    return model


if __name__ == '__main__':
    model = build_model(d_model=28, num_blocks=1, num_heads=4)
    model.summary()
    print(f"Total Params: {model.count_params()}")