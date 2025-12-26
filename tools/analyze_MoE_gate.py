import tensorflow as tf
from tensorflow.keras import layers, models

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
    def __init__(self, d_model, num_heads, rate=0.1, **kwargs):
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
    def __init__(self, d_model, rate=0.1, **kwargs):
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
    def __init__(self, d_model, num_heads, num_mha=4, num_ffn=4, steps=8, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.steps = steps
        self.num_options = num_mha + num_ffn

        self.layer_pool = []

        for i in range(num_mha):
            self.layer_pool.append(MHA(d_model, num_heads, name=f"pool_mha_{i}"))
        for i in range(num_ffn):
            self.layer_pool.append(FFN(d_model, name=f"pool_ffn_{i}"))

        self.router_dense = layers.Dense(self.num_options, name="router")
        self.step_embedding = layers.Embedding(steps, d_model)

    def call(self, x, training=False):
        for i in range(self.steps):
            step_vec = self.step_embedding(tf.convert_to_tensor([i]))
            x_pooled = tf.reduce_mean(x, axis=1)

            router_input = x_pooled + tf.cast(step_vec, x_pooled.dtype)
            logits = self.router_dense(router_input)
            probs = tf.nn.softmax(logits, axis=-1)
            outputs = [layer(x, training=training) for layer in self.layer_pool]
            stacked_outputs = tf.stack(outputs, axis=1)

            probs_bc = tf.expand_dims(probs, axis=-1)
            probs_bc = tf.expand_dims(probs_bc, axis=-1)
            probs_bc = tf.cast(probs_bc, stacked_outputs.dtype)

            weighted_sum = tf.reduce_sum(stacked_outputs * probs_bc, axis=1)

            x = tf.cast(weighted_sum, x.dtype)
        return x
    
def build_model(input_shape=(8, 8, 2), d_model=32, num_blocks=2, num_heads=4):
    inputs = layers.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Reshape((64, 2))(inputs)
    x = layers.Dense(d_model)(x)
    x = TokenAndPositionEmbedding(d_model, 64)([x, inputs])

    for _ in range(num_blocks):
        x = DynamicAssembly(d_model, num_heads, num_mha=3, num_ffn=3, steps=8)(x)

    # # Policy Head
    # policy_logits = layers.Dense(1, name="policy_logits")(x)
    # policy_logits = layers.Reshape((64,))(policy_logits)
    # policy_head = layers.Activation('softmax', name='policy', dtype='float32')(policy_logits)

    # # Value Head
    # value_pooled = layers.GlobalAveragePooling1D()(x)
    # value_head = layers.Dense(1, activation='tanh', name='value', dtype='float32')(value_pooled)

    # Policy Head
    policy_x = layers.Dense(d_model, activation='relu', name="policy_hidden")(x)
    policy_logits = layers.Dense(1, name="policy_logits")(policy_x)
    policy_logits = layers.Reshape((64,))(policy_logits)
    policy_head = layers.Activation('softmax', name='policy', dtype='float32')(policy_logits)

    # Value Head
    value_x = layers.Flatten()(x)
    value_x = layers.Dense(256, activation='relu', name="value_hidden")(value_x)
    value_head = layers.Dense(1, activation='tanh', name='value', dtype='float32')(value_x)

    model = models.Model(inputs=inputs, outputs=[policy_head, value_head])

    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
    import numpy as np
    dummy_input = np.random.rand(1, 8, 8, 2).astype(np.float32)
    policy, value = model.predict(dummy_input)
    
    print("\n--- Output Check ---")
    print(f"Policy shape: {policy.shape} (Should be 1, 64)")
    print(f"Value shape:  {value.shape}  (Should be 1, 1)")
    print("Build successful!")