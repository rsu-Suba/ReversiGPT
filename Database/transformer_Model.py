import os, sys
import tensorflow as tf, keras
from tensorflow.keras import layers, models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import SELF_PLAY_MODEL_PATH, TRANSFORMER_MODEL_PATH

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super(TokenAndPositionEmbedding, self).get_config()
        config.update({
            "position": self.position,
            "d_model": self.d_model,
        })
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                                     tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                                     d_model)
        angle_rads_sin = tf.sin(angle_rads[:, 0::2])
        angle_rads_cos = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([angle_rads_sin, angle_rads_cos], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def call(self, x):
        return x + tf.cast(self.pos_encoding[:, :tf.shape(x)[1], :], dtype=x.dtype)

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, rate=0.15, **kwargs):
        super().__init__(**kwargs)
        ff_dim = d_model * 4
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation='gelu'), layers.Dense(d_model),])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, input, training=False):
        x_f32 = tf.cast(input, tf.float32)

        # Multi-Head Attention
        normed_inputs = self.layernorm1(x_f32)
        attn_output = self.att(
            query=normed_inputs,
            value=normed_inputs,
            key=normed_inputs,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        x_f32 = x_f32 + tf.cast(attn_output, tf.float32)  # Res 1

        # Feed Forward Network
        normed_x = self.layernorm2(x_f32)
        ffn_output = self.ffn(normed_x)
        ffn_output = self.dropout2(ffn_output, training=training)
        x_f32 = x_f32 + tf.cast(ffn_output, tf.float32)  # Res 2

        return tf.cast(x_f32, input.dtype)


def build_model(input_shape=(8, 8, 2), d_model=256, num_transformer_blocks=6, num_heads=8):
    inputs = layers.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Reshape((64, 2))(inputs)
    x = layers.Dense(d_model)(x)
    x = TokenAndPositionEmbedding(64, d_model)(x)

    for _ in range(num_transformer_blocks):
        x = TransformerBlock(d_model, num_heads)(x)

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
    transformer_model = build_model()
    print("--- Transformer model architecture ---")
    transformer_model.summary()

    os.makedirs(os.path.dirname(TRANSFORMER_MODEL_PATH), exist_ok=True)

    transformer_model.save(TRANSFORMER_MODEL_PATH)
    print(f"New model -> {TRANSFORMER_MODEL_PATH}")
