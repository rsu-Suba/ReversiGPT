import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
import subprocess
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI.models.old_transformer import TokenAndPositionEmbedding, TransformerBlock as OriginalTransformerBlock
from AI.training.scheduler import WarmupCosineDecay

MODEL_PATH = 'models/TF/3G.h5'
TEMP_SAVED_MODEL_DIR = 'temp_saved_model'
OUTPUT_DIR = 'docs/tfjs_model'

def approximate_gelu(x):
    # x * sigmoid(1.5957691216057308 * (x + 0.044715 * x^3))
    inner = 1.5957691216057308 * (x + 0.044715 * tf.math.pow(x, 3))
    return x * tf.math.sigmoid(inner)

def approximate_tanh(x):
    # tanh(x) = 2 * sigmoid(2x) - 1
    return 2.0 * tf.math.sigmoid(2.0 * x) - 1.0

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.rate = rate
        
        ff_dim = d_model * 4
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation=approximate_gelu),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        x_f32 = tf.cast(inputs, tf.float32)
        normed_inputs = self.layernorm1(x_f32)
        attn_output = self.att(
            query=normed_inputs,
            value=normed_inputs,
            key=normed_inputs,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        x_f32 = x_f32 + tf.cast(attn_output, tf.float32)
        normed_x = self.layernorm2(x_f32)
        ffn_output = self.ffn(normed_x)
        x_f32 = x_f32 + tf.cast(ffn_output, tf.float32)
        return tf.cast(x_f32, inputs.dtype)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "rate": self.rate,
        })
        return config

def build_custom_model(input_shape=(8, 8, 2), d_model=256, num_blocks=8, num_heads=8):
    inputs = layers.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Reshape((64, 2))(inputs)
    x = layers.Dense(d_model)(x)
    x = TokenAndPositionEmbedding(64, d_model)(x)

    for _ in range(num_blocks):
        x = TransformerBlock(d_model, num_heads)(x)

    # Policy Head
    policy_logits = layers.Dense(1, name="policy_logits")(x)
    policy_logits = layers.Reshape((64,))(policy_logits)
    policy_head = layers.Activation('softmax', name='policy', dtype='float32')(policy_logits)

    # Value Head
    value_pooled = layers.GlobalAveragePooling1D()(x)
    value_head = layers.Dense(1, activation=approximate_tanh, name='value', dtype='float32')(value_pooled)

    model = models.Model(inputs=inputs, outputs=[policy_head, value_head])
    return model

def convert():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    print(f"Loading original model from {MODEL_PATH}...")
    try:
        custom_objects = {
            'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
            'TransformerBlock': OriginalTransformerBlock,
            'WarmupCosineDecay': WarmupCosineDecay
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            original_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            
        print("Original model loaded.")
        
        d_model = 256
        num_blocks = 0
        num_heads = 8
        
        for layer in original_model.layers:
            if isinstance(layer, TokenAndPositionEmbedding):
                d_model = layer.d_model
                print(f"Detected d_model: {d_model}")
            if isinstance(layer, OriginalTransformerBlock):
                num_blocks += 1
                if num_blocks == 1:
                    num_heads = layer.att.num_heads
                    print(f"Detected num_heads: {num_heads}")
        
        print(f"Detected num_blocks: {num_blocks}")
        
        print(f"Building custom model with d_model={d_model}, num_blocks={num_blocks}, num_heads={num_heads}...")
        new_model = build_custom_model(d_model=d_model, num_blocks=num_blocks, num_heads=num_heads)
        
        print(f"Original model weights count: {len(original_model.get_weights())}")
        print(f"New model weights count: {len(new_model.get_weights())}")
        
        if len(original_model.get_weights()) == len(new_model.get_weights()):
            print("Weights count match. Copying weights...")
            new_model.set_weights(original_model.get_weights())
        else:
            print("Weights count MISMATCH!")
            return

    except Exception as e:
        print(f"Failed during model reconstruction: {e}")
        return

    if os.path.exists(TEMP_SAVED_MODEL_DIR):
        shutil.rmtree(TEMP_SAVED_MODEL_DIR)
    
    print(f"Saving new model to SavedModel format at {TEMP_SAVED_MODEL_DIR}...")
    new_model.export(TEMP_SAVED_MODEL_DIR)

    print("Converting to TensorFlow.js graph model...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    command = [
        'tensorflowjs_converter',
        '--input_format=tf_saved_model',
        '--output_format=tfjs_graph_model',
        '--signature_name=serving_default',
        '--saved_model_tags=serve',
        TEMP_SAVED_MODEL_DIR,
        OUTPUT_DIR
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Conversion successful! Output saved to {OUTPUT_DIR}")
        else:
            print("Conversion failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Error executing converter: {e}")

    if os.path.exists(TEMP_SAVED_MODEL_DIR):
        shutil.rmtree(TEMP_SAVED_MODEL_DIR)

if __name__ == '__main__':
    convert()
