import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import TRANSFORMER_MODEL_PATH

def get_positional_encoding(seq_len, d_model):
    position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
    
    # Apply sin to even indices in the array; 2i
    pe_sin = tf.sin(position * div_term)
    # Apply cos to odd indices in the array; 2i+1
    pe_cos = tf.cos(position * div_term)
    
    # Interleave sin and cos components
    pe = tf.reshape(tf.concat([pe_sin, pe_cos], axis=-1), [seq_len, d_model])
    return tf.constant(pe)

if __name__ == '__main__':
    custom_objects = {'get_positional_encoding': get_positional_encoding}
    try:
        model = tf.keras.models.load_model(TRANSFORMER_MODEL_PATH, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully!")
        model.summary()
    except Exception as e:
        print(f"Model loading failed: {e}")
