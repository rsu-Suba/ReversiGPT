import os
import sys
import tensorflow as tf
import subprocess
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tools.moe1_tfjs_def as moe1_def
from AI.models import MoE_1 as original_moe1

MODEL_PATH = 'models/TF/MoE-1.h5'
TEMP_SAVED_MODEL_DIR = 'temp_saved_model_moe1'
OUTPUT_DIR = 'docs/tfjs_model'

def convert():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    print(f"Loading original model from {MODEL_PATH}...")
    try:
        # Load original model to get weights
        # Need custom objects for original model
        custom_objects = {
            'TokenAndPositionEmbedding': original_moe1.TokenAndPositionEmbedding,
            'MHA': original_moe1.MHA,
            'FFN': original_moe1.FFN,
            'DynamicAssembly': original_moe1.DynamicAssembly
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            original_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            
        print("Original model loaded.")
        
        # Configuration for MoE-1 (Fixed based on known specs or inspection)
        config = {
            'embed_dim': 128,
            'block': 4,
            'head': 4,
            'num_mha': 2,
            'num_ffn': 2,
            'steps': 2
        }
        
        print("Building new TFJS-compatible model...")
        new_model = moe1_def.build_moe1_model(config)
        
        # Build the model by passing dummy input
        new_model(tf.zeros((1, 8, 8, 2)))
        
        print(f"Original weights: {len(original_model.get_weights())}")
        print(f"New weights: {len(new_model.get_weights())}")
        
        if len(original_model.get_weights()) != len(new_model.get_weights()):
            print("Warning: Weight count mismatch. Attempting partial load or strict load.")
            # Usually strict load works if architecture is identical
        
        print("Copying weights...")
        new_model.set_weights(original_model.get_weights())
        print("Weights copied.")

    except Exception as e:
        print(f"Failed during model reconstruction: {e}")
        import traceback
        traceback.print_exc()
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