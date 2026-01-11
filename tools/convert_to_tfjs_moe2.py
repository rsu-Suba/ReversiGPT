import os
import sys
import shutil
import subprocess
import tensorflow as tf
from tensorflow import keras

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Original MoE_2 to register custom objects for loading
try:
    from AI.models import MoE_2
    print("Successfully imported AI.models.MoE_2 (Original)")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

# Import New Definition for TFJS (with split tanh)
import moe2_tfjs_def
print("Successfully imported moe2_tfjs_def (New Structure)")

MODEL_PATH = 'models/TF/backup/MoE-2.keras'
TEMP_SAVED_MODEL_DIR = 'temp_saved_model_moe2'
OUTPUT_DIR = 'docs/tfjs_model'

def convert():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    print(f"Loading original model from {MODEL_PATH}...")
    
    try:
        # Load original model
        old_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Original model loaded successfully.")
        
        # Build new model with same config but split activation
        # We need to extract config from old model or use default if it matches
        # Assuming standard config for MoE-2
        # Let's verify config if possible, or just use default as per MoE_2.py
        
        print("Building new model structure...")
        new_model = moe2_tfjs_def.build_model()
        
        # Verify shapes compatibility (basic check)
        # We can try to set weights directly. 
        # Since the only change is splitting Dense(act='tanh') -> Dense() + Activation('tanh'),
        # the number of weights (kernels/biases) remains exactly the same.
        # The order of weights in get_weights() (flat list) should be identical.
        
        print(f"Old model params: {old_model.count_params()}")
        print(f"New model params: {new_model.count_params()}")
        
        if old_model.count_params() != new_model.count_params():
            print("WARNING: Parameter count mismatch! Weight transfer might fail.")
        
        print("Transferring weights...")
        new_model.set_weights(old_model.get_weights())
        print("Weights transferred.")
        
    except Exception as e:
        print(f"Failed during model processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Clean up temp dir
    if os.path.exists(TEMP_SAVED_MODEL_DIR):
        shutil.rmtree(TEMP_SAVED_MODEL_DIR)
        
    print(f"Saving new model to SavedModel format at {TEMP_SAVED_MODEL_DIR}...")
    
    try:
        new_model.export(TEMP_SAVED_MODEL_DIR)
        print("Model exported using model.export()")
    except AttributeError:
        print("model.export() not found, using tf.saved_model.save()")
        tf.saved_model.save(new_model, TEMP_SAVED_MODEL_DIR)

    print("Converting to TensorFlow.js graph model...")
    
    if shutil.which('tensorflowjs_converter') is None:
        print("Error: tensorflowjs_converter not found in PATH.")
        return

    # Clear output dir
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
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
    
    print(f"Running command: {' '.join(command)}")
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

    # Cleanup
    if os.path.exists(TEMP_SAVED_MODEL_DIR):
        shutil.rmtree(TEMP_SAVED_MODEL_DIR)

if __name__ == '__main__':
    convert()
