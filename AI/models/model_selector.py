import os
import sys
import tensorflow as tf
import keras
import AI.models.MoE_1 as moe_1
import AI.models.MoE_2 as moe_2
import AI.models.transformer as transformer_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), '..')))

custom_objects = {
    'MHA': moe_1.MHA,
    'FFN': moe_1.FFN,
    'DynamicAssembly': moe_1.DynamicAssembly,
    # 'TokenAndPositionEmbedding': moe_1.TokenAndPositionEmbedding, # Comented out to fix 3G.h5 loading
    'TokenAndPositionEmbedding': transformer_model.TokenAndPositionEmbedding, # Use Transformer version
    
    # Register Transformer classes
    'TransformerBlock': transformer_model.TransformerBlock,
    
    'fast_gelu': moe_2.fast_gelu,
    'pos_embedding_logic': moe_2.pos_embedding_logic,
    'stone_count_logic': moe_2.stone_count_logic,
    'move_calc_logic': moe_2.move_calc_logic,
    'slice_prob_logic': moe_2.slice_prob_logic
}

# Fix: If the model uses the transformer's embedding, we might need a way to distinguish.
# But for now, let's add TransformerBlock which is the main missing piece.
# Actually, let's verify if TokenAndPositionEmbedding is needed.
# The error message only complained about 'TransformerBlock'.


def identify_architecture(model):
    layer_names = [l.name for l in model.layers]
    for name in layer_names:
        if 'moe_block' in name:
            return 'moe_2'
        
    all_layers = []
    def _collect(m):
        if hasattr(m, 'layers'):
            for l in m.layers:
                all_layers.append(l)
                _collect(l)
    _collect(model)
    
    for l in all_layers:
        type_name = type(l).__name__
        if 'DynamicAssembly' in type_name: return 'moe_1'
        if 'TransformerBlock' in type_name: return 'transformer'
        
    return 'unknown'

def try_load_model(model_path, config=None):
    # Try loading with default custom objects (Transformer version favored currently)
    try:
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path, compile=False)
            arch = identify_architecture(model)
            print(f"Successfully loaded model directly. Identified architecture: {arch.upper()} ({model_path})")
            return model
    except Exception as e:
        print(f"Direct load (Transformer preference) failed: {e}")
        
        # Retry with MoE-1 TokenAndPositionEmbedding
        print("Retrying with MoE-1 TokenAndPositionEmbedding...")
        modified_custom_objects = custom_objects.copy()
        modified_custom_objects['TokenAndPositionEmbedding'] = moe_1.TokenAndPositionEmbedding
        
        try:
            with tf.keras.utils.custom_object_scope(modified_custom_objects):
                model = tf.keras.models.load_model(model_path, compile=False)
                arch = identify_architecture(model)
                print(f"Successfully loaded model directly (MoE fallback). Identified architecture: {arch.upper()} ({model_path})")
                return model
        except Exception as e2:
            print(f"Direct load (MoE fallback) failed: {e2}")
            pass

    if config:
        try:
            print(f"Building model from config for {model_path}")
            model = create_model(config)
            _build_and_load_weights(model, model_path)
            return model
        except Exception as e:
            raise ValueError(f"Failed to build/load model from config: {e}")

    raise ValueError(f"Could not load model {model_path}. Direct load failed and no config provided.")

def _build_and_load_weights(model, weights_path):
    if not os.path.exists(weights_path):
        print(f"No weight file found at {weights_path}. Starting with fresh weights.")
        return

    dummy_board = tf.zeros((1, 8, 8, 2))
    try:
        model(dummy_board)
    except Exception:
        pass
        
    try:
        model.load_weights(weights_path, skip_mismatch=True)
        print(f"Weights restored from {weights_path}")
    except Exception as e:
        print(f"Failed to load weights from {weights_path}: {e}")
        raise e

def create_model(config):
    arch = config.get('arch', 'transformer').lower()
    if arch == 'moe_2':
        return moe_2.build_model(config)
    elif arch == 'moe_1':
        return moe_1.build_model(config)
    else:
        return transformer_model.build_model(
            d_model=config['embed_dim'], num_blocks=config['block'], num_heads=config['head']
        )
