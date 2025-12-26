import os
import sys
import tensorflow as tf
import AI.models.transformer as transformer_model
import AI.models.static_MoE as static_moe_model
import AI.models.dynamic_MoE as dynamic_moe_model
import AI.models.switch_MoE as switch_moe_model
import AI.models.old_dynamic_MoE as old_dynamic_moe_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AI.training.scheduler import WarmupCosineDecay

def try_load_model(model_path):
    try:
        custom_objects = {
            'TokenAndPositionEmbedding': transformer_model.TokenAndPositionEmbedding,
            'TransformerBlock': transformer_model.TransformerBlock
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded model as Transformer: {model_path}")
            return model
    except Exception as e:
        pass

    try:
        custom_objects = {
            'TokenAndPositionEmbedding': dynamic_moe_model.TokenAndPositionEmbedding,
            'MHA': dynamic_moe_model.MHA,
            'FFN': dynamic_moe_model.FFN,
            'DynamicAssembly': dynamic_moe_model.DynamicAssembly
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded model as Dynamic MoE: {model_path}")
            return model
    except Exception as e:
        pass

    try:
        custom_objects = {
            'TokenAndPositionEmbedding': old_dynamic_moe_model.TokenAndPositionEmbedding,
            'MHA': old_dynamic_moe_model.MHA,
            'FFN': old_dynamic_moe_model.FFN,
            'DynamicAssembly': old_dynamic_moe_model.DynamicAssembly
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded model as Old Dynamic MoE (Backup): {model_path}")
            return model
    except Exception as e:
        pass

    try:
        custom_objects = {
            'TokenAndPositionEmbedding': static_moe_model.TokenAndPositionEmbedding,
            'TransformerBlock': static_moe_model.TransformerBlock,
            'ExpertSearch': static_moe_model.ExpertSearch,
            'ExpertThink': static_moe_model.ExpertThink
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded model as Static MoE: {model_path}")
            return model
    except Exception as e:
        pass

    try:
        custom_objects = {
            'TokenAndPositionEmbedding': switch_moe_model.TokenAndPositionEmbedding,
            'PhaseExpert': switch_moe_model.PhaseExpert,
            'PhaseTransformerBlock': switch_moe_model.PhaseTransformerBlock,
            'build_model': switch_moe_model.build_model,
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded model as Switch MoE: {model_path}")
            return model
    except Exception as e:
        pass

    raise ValueError(f"Could not load model {model_path} with any known architecture.")

def create_model(config):
    arch = config.get('arch', 'transformer').lower()
    print(f"Creating model architecture: {arch}")

    if arch == 'switch':
        return switch_moe_model.build_model(
            d_model=config['embed_dim'],
            num_blocks=config['block'],
            num_heads=config['head'],
            dropout_rate=config.get('dropout_rate', 0.25)
        )
    elif arch == 'dynamic':
        return dynamic_moe_model.build_model(
            d_model=config['embed_dim'],
            num_blocks=config['block'],
            num_heads=config['head'],
            num_mha=config.get('num_mha', 2),
            num_ffn=config.get('num_ffn', 2),
            steps=config.get('steps', 3),
            dropout_rate=config.get('dropout_rate', 0.25)
        )
    elif arch == 'static':
        return static_moe_model.build_model(
            d_model=config['embed_dim'],
            num_blocks=config['block'],
            num_heads=config['head'],
            dropout_rate=config.get('dropout_rate', 0.1)
        )
    else:
        return transformer_model.build_model(
            d_model=config['embed_dim'],
            num_blocks=config['block'],
            num_heads=config['head']
        )