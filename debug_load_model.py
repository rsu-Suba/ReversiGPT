import sys
import os
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI.models import transformer as transformer_model
from AI.models import static_MoE as static_moe_model
from AI.models import dynamic_MoE as dynamic_moe_model
from AI.models import switch_MoE as switch_moe_model
from AI.training.scheduler import WarmupCosineDecay

model_path = "./models/TF/MoE-1-optuna.h5"

print(f"Attempting to load {model_path} with debug info...")

print("\n--- Attempt 1: Transformer ---")
try:
    custom_objects = {
        'TokenAndPositionEmbedding': transformer_model.TokenAndPositionEmbedding,
        'TransformerBlock': transformer_model.TransformerBlock,
        'WarmupCosineDecay': WarmupCosineDecay
    }
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path)
        print("Success!")
except Exception as e:
    print(f"Failed: {e}")

print("\n--- Attempt 2: Dynamic MoE ---")
try:
    custom_objects = {
        'TokenAndPositionEmbedding': dynamic_moe_model.TokenAndPositionEmbedding,
        'MHA': dynamic_moe_model.MHA,
        'FFN': dynamic_moe_model.FFN,
        'DynamicAssembly': dynamic_moe_model.DynamicAssembly,
        'WarmupCosineDecay': WarmupCosineDecay
    }
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path)
        print("Success!")
except Exception as e:
    print(f"Failed: {e}")

print("\n--- Attempt 3: Static MoE ---")
try:
    custom_objects = {
        'TokenAndPositionEmbedding': static_moe_model.TokenAndPositionEmbedding,
        'TransformerBlock': static_moe_model.TransformerBlock,
        'ExpertSearch': static_moe_model.ExpertSearch,
        'ExpertThink': static_moe_model.ExpertThink,
        'WarmupCosineDecay': WarmupCosineDecay
    }
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path)
        print("Success!")
except Exception as e:
    print(f"Failed: {e}")
