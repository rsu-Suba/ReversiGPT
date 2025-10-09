import tensorflow as tf
from tensorflow.keras import layers, models
from config import MODELS_DIR
from Database.createModel import create_dual_resnet_model

model = create_dual_resnet_model()
old_model_path = f'{MODELS_DIR}/17G_07-25-25.h5'
model.load_weights(old_model_path)
new_model_path = f'{MODELS_DIR}/fixed_17G_model_savedmodel'
model.export(new_model_path)

print(f"Model has been re-saved to {new_model_path}")

import numpy as np

dummy_input = np.zeros((1, 8, 8, 2), dtype=np.float32)
policy_pred, value_pred = model.predict(dummy_input)

print("\n--- Model Output Test ---")
print("Policy Prediction (first 10 values):", policy_pred[0, :10])
print("Value Prediction:", value_pred[0])

