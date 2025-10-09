import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from tensorflow.keras import layers, models

def create_dual_resnet_model(input_shape=(8, 8, 2), num_residual_blocks=7):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(num_residual_blocks):
        residual = x
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])
        x = layers.ReLU()(x)

    policy_head = layers.Conv2D(1, (1, 1), padding='same')(x)
    policy_head = layers.BatchNormalization()(policy_head)
    policy_head = layers.ReLU()(policy_head)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Dense(64, activation='softmax', name='policy', dtype='float32')(policy_head)

    value_head = layers.Conv2D(1, (1, 1), padding='same')(x)
    value_head = layers.BatchNormalization()(value_head)
    value_head = layers.ReLU()(value_head)
    value_head = layers.Flatten()(value_head)
    value_head = layers.Dense(64, activation='relu')(value_head)
    value_head = layers.Dense(32, activation='relu')(value_head)
    value_head = layers.Dense(1, activation='tanh', name='value', dtype='float32')(value_head)

    model = models.Model(inputs=inputs, outputs=[policy_head, value_head])

    return model

if __name__ == '__main__':
    from config import SELF_PLAY_MODEL_PATH
    import os

    othello_ai_model = create_dual_resnet_model()
    print("--- AI model architecture ---")
    othello_ai_model.summary()

    os.makedirs(os.path.dirname(SELF_PLAY_MODEL_PATH), exist_ok=True)

    # othello_ai_model.save(SELF_PLAY_MODEL_PATH)
    print(f"New model -> {SELF_PLAY_MODEL_PATH}")