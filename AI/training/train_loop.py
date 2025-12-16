import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import math
import glob
import json
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import mixed_precision
from tqdm import tqdm
from AI.config_loader import load_config
from AI.config import (
    TRAINING_DATA_DIR,
    CURRENT_GENERATION_DATA_SUBDIR,
    EPOCHS
)

config = load_config()
print(f"Loaded config for model: {config['model_name']}")

TRAINED_MODEL_SAVE_PATH = config.get('model_save_path', './models/TF/model.h5')
BATCH_SIZE = config.get('batch_size', 256)
learning_rate = config.get('learning_rate', 1e-4)
label_smoothing_value = config.get('label_smoothing', 0.04165291567423903)

IS_MOE = 'moe' in config['model_name'].lower()

if IS_MOE:
    from AI.models.dynamic_MoE import TokenAndPositionEmbedding, DynamicAssembly, build_model
    print("Using MoE Architecture")
else:
    from AI.models.transformer import TokenAndPositionEmbedding, TransformerBlock, build_model
    print("Using Standard Transformer Architecture")

mixed_precision.set_global_policy('mixed_float16')

@tf.keras.utils.register_keras_serializable()
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps, alpha=0.0):
        super(WarmupCosineDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha

    def __call__(self, step):
        return self.initial_learning_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha,
        }

def _parse_function(example_proto):
    feature_description = {
        'input_planes': tf.io.FixedLenFeature([], tf.string),
        'policy': tf.io.FixedLenFeature([], tf.string),
        'value': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    input_planes = tf.io.parse_tensor(parsed_features['input_planes'], out_type=tf.float32)
    policy = tf.io.parse_tensor(parsed_features['policy'], out_type=tf.float32)
    value = parsed_features['value']

    input_planes = tf.reshape(input_planes, (8, 8, 2))
    policy = tf.reshape(policy, (64,))

    return input_planes, policy, value

def _preprocess_and_augment(input_planes, policy, value):
    policy = tf.reshape(policy, (8, 8))
    policy_3d = policy[..., tf.newaxis]
    transform_idx = tf.random.uniform(shape=[], minval=0, maxval=8, dtype=tf.int32)

    def apply_transform(img, pol, idx):
        img, pol = tf.cond(idx >= 4, lambda: (tf.image.transpose(img), tf.image.transpose(pol)), lambda: (img, pol))
        idx = idx % 4
        img, pol = tf.cond(tf.logical_or(tf.equal(idx, 2), tf.equal(idx, 3)), lambda: (tf.image.flip_up_down(img), tf.image.flip_up_down(pol)), lambda: (img, pol))
        img, pol = tf.cond(tf.logical_or(tf.equal(idx, 1), tf.equal(idx, 3)), lambda: (tf.image.flip_left_right(img), tf.image.flip_left_right(pol)), lambda: (img, pol))
        
        return img, pol

    img, transformed_pol_3d = apply_transform(input_planes, policy_3d, transform_idx)
    transformed_pol_2d = tf.squeeze(transformed_pol_3d, axis=-1)
    
    transformed_pol_flat = tf.reshape(transformed_pol_2d, (64,))

    return img, transformed_pol_flat, value

def create_dataset(tfrecord_files, batch_size, is_training=True, total_samples=None):
    if not tfrecord_files:
        raise ValueError("No TFRecord")

    dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    if is_training:
        dataset = dataset.shuffle(len(tfrecord_files))

    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=tf.data.AUTOTUNE),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training
    )

    if is_training:
        buffer_size = min(total_samples, 100000) if total_samples else 50000
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat()

    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(_preprocess_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    dataset = dataset.map(
        lambda x, p, v: (x, {'policy': p, 'value': v}),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def count_tfrecord_samples(file_paths):
    print(f"Counting samples <- {len(file_paths)} files")
    total_count = 0
    for file_path in tqdm(file_paths, desc="Counting samples"):
        count = sum(1 for _ in tf.data.TFRecordDataset(file_path))
        total_count += count
    return total_count

if __name__ == "__main__":
    tfrecord_dir = os.path.join(TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR, 'tfrecords')
    train_tfrecord_dir = os.path.join(tfrecord_dir, 'train')
    val_tfrecord_dir = os.path.join(tfrecord_dir, 'val')

    train_tfrecord_files = glob.glob(os.path.join(train_tfrecord_dir, '*.tfrecord'))
    val_tfrecord_files = glob.glob(os.path.join(val_tfrecord_dir, '*.tfrecord'))

    if not train_tfrecord_files:
        print(f"No TFRecord ->{train_tfrecord_dir}")
        exit()

    print(f"Train TFRecord : {len(train_tfrecord_files)}, Val TFRecord : {len(val_tfrecord_files)}")

    total_train_samples = count_tfrecord_samples(train_tfrecord_files)
    train_dataset = create_dataset(train_tfrecord_files, BATCH_SIZE, is_training=True, total_samples=total_train_samples)

    val_dataset = None
    total_val_samples = 0
    if val_tfrecord_files:
        total_val_samples = count_tfrecord_samples(val_tfrecord_files)
        val_dataset = create_dataset(val_tfrecord_files, BATCH_SIZE, is_training=False)

    model = None
    if os.path.exists(TRAINED_MODEL_SAVE_PATH):
        print(f"Resuming training from model: {TRAINED_MODEL_SAVE_PATH}")
        custom_objects = {
            'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
            'TransformerBlock': TransformerBlock,
            'WarmupCosineDecay': WarmupCosineDecay
        }
        model = tf.keras.models.load_model(
            TRAINED_MODEL_SAVE_PATH, 
            custom_objects=custom_objects,
            compile=False
        )

    if model is None:
        print("No existing model found or failed to load. Creating a new model.")
        model = build_model(
            d_model=config['embed_dim'],
            num_blocks=config['block'],
            num_heads=config['head']
        )
        model.summary()

    initial_lr = learning_rate
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=initial_lr, clipnorm=1.0, weight_decay=0.05),
        loss={
            'policy': tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_value),
            'value': 'mean_squared_error'
        },
        loss_weights={'policy': 1.0, 'value': 0.8},
        metrics={
            'policy': [tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='1'), tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='3')],
            'value': 'mae'
        }
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-4,
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr_on_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )

    model_checkpoint_callback = ModelCheckpoint(
        filepath=TRAINED_MODEL_SAVE_PATH,
        save_weights_only=False,
        save_best_only=False,
        save_freq='epoch',
        verbose=1
    )

    print("\n--- Train start ---")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=math.ceil(total_train_samples / BATCH_SIZE),
        validation_data=val_dataset,
        validation_steps=math.ceil(total_val_samples / BATCH_SIZE) if val_dataset else None,
        callbacks=[
            early_stopping,
            reduce_lr_on_plateau,
            model_checkpoint_callback
        ]
    )

    print("\n--- Train finish -> Save new model ---")
    model.save(TRAINED_MODEL_SAVE_PATH)
    print(f"Best model saved -> {TRAINED_MODEL_SAVE_PATH}")
