import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from absl import logging
logging.set_verbosity(logging.ERROR)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from AI.log_filter import install_log_filter
install_log_filter()
import math
import glob
import json
import datetime
import tensorflow as tf
import numpy as np
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import mixed_precision
from tqdm import tqdm
from AI.models.model_selector import try_load_model, create_model
from AI.training.scheduler import WarmupCosineDecay
from AI.config_loader import load_config
from AI.config import (
    TRAINING_DATA_DIR,
    CURRENT_GENERATION_DATA_SUBDIR,
    EPOCHS
)

# Renamed God Data Directory
GOD_DATA_DIR = 'data_pro'
SELF_PLAY_WINDOW_SIZE = 5

config = load_config()
print(f"Loaded config for model: {config['model_name']}")

# Determine save path: Env Var > Config > Default
TRAINED_MODEL_SAVE_PATH = os.environ.get('MODEL_SAVE_PATH', config.get('model_save_path', './models/TF/MoE-2-Candidate.keras'))
print(f"Model will be saved to: {TRAINED_MODEL_SAVE_PATH}")

# Determine load path: Env Var > Save Path (Default behavior)
TRAINED_MODEL_LOAD_PATH = os.environ.get('MODEL_LOAD_PATH', TRAINED_MODEL_SAVE_PATH)
print(f"Model will be loaded from: {TRAINED_MODEL_LOAD_PATH}")

BATCH_SIZE = config.get('batch_size', 256)
learning_rate = config.get('learning_rate', 5e-5)
label_smoothing_value = config.get('label_smoothing', 0.04165291567423903)
weight_decay = config.get('weight_decay', 0.05)

try:
    mixed_precision.set_global_policy('mixed_float16')
    print("Using mixed_float16 policy.")
except Exception as e:
    print(f"Warning: Failed to set mixed_float16 policy: {e}")

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

def load_tfrecords(file_pattern, limit_recent=None):
    files = glob.glob(file_pattern)
    if not files: return None
    files.sort(key=os.path.getmtime, reverse=True)
    
    if limit_recent and len(files) > limit_recent:
        print(f"Limiting {file_pattern} to recent {limit_recent} files.")
        files = files[:limit_recent]
        
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.shuffle(len(files))
    ds = ds.interleave(
        lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=tf.data.AUTOTUNE).cache(),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    return ds

def create_mixed_dataset(self_play_pattern, god_pattern, batch_size, is_training=True):
    ds_self = load_tfrecords(self_play_pattern, limit_recent=SELF_PLAY_WINDOW_SIZE)
    ds_god = load_tfrecords(god_pattern)
    
    if ds_self is None and ds_god is None: raise ValueError("No training data found.")
    
    if ds_self and ds_god:
        # Mix ratio: 10% Self-Play, 90% God-Tier (data_pro) - Conservative growth
        ds_self_rep = ds_self.repeat()
        ds_god_rep = ds_god.repeat()
        dataset = tf.data.Dataset.sample_from_datasets([ds_self_rep, ds_god_rep], weights=[0.1, 0.9])
    elif ds_self:
        dataset = ds_self
    else:
        dataset = ds_god

    if is_training:
        dataset = dataset.shuffle(buffer_size=50000)
        dataset = dataset.repeat()

    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(_preprocess_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    
    # Map using 'p' and 'v' keys as requested
    dataset = dataset.map(lambda x, p, v: (x, {'p': p, 'v': v}), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def count_tfrecord_samples(file_pattern, limit_recent=None):
    files = glob.glob(file_pattern)
    files.sort(key=os.path.getmtime, reverse=True)
    if limit_recent and len(files) > limit_recent: files = files[:limit_recent]
    print(f"Counting samples in {len(files)} files ({file_pattern})...")
    total_count = 0
    for file_path in tqdm(files, desc="Counting"):
        try: total_count += sum(1 for _ in tf.data.TFRecordDataset(file_path))
        except: pass
    return int(total_count)

if __name__ == "__main__":
    self_play_train = os.path.join(TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR, 'tfrecords', 'train', '*.tfrecord')
    self_play_val = os.path.join(TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR, 'tfrecords', 'val', '*.tfrecord')
    god_train = os.path.join(TRAINING_DATA_DIR, GOD_DATA_DIR, 'tfrecords', 'train', '*.tfrecord')
    god_val = os.path.join(TRAINING_DATA_DIR, GOD_DATA_DIR, 'tfrecords', 'val', '*.tfrecord')

    print("--- Training with Mixed Data (20% Self-Play + 80% God-Tier) [Window=5] ---")
    
    n_self = count_tfrecord_samples(self_play_train, limit_recent=SELF_PLAY_WINDOW_SIZE)
    n_god = count_tfrecord_samples(god_train)
    total_samples = n_self + n_god
    print(f"Samples: Self-Play={n_self}, Data-Pro={n_god}, Total={total_samples}")

    train_dataset = create_mixed_dataset(self_play_train, god_train, BATCH_SIZE, is_training=True)
    val_dataset = create_mixed_dataset(self_play_val, god_val, BATCH_SIZE, is_training=False)
    
    steps_per_epoch = math.ceil(total_samples / BATCH_SIZE)

    model = None
    if os.path.exists(TRAINED_MODEL_LOAD_PATH):
        print(f"Resuming training from model: {TRAINED_MODEL_LOAD_PATH}")
        try:
            model = try_load_model(TRAINED_MODEL_LOAD_PATH)
        except:
            print("Failed to load existing model. Creating new model.")
            model = None

    if model is None:
        model = create_model(config)

    model.summary()

    total_epochs = EPOCHS
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = min(steps_per_epoch * 2, 1000)

    lr_schedule = WarmupCosineDecay(
        initial_learning_rate=learning_rate * 0.5,
        decay_steps=total_steps,
        warmup_steps=warmup_steps,
        alpha=0.01
    )

    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, clipnorm=1.0, weight_decay=weight_decay)

    # Use 'p', 'v' keys as requested
    model.compile(
        optimizer=optimizer,
        loss={
            'p': tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_value),
            'v': 'mean_squared_error'
        },
        loss_weights={'p': 1.0, 'v': 1.0},
        metrics={
            'p': [
                tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='1'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='3')
            ],
            'v': [tf.keras.metrics.MeanAbsoluteError(name='m')]
        },
        jit_compile=True
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=TRAINED_MODEL_SAVE_PATH, save_weights_only=False, save_best_only=False, save_freq='epoch', verbose=1)
    ]
    
    print("\n--- Starting Training ---")
    try:
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=math.ceil((n_self*0.1 + n_god*0.1) / BATCH_SIZE) if val_dataset else None,
            callbacks=callbacks
        )
        print("\n--- Training Finished ---")
        model.save(TRAINED_MODEL_SAVE_PATH)
        print(f"Final model saved to {TRAINED_MODEL_SAVE_PATH}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        model.save(TRAINED_MODEL_SAVE_PATH)
        print("Model saved.")
