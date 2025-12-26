
import optuna
import json
from tqdm import tqdm
import tensorflow as tf
import os
import glob
import math
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tensorflow.keras.callbacks import EarlyStopping
from optuna.integration import TFKerasPruningCallback
from AI.models.model_selector import try_load_model, create_model
from AI.training.scheduler import WarmupCosineDecay
from AI.config_loader import load_config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from train_loop import create_dataset, count_tfrecord_samples
from AI.config import TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR, EPOCHS

config = load_config()
print(f"Loaded config for model: {config['model_name']}")

BATCH_SIZE = config.get('batch_size', 256)

def count_tfrecord_samples(file_paths):
    print(f"Counting samples <- {len(file_paths)} files")
    total_count = 0
    for file_path in tqdm(file_paths, desc="Counting samples"):
        try:
            ds = tf.data.TFRecordDataset(file_path)
            count = ds.reduce(np.int64(0), lambda x, _: x + 1).numpy()
            total_count += count
        except Exception as e:
            print(f"Warning: Failed to count samples in {file_path}: {e}")
            
    return int(total_count)

def get_cached_sample_count(file_paths, cache_dir):
    cache_file = os.path.join(cache_dir, 'sample_count_cache.json')
    current_file_count = len(file_paths)
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            if cache_data.get('file_count') == current_file_count:
                print(f"Using cached sample count from {cache_file} (Samples: {cache_data.get('total_samples')})")
                return cache_data.get('total_samples')
        except Exception as e:
            print(f"Failed to read cache file: {e}")
            
    total_samples = count_tfrecord_samples(file_paths)
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'file_count': current_file_count,
                'total_samples': total_samples
            }, f)
        print(f"Updated sample count cache at {cache_file}")
    except Exception as e:
        print(f"Failed to write cache file: {e}")
        
    return total_samples

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 2e-4, log=True)
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.1)

    tfrecord_dir = os.path.join(TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR, 'tfrecords')
    train_tfrecord_dir = os.path.join(tfrecord_dir, 'train')
    val_tfrecord_dir = os.path.join(tfrecord_dir, 'val')
    train_tfrecord_files = glob.glob(os.path.join(train_tfrecord_dir, '*.tfrecord'))
    val_tfrecord_files = glob.glob(os.path.join(val_tfrecord_dir, '*.tfrecord'))

    if not train_tfrecord_files:
        print(f"No TFRecord ->{train_tfrecord_dir}")
        exit()

    print(f"Train TFRecord : {len(train_tfrecord_files)}, Val TFRecord : {len(val_tfrecord_files)}")

    total_train_samples = get_cached_sample_count(train_tfrecord_files, train_tfrecord_dir)
    train_dataset = create_dataset(train_tfrecord_files, BATCH_SIZE, is_training=True, total_samples=total_train_samples)

    val_dataset = None
    total_val_samples = 0
    if val_tfrecord_files:
        total_val_samples = get_cached_sample_count(val_tfrecord_files, val_tfrecord_dir)
        val_dataset = create_dataset(val_tfrecord_files, BATCH_SIZE, is_training=False)

    
    TRAINED_MODEL_SAVE_PATH = config.get('model_save_path', './models/TF/model.h5')
    model = None
    if os.path.exists(TRAINED_MODEL_SAVE_PATH):
        print(f"Resuming training from model: {TRAINED_MODEL_SAVE_PATH}")
        model = try_load_model(TRAINED_MODEL_SAVE_PATH)

    if model is None:
        print("No existing model found or failed to load. Creating a new model.")
        model = create_model(config)

    model.summary()
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0, weight_decay=0.1)

    model.compile(
        optimizer=optimizer,
        loss={
            'policy': tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
            'value': 'mean_squared_error'
        },
        loss_weights={'policy': 1.0, 'value': 0.8},
        metrics={
            'policy': [tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='1_accu'), tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='3_accu')],
            'value': 'mae'
        },
        jit_compile=True
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        TFKerasPruningCallback(trial, 'val_loss')
    ]

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=math.ceil(total_train_samples / BATCH_SIZE),
        validation_data=val_dataset,
        validation_steps=math.ceil(total_val_samples / BATCH_SIZE),
        callbacks=callbacks,
        verbose=1
    )

    val_mae = min(history.history['val_value_mae'])
    val_acc = max(history.history['val_policy_1_accu'])
    
    composite_score = val_mae - val_acc
    
    if math.isnan(composite_score):
        return float('inf')
        
    return composite_score

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize', study_name='hyperparam_tuning_v2', storage='sqlite:///db.sqlite3', load_if_exists=True)
    study.optimize(objective, n_trials=5)
