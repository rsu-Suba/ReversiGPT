
import optuna
import tensorflow as tf
import os
import glob
import math
import sys
from tensorflow.keras.callbacks import EarlyStopping
from optuna.integration import TFKerasPruningCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from AI.models.transformer_model import build_model
from train_loop import create_dataset, count_tfrecord_samples
from AI.config import TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR, BATCH_SIZE, EPOCHS

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 2e-4, log=True)
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.1)
    value_loss_weight = 1.0

    tfrecord_dir = os.path.join(TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR, 'tfrecords')
    train_tfrecord_dir = os.path.join(tfrecord_dir, 'train')
    val_tfrecord_dir = os.path.join(tfrecord_dir, 'val')
    train_tfrecord_files = glob.glob(os.path.join(train_tfrecord_dir, '*.tfrecord'))
    val_tfrecord_files = glob.glob(os.path.join(val_tfrecord_dir, '*.tfrecord'))

    if not train_tfrecord_files or not val_tfrecord_files:
        raise optuna.exceptions.TrialPruned("TFRecord files not found.")

    total_train_samples = getattr(objective, 'total_train_samples', None)
    if total_train_samples is None:
        objective.total_train_samples = count_tfrecord_samples(train_tfrecord_files)
        total_train_samples = objective.total_train_samples

    total_val_samples = getattr(objective, 'total_val_samples', None)
    if total_val_samples is None:
        objective.total_val_samples = count_tfrecord_samples(val_tfrecord_files)
        total_val_samples = objective.total_val_samples
    
    train_dataset = create_dataset(train_tfrecord_files, BATCH_SIZE, is_training=True, total_samples=total_train_samples)
    val_dataset = create_dataset(val_tfrecord_files, BATCH_SIZE, is_training=False)

    model = build_model()
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0, weight_decay=0.05)

    model.compile(
        optimizer=optimizer,
        loss={
            'policy': tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
            'value': 'mean_squared_error'
        },
        loss_weights={'policy': 1.0, 'value': value_loss_weight},
        metrics={
            'policy': [tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='1_accu'), tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='3_accu')],
            'value': 'mae'
        }
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
    study.optimize(objective, n_trials=10)
