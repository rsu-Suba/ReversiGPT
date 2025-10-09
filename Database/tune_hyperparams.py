
import optuna
import tensorflow as tf
import os
import glob
import math
import sys
from tensorflow.keras.callbacks import EarlyStopping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Database.createModel import create_dual_resnet_model
from Database.trainModel import create_dataset, count_tfrecord_samples
from config import TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR, BATCH_SIZE, EPOCHS

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.3)
    value_loss_weight = trial.suggest_float('value_loss_weight', 0.5, 1.5)

    tfrecord_dir = os.path.join(TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR, 'tfrecords')
    train_tfrecord_dir = os.path.join(tfrecord_dir, 'train')
    val_tfrecord_dir = os.path.join(tfrecord_dir, 'val')
    train_tfrecord_files = glob.glob(os.path.join(train_tfrecord_dir, '*.tfrecord'))
    val_tfrecord_files = glob.glob(os.path.join(val_tfrecord_dir, '*.tfrecord'))

    if not train_tfrecord_files or not val_tfrecord_files:
        raise optuna.exceptions.TrialPruned("TFRecord files not found.")

    total_train_samples = getattr(objective, 'total_train_samples', None)
    if total_train_samples is None:
        objective.total_train_samples = count_tfrecord_samples(train_tfrecord_files) * 8
        total_train_samples = objective.total_train_samples

    total_val_samples = getattr(objective, 'total_val_samples', None)
    if total_val_samples is None:
        objective.total_val_samples = count_tfrecord_samples(val_tfrecord_files) * 8
        total_val_samples = objective.total_val_samples
    
    train_dataset = create_dataset(train_tfrecord_files, BATCH_SIZE, is_training=True, total_samples=total_train_samples)
    val_dataset = create_dataset(val_tfrecord_files, BATCH_SIZE, is_training=False)

    model = create_dual_resnet_model()
    
    if optimizer_name == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss={
            'policy_output': tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
            'value_output': 'mean_squared_error'
        },
        loss_weights={'policy_output': 1.0, 'value_output': value_loss_weight},
        metrics={
            'policy_output': [tf.keras.metrics.KLDivergence(name='kl_divergence'), tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')],
            'value_output': 'mean_absolute_error'
        }
    )

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=math.ceil(total_train_samples / BATCH_SIZE),
        validation_data=val_dataset,
        validation_steps=math.ceil(total_val_samples / BATCH_SIZE),
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)],
        verbose=1
    )

    val_loss = min(history.history['val_loss'])
    if math.isnan(val_loss):
        return float('inf')
        
    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value (min val_loss): ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

