
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import math
import glob
import json
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras import mixed_precision
from tqdm import tqdm
from AI.models.transformer_model import TokenAndPositionEmbedding, TransformerBlock, build_model
from AI.config import (
    TRAINING_DATA_DIR,
    TRAINED_MODEL_SAVE_PATH,
    CURRENT_GENERATION_DATA_SUBDIR,
    EPOCHS,
    BATCH_SIZE,
    learning_rate
)
mixed_precision.set_global_policy('mixed_float16')

@tf.keras.utils.register_keras_serializable()
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps, alpha=0.0):
        super(WarmupCosineDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.cosine_decay = CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps - warmup_steps,
            alpha=alpha
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_percent_done = step / self.warmup_steps
        warmup_learning_rate = self.initial_learning_rate * warmup_percent_done

        is_warmup = step < self.warmup_steps

        learning_rate = tf.cond(
            is_warmup,
            lambda: warmup_learning_rate,
            lambda: self.cosine_decay(step - self.warmup_steps)
        )
        return learning_rate

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

    transforms = [
        lambda img, pol: (img, pol),
        lambda img, pol: (tf.image.flip_left_right(img), tf.image.flip_left_right(pol)),
        lambda img, pol: (tf.image.flip_up_down(img), tf.image.flip_up_down(pol)),
        lambda img, pol: (tf.image.flip_left_right(tf.image.flip_up_down(img)), tf.image.flip_left_right(tf.image.flip_up_down(pol))),
        lambda img, pol: (tf.image.transpose(img), tf.image.transpose(pol)),
        lambda img, pol: (tf.image.flip_left_right(tf.image.transpose(img)), tf.image.flip_left_right(tf.image.transpose(pol))),
        lambda img, pol: (tf.image.flip_up_down(tf.image.transpose(img)), tf.image.flip_up_down(tf.image.transpose(pol))),
        lambda img, pol: (tf.image.flip_left_right(tf.image.flip_up_down(tf.image.transpose(img))), tf.image.flip_left_right(tf.image.flip_up_down(tf.image.transpose(pol))))
    ]

    augmented_images = []
    augmented_policies = []
    augmented_values = []

    for transform_func in transforms:
        img, transformed_pol_3d = transform_func(input_planes, policy_3d)
        transformed_pol_2d = tf.squeeze(transformed_pol_3d, axis=-1)
        augmented_images.append(img)
        augmented_policies.append(transformed_pol_2d)
        augmented_values.append(value)

    images = tf.stack(augmented_images)
    policies = tf.reshape(tf.stack(augmented_policies), (8, 64))
    values = tf.stack(augmented_values)

    return images, policies, values

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
    dataset = dataset.unbatch()
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
    total_train_samples *= 8
    train_dataset = create_dataset(train_tfrecord_files, BATCH_SIZE, is_training=True, total_samples=total_train_samples)
    
    val_dataset = None
    total_val_samples = 0
    if val_tfrecord_files:
        total_val_samples = count_tfrecord_samples(val_tfrecord_files)
        total_val_samples *= 8
        val_dataset = create_dataset(val_tfrecord_files, BATCH_SIZE, is_training=False)

    initial_learning_rate = learning_rate
    steps_per_epoch = math.ceil(total_train_samples / BATCH_SIZE)
    TARGET_DECAY_EPOCHS = 40
    decay_steps = steps_per_epoch * TARGET_DECAY_EPOCHS
    warmup_epochs = 5
    warmup_steps = warmup_epochs * steps_per_epoch

    lr_schedule = WarmupCosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        alpha=0.000005
    )

    model = None
    if os.path.exists(TRAINED_MODEL_SAVE_PATH):
        print(f"Resuming training from model: {TRAINED_MODEL_SAVE_PATH}")
        custom_objects = {
            'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
            'TransformerBlock': TransformerBlock,
            'WarmupCosineDecay': WarmupCosineDecay
        }
        model = tf.keras.models.load_model(
            TRAINED_MODEL_SAVE_PATH, custom_objects=custom_objects
        )

    if model is None:
        print("No existing model found or failed to load. Creating a new model.")
        model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, clipnorm=1.0, weight_decay=0.1),
        loss={
            'policy': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.028968626957636873),
            'value': 'mean_squared_error'
        },
        loss_weights={'policy': 1.0, 'value': 0.5544360525425724},
        metrics={
            'policy': [tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='1'), tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='3')],
            'value': 'mae'
        }
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-5,
        patience=6,
        restore_best_weights=True,
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
            model_checkpoint_callback
        ]
    )

    print("\n--- Train finish -> Save new model ---")
    model.save(TRAINED_MODEL_SAVE_PATH)
    print(f"Best model saved -> {TRAINED_MODEL_SAVE_PATH}")