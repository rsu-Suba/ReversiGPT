import sys
import os
import tensorflow as tf
import math
import glob
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AI.models.model_selector import try_load_model
from AI.config_loader import load_config
from AI.config import TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR

def _parse_function(example_proto):
    feature_description = {
        'input_planes': tf.io.FixedLenFeature([], tf.string),
        'policy': tf.io.FixedLenFeature([], tf.string),
        'value': tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    x = tf.io.parse_tensor(parsed['input_planes'], out_type=tf.float32)
    p = tf.io.parse_tensor(parsed['policy'], out_type=tf.float32)
    v = parsed['value']
    return tf.reshape(x, (8, 8, 2)), tf.reshape(p, (64,)), v

def create_val_dataset(files, batch_size):
    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, p, v: (x, (p, v)))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    print("--- MoE-1 Performance Evaluation ---")

    config = load_config(type('Args', (), {'model': 'moe-1'}))
    model_path = "./models/TF/MoE-1.H5"
    batch_size = 256

    print(f"Loading {model_path}...")
    model = try_load_model(model_path, config=config)

    tfrecord_dir = os.path.join(TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR, 'tfrecords', 'val')
    files = glob.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
    if not files:
        print(f"No validation files found in {tfrecord_dir}")
        return
    
    val_ds = create_val_dataset(files, batch_size)
    
    model.compile(
        loss=['categorical_crossentropy', 'mse'],
        metrics={
            model.output_names[0]: [
                tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='p1'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='p3')
            ],
            model.output_names[1]: [tf.keras.metrics.MeanAbsoluteError(name='mae')]
        }
    )

    print(f"Evaluating on {len(files)} files...")
    results = model.evaluate(val_ds, verbose=1)

    print("\n" + "="*30)
    print(f"MoE-1 Evaluation Results:")
    for name, val in zip(model.metrics_names, results):
        print(f"  {name:<15}: {val:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
