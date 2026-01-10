import os
import sys
import tensorflow as tf
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def analyze_dataset(name):
    tfrecord_dir = os.path.join("data", name, 'tfrecords', 'train')
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrecord"))
    
    if not files:
        print(f"No TFRecords found for {name}.")
        return

    def _parse(proto):
        f = {'input_planes': tf.io.FixedLenFeature([], tf.string),
             'policy': tf.io.FixedLenFeature([], tf.string),
             'value': tf.io.FixedLenFeature([], tf.float32)}
        return tf.io.parse_single_example(proto, f)

    ds = tf.data.TFRecordDataset(files).map(_parse).take(1000)

    normal_count = 0
    reversed_count = 0
    total_valid = 0

    for record in ds:
        val = record['value'].numpy()
        planes = tf.io.parse_tensor(record['input_planes'], tf.float32).numpy()
        
        p_stones = np.sum(planes[:,:,0])
        o_stones = np.sum(planes[:,:,1])
        
        if abs(val) < 0.1 or p_stones == o_stones:
            continue
            
        total_valid += 1
        is_p_winning = p_stones > o_stones
        is_val_positive = val > 0
        
        if is_p_winning == is_val_positive:
            normal_count += 1
        else:
            reversed_count += 1

    print(f"\nDataset: {name}")
    print(f"  Valid samples analyzed: {total_valid}")
    print(f"  Normal correlation (Win=Positive): {normal_count}")
    print(f"  Reversed correlation (Win=Negative): {reversed_count}")
    
    if normal_count > reversed_count:
        print("  -> Result: Likely Normal")
    elif reversed_count > normal_count:
        print("  -> Result: Likely Reversed!")
    else:
        print("  -> Result: UNKNOWN")

if __name__ == "__main__":
    analyze_dataset("3G_god")