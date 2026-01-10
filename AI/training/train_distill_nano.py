import os
import sys
import math
import glob
import tensorflow as tf
import numpy as np
from keras import mixed_precision
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from AI.models.model_selector import try_load_model, create_model, custom_objects
from AI.config_loader import load_config
from AI.config import TRAINING_DATA_DIR, CURRENT_GENERATION_DATA_SUBDIR
import submission.nano_tf as nano_tf

TEACHER_MODEL_PATH = "./models/TF/MoE-1.h5"
STUDENT_MODEL_NAME = "nano-v2"
BATCH_SIZE = 512
LEARNING_RATE = 1e-5
EPOCHS = 30
STEPS_PER_EPOCH = 3000
TEMPERATURE = 0.8 

def load_teacher():
    print(f"[Distill] Loading Teacher Model: {TEACHER_MODEL_PATH}")
    teacher_config = {'arch': 'moe_1'}
    model = try_load_model(TEACHER_MODEL_PATH, config={'arch': 'moe_1', 'embed_dim': 128})
    model.trainable = False
    return model

def get_dataset(subdir, batch_size):
    data_dir = os.path.join(TRAINING_DATA_DIR, 'MoE_1', 'tfrecords', 'train')
    if not os.path.exists(data_dir):
        data_dir = os.path.join(TRAINING_DATA_DIR, subdir, 'tfrecords', 'train')
        
    files = glob.glob(os.path.join(data_dir, '*.tfrecord'))
    if not files:
        data_dir = os.path.join(TRAINING_DATA_DIR, 'MoE_1', 'tfrecords')
        files = glob.glob(os.path.join(data_dir, '*.tfrecord'))
        
    if not files:
        raise FileNotFoundError(f"No TFRecords found in {data_dir}")
    
    print(f"[Distill] Using data from: {data_dir} ({len(files)} files)")

    def _parse(proto):
        f = {'input_planes': tf.io.FixedLenFeature([], tf.string),
             'policy': tf.io.FixedLenFeature([], tf.string),
             'value': tf.io.FixedLenFeature([], tf.float32)}
        p = tf.io.parse_single_example(proto, f)
        inp = tf.io.parse_tensor(p['input_planes'], tf.float32)
        inp.set_shape([8, 8, 2])
        pol = tf.io.parse_tensor(p['policy'], tf.float32)
        pol.set_shape([64])
        return (inp, pol, p['value'])

    ds = tf.data.TFRecordDataset(files).map(_parse).shuffle(20000).repeat().batch(batch_size)
    return iter(ds)

def distill():
    mixed_precision.set_global_policy('mixed_float16')
    config = load_config(type('Args', (), {'model': STUDENT_MODEL_NAME}))
    teacher = load_teacher()

    if STUDENT_MODEL_NAME == "nano-v2":
        print("[Distill] Using Nano-v2 (submission.nano_tf) architecture")
        student = nano_tf.build_model(config)
    else:
        student = try_load_model(config['model_save_path'], config=config)
        if student is None:
            print("[Distill] Creating new student model")
            student = create_model(config)
    
    student(tf.zeros((1, 8, 8, 2)))
    data_iter = get_dataset(CURRENT_GENERATION_DATA_SUBDIR, BATCH_SIZE)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)
    mse = tf.keras.losses.MeanSquaredError()
    kl_loss = tf.keras.losses.KLDivergence()

    @tf.function(jit_compile=True)
    def train_step(x, y_p, y_v):
        t_p, t_v = teacher(x, training=False)
        t_v_normalized = t_v
        with tf.GradientTape() as tape:
            s_p, s_v = student(x, training=True)
            soft_t_p = tf.nn.softmax(tf.math.log(t_p + 1e-9) / TEMPERATURE)
            loss_distill_p = kl_loss(soft_t_p, s_p)
            loss_distill_v = mse(t_v_normalized, s_v)
            total_loss = loss_distill_p + loss_distill_v
            
        grads = tape.gradient(total_loss, student.trainable_variables)
        optimizer.apply_gradients(zip(grads, student.trainable_variables))
        return total_loss, loss_distill_p, loss_distill_v

    print(f"\n--- Distillation: {STUDENT_MODEL_NAME} learning from MoE-1 ---")
    
    for epoch in range(EPOCHS):
        pbar = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch+1}/{EPOCHS}")
        avg_loss = 0
        
        for _ in pbar:
            x, y_p, y_v = next(data_iter)
            loss, lp, lv = train_step(x, y_p, y_v)
            avg_loss += loss
            pbar.set_postfix({'loss': f"{loss:.4f}", 'p': f"{lp:.4f}", 'v': f"{lv:.4f}"})
        
        save_path = config['model_save_path']
        student.save(save_path)
        print(f" Saved: {save_path} | Avg Loss: {avg_loss/STEPS_PER_EPOCH:.4f}")

if __name__ == "__main__":
    distill()
