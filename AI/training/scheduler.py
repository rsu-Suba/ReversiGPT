import tensorflow as tf
import numpy as np

@tf.keras.utils.register_keras_serializable()
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps, alpha=0.0):
        super(WarmupCosineDecay, self).__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.alpha = tf.cast(alpha, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # 1. Warmup phase
        def warmup():
            return self.initial_learning_rate * (step / self.warmup_steps)
        
        # 2. Cosine Decay phase
        def cosine_decay():
            completed_fraction = (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps)
            completed_fraction = tf.minimum(1.0, completed_fraction)
            cosine_decayed = 0.5 * (1.0 + tf.cos(np.pi * completed_fraction))
            decayed = (1.0 - self.alpha) * cosine_decayed + self.alpha
            return self.initial_learning_rate * decayed

        return tf.cond(step < self.warmup_steps, warmup, cosine_decay)

    def get_config(self):
        return {
            "initial_learning_rate": float(self.initial_learning_rate),
            "decay_steps": float(self.decay_steps),
            "warmup_steps": float(self.warmup_steps),
            "alpha": float(self.alpha),
        }
