import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class CustomSchedule(LearningRateSchedule):
    """
    An object representing a learning rate schedule, i.e. a policy to adapt
    the learning rate along the model's optimization history.
    """
    def __init__(self, dModel, warmupSteps=4000):
        super().__init__()

        self.dModel = dModel
        self.dModel = tf.cast(self.dModel, dtype=tf.float32)
        self.warmupSteps = warmupSteps

    def __call__(self, step):
        """
        Builds the custom schedule's actual logic.
        """
        step = tf.cast(step, dtype=tf.float32)

        # Note: rsqrt is the reciprocal of sqrt (rsqrt(x) = 1/sqrt(x)), acting
        #       element-wise on tensors.
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmupSteps ** -1.5)

        return tf.math.rsqrt(self.dModel) * tf.math.minimum(arg1, arg2)
