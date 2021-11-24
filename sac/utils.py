import tensorflow as tf


def td_error(next_value, reward, terminal, discount):
    td = reward + (1.0 - terminal) * discount * next_value
    return td


def normalize_clip(data, mean, stddev, clip):
    stddev = tf.where(stddev < 1e-6, tf.cast(1.0, stddev.dtype), stddev)
    return tf.clip_by_value((data - mean) / stddev,
                            -clip, clip)


def clone_model(a, b, tau=1.0):
    for var_a, var_b in zip(a.trainable_variables, b.trainable_variables):
        var_b.assign(tau * var_a + (1.0 - tau) * var_b)
