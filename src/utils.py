import numpy as np
import tensorflow as tf


def compute_euclidian_distance_square(x1, x2):
    distance = tf.reduce_sum(tf.square(tf.subtract(x1, x2)), axis=1)
    return distance


def unison_shuffle(arrays, perm_length):
    permutation = np.random.permutation(perm_length)
    return [a[permutation] for a in arrays]
