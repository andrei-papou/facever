import os
from time import time
import numpy as np
import tensorflow as tf
from constants import DATA_DIRECTORY


def compute_euclidian_distance_square(x1, x2):
    distance = tf.reduce_sum(tf.square(tf.subtract(x1, x2)), axis=1)
    return distance


def unison_shuffle(arrays, perm_length):
    permutation = np.random.permutation(perm_length)
    return [np.array(a)[permutation] for a in arrays]


def generate_model_id():
    return str(int(time()))


def get_model_file_path(model_id):
    return os.path.join(DATA_DIRECTORY, 'model_{}.ckpt'.format(model_id))
