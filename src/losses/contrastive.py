import tensorflow as tf
from tensorflow.python.framework.function import Defun
from utils import compute_euclidian_distance_square


def contrastive_loss(margin, threshold=1e-5):

    @Defun(tf.float32, tf.float32, tf.float32, tf.float32)
    def backward(out1, out2, labels, diff):
        d_ = out1 - out2
        d_square = tf.reduce_sum(tf.square(d_), 1)
        d = tf.sqrt(d_square)

        minus = margin - d
        right_diff = minus / (d + threshold)
        right_diff = d_ * tf.reshape(right_diff * tf.to_float(tf.greater(minus, 0)), [-1, 1])

        batch_size = tf.to_float(tf.slice(tf.shape(out1), [0], [1]))
        data1_diff = diff * ((d_ + right_diff) * tf.reshape(labels, [-1, 1]) - right_diff) / batch_size
        data2_diff = -data1_diff
        return data1_diff, data2_diff, tf.zeros_like(labels)

    @Defun(tf.float32, tf.float32, tf.float32, grad_func=backward)
    def forward(out1, out2, labels):
        d_square = compute_euclidian_distance_square(out1, out2)
        d = tf.sqrt(d_square)
        similar_part = labels * d_square
        different_part = (1.0 - labels) * tf.square(tf.maximum(margin - d, 0.0))
        return tf.reduce_mean(similar_part + different_part) / 2

    return forward


def contrastive_loss_caffe(out1, out2, labels, margin):
    distance = compute_euclidian_distance_square(out1, out2)
    positive_part = labels * distance
    negative_part = (1 - labels) * tf.maximum(tf.square(margin) - distance, 0.0)
    return tf.reduce_mean(positive_part + negative_part) / 2
