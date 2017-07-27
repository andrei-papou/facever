import tensorflow as tf


def chopra(inputs, embedding_size=128):
    w_init = None
    # 1 x 64 x 64
    net = tf.layers.conv2d(inputs, kernel_size=(7, 7), filters=15, strides=1, activation=tf.nn.relu, kernel_initializer=w_init)
    # 15 x 58 x 58
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2)
    # 15 x 29 x 29
    net = tf.layers.conv2d(net, kernel_size=(6, 6), filters=45, strides=1, activation=tf.nn.relu, kernel_initializer=w_init)
    # 45 x 24 x 24
    net = tf.layers.max_pooling2d(net, pool_size=(4, 4), strides=4)
    # 45 x 6 x 6
    net = tf.layers.conv2d(net, kernel_size=(6, 6), filters=256, strides=1, activation=tf.nn.relu, kernel_initializer=w_init)
    # 256 x 1 x 1
    net = tf.reshape(net, [-1, 256])
    return tf.layers.dense(net, units=embedding_size, activation=tf.nn.relu, kernel_initializer=w_init)
