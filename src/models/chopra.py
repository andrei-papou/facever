import tensorflow as tf


w_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)


def chopra(inputs, embedding_size=128):
    # 1 x 64 x 64
    net = tf.layers.conv2d(inputs, kernel_size=(7, 7), filters=15, strides=1, activation=tf.nn.relu, kernel_initializer=w_init)
    # 15 x 58 x 58
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2)
    net = tf.layers.dropout(net, rate=0.1)
    # 15 x 29 x 29
    net = tf.layers.conv2d(net, kernel_size=(6, 6), filters=45, strides=1, activation=tf.nn.relu, kernel_initializer=w_init)
    # 45 x 24 x 24
    net = tf.layers.max_pooling2d(net, pool_size=(4, 4), strides=4)
    net = tf.layers.dropout(net, rate=0.1)
    # 45 x 6 x 6
    net = tf.layers.conv2d(net, kernel_size=(6, 6), filters=256, strides=1, activation=tf.nn.relu, kernel_initializer=w_init)
    # 256 x 1 x 1
    net = tf.reshape(net, [-1, 256])
    net = tf.layers.dense(net, units=512, activation=tf.nn.relu, kernel_initializer=w_init)
    net = tf.layers.dropout(net, rate=0.4)
    return tf.layers.dense(net, units=embedding_size, activation=tf.nn.relu, kernel_initializer=w_init)


def chopra_deep(inputs, embedding_size=128):
    # 1 x 64 x 64
    net = tf.layers.conv2d(inputs, kernel_size=(3, 3), filters=25, strides=1, activation=tf.nn.relu, kernel_initializer=w_init)
    # 25 x 62 x 62
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2)
    # 25 x 31 x 31
    net = tf.layers.conv2d(net, kernel_size=(4, 4), filters=50, strides=1, activation=tf.nn.relu, kernel_initializer=w_init)
    # 50 x 28 x 28
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2)
    # 50 x 14 x 14
    net = tf.layers.conv2d(net, kernel_size=(5, 5), filters=75, strides=1, activation=tf.nn.relu, kernel_initializer=w_init)
    # 75 x 10 x 10
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2)
    # 75 x 5 x 5
    net = tf.layers.conv2d(net, kernel_size=(5, 5), filters=256, strides=1, activation=tf.nn.relu, kernel_initializer=w_init)
    net = tf.reshape(net, [-1, 256])
    net = tf.layers.dense(net, units=512, activation=tf.nn.relu, kernel_initializer=w_init)
    return tf.layers.dense(net, units=embedding_size, activation=tf.nn.relu, kernel_initializer=w_init)
