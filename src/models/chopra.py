import tensorflow as tf


def chopra(inputs, embedding_size=128):
    # reg = tf.contrib.layers.l2_regularizer(50.0)  # close to 60%
    reg = tf.contrib.layers.l1_regularizer(0.5)
    # reg = None
    w_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    # w_init = None
    # 1 x 64 x 64
    net = tf.layers.conv2d(
        inputs,
        kernel_size=(7, 7),
        filters=15,
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=w_init,
        kernel_regularizer=reg)
    # 15 x 58 x 58
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2)
    net = tf.layers.dropout(net, rate=0.25)
    # 15 x 29 x 29
    net = tf.layers.conv2d(
        net,
        kernel_size=(6, 6),
        filters=45,
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=w_init,
        kernel_regularizer=reg)
    # 45 x 24 x 24
    net = tf.layers.max_pooling2d(net, pool_size=(4, 4), strides=4)
    net = tf.layers.dropout(net, rate=0.25)
    # 45 x 6 x 6
    net = tf.layers.conv2d(
        net,
        kernel_size=(6, 6),
        filters=128,
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=w_init,
        kernel_regularizer=reg)
    # 256 x 1 x 1
    net = tf.reshape(net, [-1, 128])
    net = tf.layers.dense(net, units=256, activation=tf.nn.relu, kernel_regularizer=reg, kernel_initializer=w_init)
    net = tf.layers.dropout(net, rate=0.25)
    # net = tf.layers.dropout(net, rate=0.4)
    # net = tf.layers.dense(net, units=256, activation=tf.nn.relu, kernel_regularizer=reg, kernel_initializer=w_init)
    # net = tf.layers.dropout(net, rate=0.75)
    return tf.layers.dense(net, units=embedding_size, activation=tf.nn.relu, kernel_regularizer=reg, kernel_initializer=w_init)
