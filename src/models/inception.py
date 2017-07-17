import tensorflow as tf
from tensorflow.python.ops.init_ops import random_normal_initializer


weights_init = random_normal_initializer(mean=0.0, stddev=0.001)


def block_5x5(inputs, filters=32):
    branch_5x5 = tf.layers.conv2d(inputs, kernel_size=(1, 1), strides=1, filters=filters, activation=tf.nn.relu,
                                  kernel_initializer=weights_init)
    branch_5x5 = tf.layers.conv2d(branch_5x5, kernel_size=(5, 5), strides=1, filters=filters,
                                  padding='same', kernel_initializer=weights_init, activation=tf.nn.relu)
    branch_3x3 = tf.layers.conv2d(inputs, kernel_size=(1, 1), strides=1, filters=filters, activation=tf.nn.relu)
    branch_3x3 = tf.layers.conv2d(branch_3x3, kernel_size=(3, 3), strides=1, filters=filters,
                                  padding='same', kernel_initializer=weights_init, activation=tf.nn.relu)
    branch_pool = tf.layers.max_pooling2d(inputs, pool_size=(2, 2), strides=1, padding='same')
    branch_pool = tf.layers.conv2d(branch_pool, kernel_size=(1, 1), strides=1, filters=filters, activation=tf.nn.relu,
                                   kernel_initializer=weights_init)
    branch_1x1 = tf.layers.conv2d(inputs, kernel_size=(1, 1), strides=1, filters=filters, activation=tf.nn.relu,
                                  kernel_initializer=weights_init)

    block = tf.concat([branch_5x5, branch_3x3, branch_1x1, branch_pool], axis=3)
    # block = tf.layers.conv2d(block, kernel_size=(1, 1), strides=1, filters=filters, activation=tf.nn.relu)

    return block


def block_3x3(inputs, filters=32):
    branch_3x3_2 = tf.layers.conv2d(inputs, kernel_size=(1, 1), strides=1, filters=filters, activation=tf.nn.relu,
                                    kernel_initializer=weights_init)
    branch_3x3_2 = tf.layers.conv2d(branch_3x3_2, kernel_size=(3, 3), strides=1, filters=filters,
                                    padding='same', activation=tf.nn.relu, kernel_initializer=weights_init)
    branch_3x3_2 = tf.layers.conv2d(branch_3x3_2, kernel_size=(3, 3), strides=1, filters=filters,
                                    padding='same', activation=tf.nn.relu, kernel_initializer=weights_init)
    branch_3x3 = tf.layers.conv2d(inputs, kernel_size=(1, 1), strides=1, filters=filters, activation=tf.nn.relu,
                                  kernel_initializer=weights_init)
    branch_3x3 = tf.layers.conv2d(branch_3x3, kernel_size=(3, 3), strides=1, filters=filters,
                                  padding='same', activation=tf.nn.relu, kernel_initializer=weights_init)
    branch_pool = tf.layers.max_pooling2d(inputs, pool_size=(2, 2), strides=1, padding='same')
    branch_pool = tf.layers.conv2d(branch_pool, kernel_size=(1, 1), strides=1, filters=filters, activation=tf.nn.relu,
                                   kernel_initializer=weights_init)
    branch_1x1 = tf.layers.conv2d(inputs, kernel_size=(1, 1), strides=1, filters=filters, activation=tf.nn.relu,
                                  kernel_initializer=weights_init)

    block = tf.concat([branch_3x3_2, branch_3x3, branch_pool, branch_1x1], axis=3)
    # block = tf.layers.conv2d(block, kernel_size=(1, 1), strides=1, filters=filters, activation=tf.nn.relu)

    return block


def inception(inputs, embedding_size=64):
    end_points = {}

    # 64 x 64 x 1
    net = tf.layers.conv2d(inputs, kernel_size=(3, 3), strides=1, filters=32, activation=tf.nn.relu,
                           kernel_initializer=weights_init)
    end_points['conv1'] = net

    # 62 x 62 x 32
    net = tf.layers.conv2d(net, kernel_size=(3, 3), strides=1, filters=64, activation=tf.nn.relu,
                           kernel_initializer=weights_init)
    end_points['conv2'] = net

    # 60 x 60 x 32
    net = tf.layers.conv2d(net, kernel_size=(3, 3), strides=1, filters=64, activation=tf.nn.relu,
                           kernel_initializer=weights_init)
    end_points['conv3'] = net

    # 58 x 58 x 64
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2)
    end_points['pool1'] = net

    # 29 x 29 x 64
    net = tf.layers.conv2d(net, kernel_size=(3, 3), strides=1, filters=80, activation=tf.nn.relu,
                           kernel_initializer=weights_init)
    end_points['conv4'] = net

    # 27 x 27 x 80
    for i in range(2):
        net = block_5x5(net)
        end_points['block_5x5_{}'.format(i)] = net

    # 27 x 27 x 128
    for i in range(3):
        net = block_3x3(net)
        end_points['block_3x3_{}'.format(i)] = net

    # 27 x 27 x 128
    net = tf.layers.max_pooling2d(net, pool_size=(9, 9), strides=9)
    end_points['pool2'] = net

    # 3 x 3 x 128
    net = tf.layers.conv2d(net, kernel_size=(1, 1), strides=1, filters=64, activation=tf.nn.relu,
                           kernel_initializer=weights_init)

    # 3 x 3 x 64
    net = tf.reshape(net, [-1, 3 * 3 * 64])
    net = tf.layers.dense(net, units=96, activation=tf.nn.relu, kernel_initializer=weights_init)
    end_points['dense1'] = net
    net = tf.layers.dense(net, units=embedding_size, activation=tf.nn.relu, kernel_initializer=weights_init)
    end_points['dense2'] = net

    return net
