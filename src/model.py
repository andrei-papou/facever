import math
import tensorflow as tf
from tensorflow.python.ops.init_ops import random_normal_initializer
from utils import compute_euclidian_distance_square, unison_shuffle


class Model:

    def __init__(self, width, height, channels, var_scope='diplom_model_scope'):
        self.width = width
        self.height = height
        self.channels = channels
        self.var_scope = var_scope
        self.net_vars_created = None

    def _get_output_ten(self, inputs_ph, embedding_dimension):
        with tf.variable_scope(self.var_scope, reuse=self.net_vars_created):
            if self.net_vars_created is None:
                self.net_vars_created = True

            inputs = tf.reshape(inputs_ph, [-1, self.width, self.height, 1])
            weights_init = random_normal_initializer(mean=0.0, stddev=0.1)

            # returns 60 x 60 x 15
            net = tf.layers.conv2d(
                inputs=inputs,
                filters=15,
                kernel_size=(5, 5),
                strides=1,
                padding='valid',
                kernel_initializer=weights_init,
                activation=tf.nn.relu)
            # returns 30 x 30 x 15
            net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=2)
            # returns 24 x 24 x 45
            net = tf.layers.conv2d(
                inputs=net,
                filters=45,
                kernel_size=(7, 7),
                strides=1,
                padding='valid',
                kernel_initializer=weights_init,
                activation=tf.nn.relu)
            # returns 6 x 6 x 45
            net = tf.layers.max_pooling2d(inputs=net, pool_size=(4, 4), strides=4)
            # returns 1 x 1 x 250
            net = tf.layers.conv2d(
                inputs=net,
                filters=250,
                kernel_size=(6, 6),
                strides=1,
                kernel_initializer=weights_init,
                activation=tf.nn.relu)
            net = tf.reshape(net, [-1, 1 * 1 * 250])
            net = tf.layers.dense(
                inputs=net,
                units=256,
                kernel_initializer=weights_init,
                activation=tf.nn.sigmoid)
            net = tf.layers.dense(
                inputs=net,
                units=embedding_dimension,
                kernel_initializer=weights_init,
                activation=tf.nn.sigmoid)

            net = tf.check_numerics(net, message='model')

        return net

    @staticmethod
    def _get_loss_op(output1, output2, labels, margin):
        labels = tf.to_float(labels)
        d_sqr = compute_euclidian_distance_square(output1, output2)
        loss_non_reduced = labels * d_sqr + (1 - labels) * tf.square(tf.maximum(0., margin - d_sqr))
        return 0.5 * tf.reduce_mean(tf.cast(loss_non_reduced, dtype=tf.float64))

    @staticmethod
    def _get_accuracy_op(out1, out2, labels, margin):
        distances = tf.sqrt(compute_euclidian_distance_square(out1, out2))
        gt_than_margin = tf.cast(tf.maximum(tf.subtract(distances, margin), 0.0), dtype=tf.bool)
        predictions = tf.cast(gt_than_margin, dtype=tf.int32)
        return tf.reduce_mean(tf.cast(tf.not_equal(predictions, labels), dtype=tf.float32))

    def _monitor(self, sess: tf.Session, x1s, x2s, ys, margin, embedding_dimension,
                 training_loss=False, training_accuracy=False):
        if not (training_loss or training_accuracy):
            return

        dataset_size = ys.shape[0]
        input1_ph = tf.placeholder(dtype=tf.float32, shape=(dataset_size, self.width, self.height))
        input2_ph = tf.placeholder(dtype=tf.float32, shape=(dataset_size, self.width, self.height))
        labels_ph = tf.placeholder(dtype=tf.int32, shape=(dataset_size,))

        output1 = self._get_output_ten(input1_ph, embedding_dimension)
        output2 = self._get_output_ten(input2_ph, embedding_dimension)

        feed_dict = {input1_ph: x1s, input2_ph: x2s, labels_ph: ys}

        if training_loss:
            loss = tf.reduce_sum(self._get_loss_op(output1, output2, labels_ph, margin))
            ep_average_loss = sess.run(loss, feed_dict=feed_dict)
            print('Loss: {:.5f}'.format(ep_average_loss))

        if training_accuracy:
            accuracy = self._get_accuracy_op(output1, output2, labels_ph, margin)
            correctly_predicted = sess.run(accuracy, feed_dict=feed_dict)
            print('Accuracy: {:.3f}'.format(correctly_predicted))

    def train(self, x1s, x2s, ys, num_epochs, mini_batch_size, learning_rate, embedding_dimension, margin,
              monitor_training_loss=False, monitor_training_accuracy=False):
        input1_ph = tf.placeholder(dtype=tf.float32, shape=(mini_batch_size, self.width, self.height))
        input2_ph = tf.placeholder(dtype=tf.float32, shape=(mini_batch_size, self.width, self.height))
        labels_ph = tf.placeholder(dtype=tf.int32, shape=(mini_batch_size,))
        output1 = self._get_output_ten(input1_ph, embedding_dimension)
        output2 = self._get_output_ten(input2_ph, embedding_dimension)

        loss = self._get_loss_op(output1, output2, labels_ph, margin)
        loss = tf.Print(loss, [loss], message='loss')
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        num_batches = int(math.ceil(ys.shape[0] / mini_batch_size))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for ep in range(num_epochs):
                x1s, x2s, ys = unison_shuffle([x1s, x2s, ys], ys.shape[0])

                for bt_num in range(num_batches):
                    bt_slice = slice(bt_num * mini_batch_size, (bt_num + 1) * mini_batch_size)
                    sess.run(train_op, feed_dict={
                        input1_ph: x1s[bt_slice],
                        input2_ph: x2s[bt_slice],
                        labels_ph: ys[bt_slice]
                    })
                print('Epoch {} training complete'.format(ep))

                self._monitor(
                    sess=sess,
                    x1s=x1s,
                    x2s=x2s,
                    ys=ys,
                    margin=margin,
                    embedding_dimension=embedding_dimension,
                    training_loss=monitor_training_loss,
                    training_accuracy=monitor_training_accuracy)
