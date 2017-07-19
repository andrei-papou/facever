import math
import tensorflow as tf
from models.inception import inception
from models.chopra import chopra
from losses.contrastive import contrastive_loss, contrastive_loss_caffe
from utils import compute_euclidian_distance_square, unison_shuffle, generate_model_id, get_model_file_path
from config import TRAINING_EVAL_DATA_SLICE


class Model:

    def __init__(self, width, height, channels, model_id=None, saved_model=None, var_scope='diplom_model_scope'):
        self.width = width
        self.height = height
        self.channels = channels
        self.var_scope = var_scope
        self.net_vars_created = None
        self.model_path = get_model_file_path(model_id or generate_model_id())
        self.saved_model_path = get_model_file_path(saved_model) if saved_model is not None else None
        self.best_accuracy = 0

    def _get_output_ten(self, inputs_ph, embedding_dimension):
        with tf.variable_scope(self.var_scope, reuse=self.net_vars_created):
            if self.net_vars_created is None:
                self.net_vars_created = True

            inputs = tf.reshape(inputs_ph, [-1, self.width, self.height, 1])
            net = chopra(inputs, embedding_dimension)
            net = tf.check_numerics(net, message='model')
        return net

    @staticmethod
    def _get_loss_op(output1, output2, labels, margin):
        return contrastive_loss_caffe(output1, output2, labels, margin=margin)

    @staticmethod
    def _get_accuracy_op(out1, out2, labels, margin):
        distances = tf.sqrt(compute_euclidian_distance_square(out1, out2))
        gt_than_margin = tf.cast(tf.maximum(tf.subtract(distances, margin), 0.0), dtype=tf.bool)
        predictions = tf.cast(gt_than_margin, dtype=tf.float32)
        return tf.reduce_sum(tf.cast(tf.not_equal(predictions, labels), dtype=tf.int32))

    def _monitor(self, sess: tf.Session, x1s, x2s, ys, margin, embedding_dimension, mini_batch_size, prefix):
        dataset_size = ys.shape[0]
        input1_ph = tf.placeholder(dtype=tf.float32, shape=(mini_batch_size, self.width, self.height))
        input2_ph = tf.placeholder(dtype=tf.float32, shape=(mini_batch_size, self.width, self.height))
        labels_ph = tf.placeholder(dtype=tf.float32, shape=(mini_batch_size,))

        output1 = self._get_output_ten(input1_ph, embedding_dimension)
        output2 = self._get_output_ten(input2_ph, embedding_dimension)

        num_batches = int(math.ceil(dataset_size / mini_batch_size))

        loss = 0
        correctly_predicted = 0

        accuracy_ten = self._get_accuracy_op(output1, output2, labels_ph, margin)
        batch_loss = tf.reduce_sum(self._get_loss_op(output1, output2, labels_ph, margin))

        for i in range(num_batches):
            bt_slice = slice(i * mini_batch_size, (i + 1) * mini_batch_size)
            feed_dict = {
                input1_ph: x1s[bt_slice],
                input2_ph: x2s[bt_slice],
                labels_ph: ys[bt_slice]
            }
            correctly_predicted += sess.run(accuracy_ten, feed_dict=feed_dict)
            loss += sess.run(batch_loss, feed_dict=feed_dict)

        print('{} loss:       {}'.format(prefix, loss / num_batches))
        accuracy = correctly_predicted / dataset_size
        print('{} accuracy:   {}'.format(prefix, accuracy))
        print('')  # new line

        # TODO: move from training accuracy to validation accuracy if everything goes well
        if prefix == 'Training' and accuracy > self.best_accuracy:
            self._save(sess)
            self.best_accuracy = accuracy

    def _load_or_initialize(self, sess):
        if self.saved_model_path is None:
            sess.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(sess, self.saved_model_path)

    def _save(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, self.model_path)

    def train(self, x1s, x2s, ys, num_epochs, mini_batch_size, learning_rate, embedding_dimension, margin,
              validation_x1s=None, validation_x2s=None, validation_ys=None):
        input1_ph = tf.placeholder(dtype=tf.float32, shape=(mini_batch_size, self.width, self.height))
        input2_ph = tf.placeholder(dtype=tf.float32, shape=(mini_batch_size, self.width, self.height))
        labels_ph = tf.placeholder(dtype=tf.float32, shape=(mini_batch_size,))
        output1 = self._get_output_ten(input1_ph, embedding_dimension)
        output2 = self._get_output_ten(input2_ph, embedding_dimension)

        loss = self._get_loss_op(output1, output2, labels_ph, margin)
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        num_batches = int(math.ceil(ys.shape[0] / mini_batch_size))

        with tf.Session() as sess:
            self._load_or_initialize(sess)

            for ep in range(num_epochs):
                x1s, x2s, ys = unison_shuffle([x1s, x2s, ys], ys.shape[0])

                for bt_num in range(num_batches):
                    bt_slice = slice(bt_num * mini_batch_size, (bt_num + 1) * mini_batch_size)
                    feed_dict = {input1_ph: x1s[bt_slice], input2_ph: x2s[bt_slice], labels_ph: ys[bt_slice]}

                    sess.run(train_op, feed_dict=feed_dict)
                    # print('Batch {}/{} has been processed'.format(bt_num, num_batches))

                print('Epoch {} training complete'.format(ep))

                self._monitor(
                    sess=sess,
                    x1s=x1s,
                    x2s=x2s,
                    ys=ys,
                    margin=margin,
                    mini_batch_size=mini_batch_size,
                    embedding_dimension=embedding_dimension,
                    prefix='Training')

                self._monitor(
                    sess=sess,
                    x1s=validation_x1s,
                    x2s=validation_x2s,
                    ys=validation_ys,
                    margin=margin,
                    mini_batch_size=mini_batch_size,
                    embedding_dimension=embedding_dimension,
                    prefix='Validation')
