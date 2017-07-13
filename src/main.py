import numpy as np
import tensorflow as tf
from model import Model
from data import get_data


tf.logging.set_verbosity(tf.logging.INFO)


def main():
    np.random.seed()

    x1s_training, x2s_training, ys_training = get_data(subset='train')
    model = Model(64, 64, 1)
    model.train(
        x1s=x1s_training,
        x2s=x2s_training,
        ys=ys_training,
        num_epochs=200,
        embedding_dimension=64,
        mini_batch_size=50,
        learning_rate=0.0001,
        margin=0.5,
        monitor_training_loss=True,
        monitor_training_accuracy=True)


if __name__ == '__main__':
    main()
