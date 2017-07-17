import sys
import argparse
import numpy as np
import tensorflow as tf
from model import Model
from data import get_data
from config import TRAINING_DATA_SLICE, VALIDATION_DATA_SLICE


tf.logging.set_verbosity(tf.logging.INFO)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_id', type=str, help='Postfix for model filename')
    parser.add_argument('--saved_model', type=str, help='Id of the model to start training with')

    return parser.parse_args(argv)


def main(args):
    np.random.seed()

    x1s_training, x2s_training, ys_training = get_data(subset='train')
    model = Model(64, 64, 1, model_id=args.model_id)
    model.train(
        x1s=x1s_training[TRAINING_DATA_SLICE],
        x2s=x2s_training[TRAINING_DATA_SLICE],
        ys=ys_training[TRAINING_DATA_SLICE],
        validation_x1s=x1s_training[VALIDATION_DATA_SLICE],
        validation_x2s=x2s_training[VALIDATION_DATA_SLICE],
        validation_ys=ys_training[VALIDATION_DATA_SLICE],
        num_epochs=2000,
        embedding_dimension=128,
        mini_batch_size=50,
        learning_rate=0.001,
        margin=0.5)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
