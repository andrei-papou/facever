import sys
import argparse
import numpy as np
import tensorflow as tf
from model import Model
from data import get_data, DataProvider


tf.logging.set_verbosity(tf.logging.INFO)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_id', type=str, help='Postfix for model filename')
    parser.add_argument('--saved_model', type=str, help='Id of the model to start training with')

    return parser.parse_args(argv)


def main(args):
    np.random.seed()

    data_provider = DataProvider(batch_size=50)
    data_provider.fetch_data()

    model = Model(64, 64, 1, model_id=args.model_id)
    model.train(
        data_provider=data_provider,
        num_epochs=2000,
        embedding_dimension=128,
        mini_batch_size=50,
        learning_rate=0.00035,
        margin=0.5)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
