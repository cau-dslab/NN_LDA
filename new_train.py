import inspect
import json
import os
import pprint
from argparse import ArgumentParser
from functools import partial

from logistic_lda.data import create_datasets
#from logistic_lda import embeddings, models
from logistic_lda import models, embeddings
import tempfile

import tensorflow as tf
from tensorflow import keras
tf.compat.v1.enable_eager_execution()


def main(args):
    # prepare training and validation data
    if args.num_valid > 0:
        dataset_train, dataset_valid, meta_info = create_datasets(**vars(args))
    else:
        dataset_train, meta_info = create_datasets(**vars(args))

    classifier = tf.estimator.Estimator(
        model_fn=getattr(models, args.model),
        params={
        'hidden_units': args.hidden_units,
        'learning_rate': args.initial_learning_rate,
        'decay_rate': args.learning_rate_decay,
        'decay_steps': args.learning_rate_decay_steps,
        'alpha': args.topic_bias_regularization,
        'model_regularization': args.model_regularization,
        'items_per_author': args.items_per_author,
        'author_topic_weight': args.author_topic_weight,
        'author_topic_iterations': args.author_topic_iterations,
        'meta_info': meta_info,
        'embedding': partial(getattr(embeddings, args.embedding), meta_info=meta_info, args=args, mode='train'),
        'use_author_topics': True,
        },
        model_dir=args.model_dir,
        warm_start_from=args.warm_start_from)
  # train
    def input_fn():
        return iter(dataset_train).get_next()

    # dataset = iter(create_datasets(**vars(args))[0]).get_next()
    # print(dataset)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        classifier.train(
            input_fn=lambda: tf.compat.v1.data.make_one_shot_iterator(create_datasets(**vars(args))[0]).get_next(),
            # input_fn = lambda : input_fn(),
            max_steps=args.max_steps)
    # 이 부분 수정해야함
  # evaluate
    if args.num_valid > 0:
        classifier.evaluate(lambda: tf.compat.v1.data.make_one_shot_iterator(create_datasets(**vars(args))[1]).get_next())

    return 0



if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default="./data/news20_train/",type=str,
        help='Path to a TFRecord dataset')
    parser.add_argument('--batch_size', type=int, default=32,
        help='Number of items per training batch')
    parser.add_argument('--author_topic_weight', type=float, default=200,
        help='Strength of factor connecting author labels with topic proportions')
    parser.add_argument('--author_topic_iterations', type=float, default=1,
        help='Number of variational inference iterations to infer missing author labels')
    parser.add_argument('--topic_bias_regularization', type=float, default=0.5,
        help='Parameter of Dirichlet prior on topic proportions')
    parser.add_argument('--items_per_author', type=int, default=200,
        help='For simplicity, the model assumes each author has the same number of items')
    parser.add_argument('--model_regularization', type=float, default=2.5,
        help='Strength of regularization encouraging model to use many topics')
    parser.add_argument('--initial_learning_rate', type=float, default=0.001,
        help='Initial learning rate of optimizer')
    parser.add_argument('--learning_rate_decay', type=float, default=0.8,
        help='Parameter of exponential learning rate decay')
    parser.add_argument('--learning_rate_decay_steps', type=int, default=2000,
        help='Parameter of exponential learning rate decay')
    parser.add_argument('--max_epochs', type=int, default=200,
        help='Maximum number of passes through the training set')
    parser.add_argument('--max_steps', type=int, default=300000,
        help='Maximum number of updates to the model parameters')
    parser.add_argument('--num_valid', type=int, default=0,
        help='Number of training points used for validation')
    parser.add_argument('--hidden_units', type=int, nargs='*', default=[512, 256, 128],
        help='List of hidden units defining the neural network architecture')
    parser.add_argument('--model', default='mlp_custom',
    # logistic_lda 이전에 mlp 버전 먼저 구현
        choices=list(zip(*inspect.getmembers(models, inspect.isfunction)))[0],
        help='Which model function to use')
    parser.add_argument('--embedding', default='identity',
        choices=list(zip(*inspect.getmembers(embeddings, inspect.isfunction)))[0],
        help='Which embedding function to apply to data points in the training set')
    parser.add_argument('--model_dir', type=str,
        help='Path where model checkpoints will be stored')
    parser.add_argument('--overwrite', action='store_true',
        help='If given, delete model path before starting to train')
    parser.add_argument('--warm_start_from', type=str, default=None,
        help='Initialize parameters from this model')
    parser.add_argument('--cache', action='store_true',
        help='Cache data to speed up subsequent training epochs')

    # 원 코드에서 hyper parameter 저장하는 코드는 일단 제외시킨다.

    args, _ = parser.parse_known_args()

    # choices=list(zip(*inspect.getmembers(models, inspect.isfunction)))[0]
    # print(choices)
    #pprint.pprint(vars(args))
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main(args)