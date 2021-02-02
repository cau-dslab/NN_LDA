"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0

Implementations of logistic LDA and baseline models for use with Tensorflow's Estimator.
"""

from logistic_lda.utils import create_table, softmax_cross_entropy

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from logistic_lda.embeddings import one_hot

def mlp_custom(features, labels, mode, params):
    print(features)

    n_topics = len(params['meta_info']['topics'])

    # preprocess features (e.g., compute embeddings from words)
    features = params['embedding'](features)
    print(features)
    features = one_hot(features, params['meta_info'])


    # convert string labels to integers
    topic_table = create_table(params['meta_info']['topics'])
    author_topics = topic_table.lookup(features['author_topic'])

    net = features['embedding']
    for units in params['hidden_units']:
        net = layers.Dense(
        units, 
        activation="relu",
        kernel_regularizer = tf.keras.regularizers.l2(params['model_regularization']) 
        )(net)
    
    logits = layers.Dense(n_topics, activation=None,
    kernel_regularizer = tf.keras.regularizers.l2(params['model_regularization']))(net)

    
    if mode == tf.estimator.ModeKeys.PREDICT:
        probs = tf.math.reduce_max(tf.nn.softmax(logits), 1)
        predictions = tf.math.argmax(logits, 1)
        predictions = {
        'item_id': features['item_id'],
        'item_prediction': predictions,
        'item_probability': probs,
        'item_topic': topic_table.lookup(features['item_topic']),
        'author_id': features['author_id'],
        'author_prediction': predictions,
        'author_probability': probs,
        'author_topic': author_topics,
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

     # model is trained to predict which topic an author belongs to
    loss = tf.math.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(author_topics, depth=n_topics),
            logits=logits
        )
    )

    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy, acc_op = tf.metrics.accuracy(
            labels=author_topics,
            predictions=tf.math.argmax(logits, 1),
            name='acc_op')

        metric_ops = {'accuracy': (accuracy, acc_op)}

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metric_ops)

    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=tf.compat.v1.train.exponential_decay(
        learning_rate=params['learning_rate'],
        decay_rate=params['decay_rate'],
        decay_steps=params['decay_steps'],
        global_step=tf.compat.v1.train.get_global_step()))
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())

    print("mlp custom이 돌고 있어요")
    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op)

#https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator
#https://www.tensorflow.org/tutorials/structured_data/feature_columns
def mlp(features, labels, mode, params):
  """
  Model function implementing a simple MLP which can be used for topic modeling.

  Args:
    features['embedding']: A tensor of shape [B, D]
    features['author_topic']: A tensor of shape [B] containing author labels as strings
    features['item_topic']: An tensor of shape [B] containing item labels (used in PREDICT only)
    labels: This will be ignored as labels are provided via `features`
    mode: Estimator's `ModeKeys`
    params['meta_info']['topics']: A list of strings of all possible topics
    params['hidden_units']: A list of integers describing the number of hidden units
    params['learning_rate']: Learning rate used with Adam
    params['decay_rate']: Exponential learning rate decay parameter
    params['decay_steps']: Exponential learning rate decay parameter
    params['embedding']: A function which preprocesses features

  Returns:
    A `tf.estimator.EstimatorSpec`
  """

  n_topics = len(params['meta_info']['topics'])

  # preprocess features (e.g., compute embeddings from words)
  features = params['embedding'](features)

  

def logistic_lda(features, labels, mode, params):
    """
    An implementation of logistic LDA.

    Args:
        features['embedding']: A tensor of shape [B, D]
        features['author_topic']: A tensor of shape [B] containing author labels as strings
        features['author_id']: A tensor of shape [B] containing integer IDs
        features['item_topic']: A tensor of shape [B] containing item labels (use '' if unknown)
        features['item_id']: A tensor of shape [B] containing integer IDs
        labels: This will be ignored as labels are provided via `features`
        mode: Estimator's `ModeKeys`
        params['meta_info']['topics']: A list of strings of all possible topics
        params['meta_info']['author_ids']: A list of all possible author IDs (these IDs group items)
        params['hidden_units']: A list of integers describing the number of hidden units
        params['learning_rate']: Learning rate used with Adam
        params['decay_rate']: Exponential learning rate decay parameter
        params['decay_steps']: Exponential learning rate decay parameter
        params['author_topic_weight']: Controls how much author labels influence the model
        params['author_topic_iterations']: Number of iterations to infer missing author labels
        params['model_regularization']: Regularize model to make use of as many topics as possible
        params['items_per_author']: For simplicity, model assumes this many items per author
        params['alpha']: Smoothes topic distributions of authors
        params['embedding']: A function which preprocesses features
    """
    print("models")
  