"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0

This module implements embeddings so they can be reused by different models.
"""

import numpy as np
import tensorflow as tf


def identity(features, **kwargs):
  """
  Assumes embeddings are already present in the data.
  """
  return features


def one_hot(features, meta_info, **kwargs):

    num_words = len(meta_info['words'])

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
        keys=meta_info['words'],
        values=np.arange(num_words, dtype=np.int64)
        ), -1
    )

    features['embedding'] = tf.one_hot(table.lookup(features['word']), num_words)
    return features
