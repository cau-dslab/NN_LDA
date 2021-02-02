import json
import os

import tensorflow as tf

def create_datasets(dataset, num_valid=0, max_epochs=1, batch_size=128, cache=True, **kwargs):
    """
    Takes `.tfrecord` files and creates `tf.data.Dataset` objects.

    If `dataset` is `hdfs://<path>/data/` or `hdfs://<path>/data.tfrecord`, then there should also be
    a JSON file called `hdfs://<path>/data.json` which describes the features and provides additional
    meta information. The format of the JSON file should be, for example::

        {
            "features": {
            "embedding": {"shape": [N], "dtype": "float32" },
            "author_id": { "shape": [], "dtype": "int64" },
            "author_topic": { "shape": [], "dtype": "string" },
            "item_id": { "shape": [], "dtype": "int64" },
            "item_topic": { "shape": [], "dtype": "string" },
            },
            "meta": {
            "embedding_dim": 300,
            "topics": ["tv", "politics", "sports"],
            }
        }

    If `dataset` is a path to a directory, all `.tfrecord` files in the directory will be loaded.

    Args
    ----
    dataset: A string pointing to a folder or .tfrecord file
    num_valid: If > 0, split data into training and validation sets
    max_epochs: Training data will be iterated this many times
    batch_size: How many data points to return at once
    cache: Keep dataset in memory to speed up epochs

    Returns
    -------
    One or two `tf.data.Dataset` objects and a dictionary containing meta information
    """
    # load meta information
    if dataset.lower().endswith('.tfrecord'):
        meta_info_file = dataset[:-9] + '.json'
    else:
        meta_info_file = dataset.rstrip('/') + '.json'
    # tfrecord에서 json파일로 변환시킨다.
    #tf.gfile.GFile( )은 tensorflow 구조에 특화된 파일 입출력 함수
    with tf.io.gfile.GFile(meta_info_file, 'r') as handle:
        meta_info = json.load(handle)
        meta_info, features = meta_info['meta'], meta_info['features']

    # extract description of features present in the dataset
    #이를 Feature 목록을 넣어서 파싱한 후에, 파싱된 데이타셋에서 각 피쳐를 하나하나 읽으면 된다.
    for name, kwargs in features.items():
        features[name] = tf.io.FixedLenFeature(**kwargs)

    # turn serialized example into tensors
    def _parse_function(serialized):
        return tf.io.parse_single_example(serialized=serialized, features=features)

    if dataset.endswith('.tfrecord'):
        files = [dataset]
    else:
        files = tf.io.gfile.glob(os.path.join(dataset, '*.tfrecord'))

    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    if num_valid > 0:
        # split dataset into training and validation sets
        dataset_valid = dataset.take(num_valid)
        dataset_train = dataset.skip(num_valid)

        if cache:
            dataset_valid = dataset_valid.cache()
            dataset_train = dataset_train.cache()

        # take into account hyperparameters
        dataset_train = dataset_train.shuffle(10000).repeat(max_epochs).batch(batch_size)
        dataset_valid = dataset_valid.batch(batch_size)

        return dataset_train, dataset_valid, meta_info

    else:
        if cache:
            dataset = dataset.cache()
        dataset = dataset.shuffle(1000).repeat(max_epochs).batch(batch_size)
        return dataset, meta_info

        