import os
import tensorflow as tf
from helpers import get_record_path


AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_tfrecords(dataset_type, augmented):
    path = get_record_path(dataset_type, None, augmented)
    file_list = os.listdir(path)
    return [os.path.join(path, name) for name in file_list]


def decode(serialized_example, modal_type):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'bmode': tf.io.FixedLenFeature([], tf.string),
            'pd': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string)
        }
    )

    volume_shape = tf.stack([features['height'], features['width'], features['depth']])

    if modal_type == "multi_modal":
        bmode = tf.io.decode_raw(features['bmode'], tf.float32)
        bmode = tf.reshape(bmode, volume_shape)
        bmode = tf.expand_dims(bmode, axis=-1)
        bmode = tf.cast(bmode, tf.float32)

        pd = tf.io.decode_raw(features['pd'], tf.float32)
        pd = tf.reshape(pd, volume_shape)
        pd = tf.expand_dims(pd, axis=-1)
        pd = tf.cast(pd, tf.float32)

        input_vol = {
            "input_1": bmode,
            "input_2": pd
        }
    else:
        input_vol = tf.io.decode_raw(features[modal_type], tf.float32)
        input_vol = tf.reshape(input_vol, volume_shape)
        input_vol = tf.expand_dims(input_vol, axis=-1)
        input_vol = tf.cast(input_vol, tf.float32)

    label = tf.io.decode_raw(features['label'], tf.uint8)
    label = tf.reshape(label, volume_shape)
    label = tf.expand_dims(label, axis=-1)
    label = tf.cast(label, tf.uint8)

    return input_vol, label


def load_dataset(dataset_type, batch_size, num_epochs, modal_type, augmented=False):
    files = get_tfrecords(dataset_type, augmented)
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(lambda x: decode(x, modal_type))
    dataset = dataset.batch(batch_size)
    if dataset_type == "train":
        dataset = dataset.repeat(num_epochs)
    return dataset.prefetch(buffer_size=AUTOTUNE)
