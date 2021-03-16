import os
import shutil
import argparse
import tensorflow as tf
import numpy as np
from config import data_path_prefix, dataset_output_path, vol_x, vol_y, vol_z, train_samples, validation_samples, \
    test_samples
from helpers import print_section, create_dataset_csv, get_record_path, augmented_dataset_output_path, generate_slices, generate_nifti
from preprocess import preprocess


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_dataset_path(dataset_type):
    if dataset_type == "train":
        return os.path.join(data_path_prefix, 'Training')
    if dataset_type == "test":
        return os.path.join(data_path_prefix, 'Test')
    if dataset_type == "validation":
        return os.path.join(data_path_prefix, 'Validation')


def get_data_split(dataset_type, volume_list):
    if dataset_type == "train":
        return volume_list[:train_samples]
    if dataset_type == "validation":
        return volume_list[train_samples:train_samples + validation_samples]
    if dataset_type == "test":
        return volume_list[train_samples + validation_samples: train_samples + validation_samples + test_samples]


def encode(volume_list, dataset_type, augment_data=False):
    print_section('Starting to encode to .tfrecords file')
    create_dataset_csv(dataset_type, volume_list, augment_data)

    print_section('Pre-processing volumes')
    preprocessed_volumes = preprocess(volume_list, (dataset_type == 'train' and augment_data))
    # Shuffle dataset
    if dataset_type == "train":
        np.random.shuffle(preprocessed_volumes)

    batch_start = 0
    batch_count = 20
    while batch_count <= len(preprocessed_volumes):
        with tf.io.TFRecordWriter(get_record_path(dataset_type, batch_start, augment_data)) as writer:
            for j, volume_tuple in enumerate(preprocessed_volumes[batch_start:batch_count]):
                # Load the bmode, pd and label volumes
                bmode = volume_tuple[0]
                pd = volume_tuple[1]
                label = volume_tuple[2]

                # Test - write to images
                # generate_slices(dataset_type, j, bmode, pd, label, augment_data)
                # generate_nifti(dataset_type, j, bmode, pd, label, augment_data)

                # Create a feature
                feature = {
                    'height': _int64_feature(vol_x),
                    'width': _int64_feature(vol_y),
                    'depth': _int64_feature(vol_z),
                    'bmode': _bytes_feature(bmode.tostring()),
                    'pd': _bytes_feature(pd.tostring()),
                    'label': _bytes_feature(label.tostring())
                }
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

        print("Num in batch", len(preprocessed_volumes[batch_start:batch_count]))
        print("Done writing record " + str(batch_count) + "/" + str(len(preprocessed_volumes)) + "_size=" + str(bmode.shape) + str(pd.shape) + str(label.shape))
        writer.close()
        batch_start = batch_count
        batch_count += 20

    print_section('Finished encoding to .tfrecords file')


def create_dataset(dataset_type, augment_data=False):
    print_section('Deleting old ' + dataset_type + 'ing data')
    dir = augmented_dataset_output_path if augment_data else dataset_output_path

    if os.path.exists(os.path.join(dir, dataset_type)):
        shutil.rmtree(os.path.join(dir, dataset_type))

    print_section('Creating ' + dataset_type + 'ing data')

    data_path = data_path_prefix
    images = os.listdir(data_path)
    images = get_data_split(dataset_type, images)

    all_bmodes = [os.path.join(data_path, img, img + '.nii.gz') for img in images]
    all_pds = [os.path.join(data_path, img, img + '_pd.nii.gz') for img in images]
    all_labels = [os.path.join(data_path, img, img + '_thresh.nii.gz') for img in images]

    volume_list = list(zip(all_bmodes, all_pds, all_labels))
    encode(volume_list, dataset_type, augment_data)

    print_section('Finished creating ' + dataset_type + 'ing dataset')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process configuration")
    parser.add_argument("--augment", required=False, type=bool, action="store")
    args = parser.parse_args()

    augment_arg = False

    if args.augment:
        augment_arg = True

    create_dataset("train", augment_arg)
    create_dataset("validation", augment_arg)
    create_dataset("test", augment_arg)
