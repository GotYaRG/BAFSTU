#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys
import shutil
import datetime

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.data.ops import dataset_ops
from keras.utils import dataset_utils

from db_utils.set_paths import *

if __name__ == '__main__':
    print(tf.__version__)


def get_labeL_dict_for_training_directory(training_dir):
    """
    Make a dictionary of the integer labels and the class names (as specified in the names
    of the directories inside training_dir).

    The correct order is guaranteed: in dataset_utils.index_directory, with class_names=None,
    class names are returned in alphanumerical order. With labels='inferred',
    integer labels are also assigned in alphanumerical order of the class names. Since np.unique()
    also sorts in alphanumerical order, it is guaranteed that the order of class_names
    and unique_labels is the same.

    :param training_dir: str
    :return: dict - with keys the integer labels, values the names as strings
    """
    image_paths, labels, class_names = dataset_utils.index_directory(
      f"{PATH_TO_TRAINING_DATA}/{training_dir}", labels="inferred", formats=(".png", ".jpg"), class_names=None)

    return dict(zip(np.unique(labels), class_names))


def import_plankton_images(directory, image_size, batch_size, seed,
                           validation_split, subset,
                           augment='simple', as_grayscale=False):
    """

    :param directory: str
    :param image_size: int
    :param batch_size: int
    :param seed: int
    :param validation_split: float
    :param subset: str - should be 'training' or 'validation'
    :param augment: str - either 'simple' or 'all_rotations' or None
    :param as_grayscale: boolean -
    :return:
    """
    num_channels = 3

    # if type(image_size) is not tuple or not len(image_size) == 2:
    #     raise Exception(f"'image_size' should be of shape (image_height, image_width),\
    #     while currently it's: {image_size}")

    image_paths, labels, class_names = dataset_utils.index_directory(
        f"{PATH_TO_TRAINING_DATA}/{directory}", labels="inferred",
        formats=(".png", ".jpg"),
        class_names=None, seed=seed)

    image_paths, labels = dataset_utils.get_training_or_validation_split(
          image_paths, labels, validation_split, subset)

    # Make a dictionary of the numbers of images per group
    class_dict = {}
    for label, name in zip(np.unique(labels), class_names):
        label_dict = {'length': np.sum(labels == label),
                      'name': name}
        class_dict[label] = label_dict

    if not validation_split:
        print(f"Using {sum([dic['length'] for dic in class_dict.values()])} "
              f"images for {subset}\n")

    # Convert the image paths to images using load_image()
    images_ds = convert_paths_to_ds_of_images_and_labels(image_paths=image_paths, image_size=image_size,
                                                         num_channels=num_channels, labels=labels,
                                                         as_grayscale=as_grayscale)

    if subset == 'training':
        if augment == 'simple':
            print(f"\nAugmentation '{augment}' was applied to training data")
            images_ds = augm_image_ds(images_ds, all_rotations=False)
        elif augment == 'all_rotations':
            print(f"\nAugmentation '{augment}' was applied to training data")
            images_ds = augm_image_ds(images_ds, all_rotations=True)
        else:
            print("\nArgument for data augmentation was not recognised. No augmentation applied")

        # We shuffle only train_ds, so val_ds can be evaluated manually.
        # With 10.000, the buffer size can be lower than the size of the dataset but this is
        # necessary since for large datasets the complete set might not fit into the memory
        # for shuffling. This compromises the randomness though
        # (see https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle)
        images_ds = images_ds.shuffle(buffer_size=10000, seed=seed)

    images_ds = images_ds.batch(batch_size)
    images_ds.class_names = class_names

    if as_grayscale:
        print("Imported images as grayscale")
    
    return images_ds, class_dict
          
          
def convert_paths_to_ds_of_images_and_labels(image_paths, image_size, num_channels, labels,
                                as_grayscale=False):
    """

    :param image_paths:
    :param image_size:
    :param num_channels:
    :param labels:
    :param as_grayscale:
    :return:
    """
    # Convert image_paths and labels to tf.datasets
    path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
    label_ds = dataset_ops.Dataset.from_tensor_slices(labels)

    # Convert path_ds to a dataset containing the actual images using load_image and combine with the labels
    # to a single dataset
    img_ds = path_ds.map(lambda x: load_image(x, image_size, num_channels, as_grayscale=as_grayscale))
    return dataset_ops.Dataset.zip((img_ds, label_ds))


def load_image(path, image_size, num_channels, as_grayscale=False):
    """Load an image from a path, convert pixel values to float in range [0, 1)
    and resize it with padding."""

    img = io_ops.read_file(path)
    img = image_ops.decode_image(img, channels=num_channels, expand_animations=False)

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.adjust_contrast(img, 1.3)

    if as_grayscale:
        img = to_grayscale(img)

    # Rescale before padding, because padding could add zeros. After resize_with_pad, pixel values might
    # not fully extend to 0 and 1 due to bilinear resizing.
    img = rescale_image_values(img)
    img = tf.image.resize_with_pad(img, image_size, image_size)
    return tf.reshape(img, [image_size, image_size, num_channels])


def rescale_image_values(img):
    """
    Rescale the pixel values such that the mininmum, maximum pixel
    value equals 0, 1 respectively.

    :param img: tf.image
    :return: tf.image of same shape as original image
    """
    img_min, img_max = tf.reduce_min(img), tf.reduce_max(img)
    numerator = tf.subtract(img, img_min)

    # The denominator cannot be smaller than (an arbitrary choice) 0.001 in order
    # to prevent zero-divison. In practice, this means that for images where the difference
    # between the original minimum and maximum value is small, the maximum pixel value after
    # rescaling remains < 1
    denom = tf.reduce_max(
        tf.stack(
            [tf.subtract(img_max, img_min), tf.constant(0.001)]
        ))
    return tf.divide(numerator, denom)


def augm_image(image, rotate=True):
    if rotate:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    return rescale_image_values(image)


def augm_image_ds(ds, all_rotations=False):
    ''''
    input: tf.dataset of (image, label)
    applies augm_image to all images in dataset and concatenates augmented images
    to original dataset
    return: tf.dataset of (image, label) of 2* the input length
    '''
    if all_rotations:
        # Add all 90degree-rotations of each image with randomly varying brightness/contrast
        ds_new = ds
        for num in [1, 2, 3]:
            ds_ext = ds.map(lambda x, y: (augm_image(x, rotate=False), y))
            ds_rot = ds_ext.map(lambda x, y: (tf.image.rot90(x, k=num), y))
            ds_new = ds_new.concatenate(ds_rot)
    else:
        # Add a duplicate for each image with randomly varying brightness/contrast/rotation
        ds_ext = ds.map(lambda x, y: (augm_image(x, rotate=True), y))
        ds_new = ds.concatenate(ds_ext)

    return ds_new


# Not super elegant, but in order to access the multiplier of the applied augmentations, we put these values in
# a dictionary
AUGMENT_DICT = {'simple': 2, 'all_rotations': 4}
    
    
def to_grayscale(image):
    '''
    input: tf.Tensor - image as tensor with 3 channels
    returns: tf.Tensor - image with original shape with 3 identical color channels,
    which are result of conversion of input image to grayscale
    '''
    image = tf.image.rgb_to_grayscale(image)

    # In order to properly stack the image back to its original shape,
    # we need to extract the first 2 dimensions
    image = image[:, :, 0]
    return tf.stack([image, image, image], axis=2)


def import_images_from_df(df, image_size, batch_size, as_grayscale=False):
    """
    df has to contain a column named 'image_path'
    """
    num_channels = 3

    path_ds = dataset_ops.Dataset.from_tensor_slices(df['image_path'])
    img_ds = path_ds.map(lambda x: load_image(x, image_size, num_channels, as_grayscale=as_grayscale))
    img_ds = img_ds.batch(batch_size)

    return img_ds


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    image_paths, labels, class_names = dataset_utils.index_directory(
      f"{PATH_TO_TRAINING_DATA}/alpha_spring2022", labels="inferred", formats=(".png"), class_names=None, seed=1234)
    image_paths, labels = dataset_utils.get_training_or_validation_split(
          image_paths, labels, 0.2, 'validation')

    path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
    for path in path_ds.take(3):
        print()

        fig, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.imshow(load_image(path, 128, 3, as_grayscale=True))
        ax1.set_axis_off()
        ax2.imshow(load_image(path, 128, 3, as_grayscale=False))
        ax2.set_axis_off()
        fig.suptitle('method 1')
        plt.show()

    for path in path_ds.take(1):
        print(load_image(path, 128, 3, as_grayscale=True)[:, :, 0])

    sys.exit()

    images_ds = convert_paths_to_ds_of_images_and_labels(image_paths=image_paths, image_size=128, num_channels=3,
                                                         labels=labels, as_grayscale=True)
    images_ds = augm_image_ds(images_ds, all_rotations=False)
    images_ds = images_ds.batch(64)

    for img, _ in images_ds.unbatch().take(3):
        fig, ax1 = plt.subplots()
        ax1.imshow(img)
        print(tf.reduce_min(img), tf.reduce_max(img))
        ax1.set_axis_off()
        fig.suptitle('method 2')
        plt.show()

    sys.exit()

    from plankton_cnn.pvnp_visualize import show_subsample

    show_subsample(ds)

    sys.exit()

    # # to do - select sharp images only and select images that are not in training set
    # path_to_archive = '/Users/pieter/Documents/CPICS_database/'
    # conn = sqlite3.connect("/Users/pieter/Documents/CPICS_database/Archive/image_database.db")
    # df_nl = pd.read_sql_query(
    #     "select image_path from master_table", con=conn).iloc[:1000]
    # df_nl['image_path'] = df_nl['image_path'].map(lambda x: path_to_archive + x)

    image_size = (200, 200)
    batch_size = 32

    ds_nl = import_images_from_df(df_nl, image_size, batch_size)
    for img in ds_nl.take(5):
        print(img)
