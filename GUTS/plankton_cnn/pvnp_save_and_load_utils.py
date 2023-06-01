#!/usr/bin/env python3
import urllib.error

import pandas
import pandas as pd
import numpy as np
import os
import glob
import sys
import shutil
import datetime
import time
import pathlib

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model, Sequential
import tensorflow_hub as tf_hub
from keras.utils import dataset_utils

import plankton_cnn.pvnp_import as pvnp_import
from db_utils.set_paths import *
from plankton_cnn.pvnp_import import AUGMENT_DICT
from plankton_cnn.pvnp_models import model_dict

#from plankton_cnn.pvnp_use import apply_model_to_df
# Jasper van Leeuwen
# This has been commented out to prevent a circular import error when importing apply_model_to_df() from pvnp_use.py
# Might have to be uncommented at a later point if something breaks elsewhere


def load_train_df_per_model(training_data_dir, val_split=0.2, return_label_dict=False):
    """

    :param training_data_dir: str - path to training data set, without the suffix '_train' or '_val'
    :param val_split: float
    :param return_label_dict:
    :return: DataFrame - with columns 'image_name', 'image_path',
                         'label' as int, 'name' (str)
    """
    training_data_dir_train, _, val_split = import_train_or_val_dir(training_data_dir, val_split)

    # Load the validation set as a DataFrame
    val_df, label_dict = load_val_df_with_names(
        train_prefix=None, model_name=None, seed=1234, subset='training',
        training_data_dir=training_data_dir_train, val_split=val_split, return_label_dict=True)

    if return_label_dict:
        output = val_df, label_dict
    else:
        output = val_df

    return output


def apply_model_to_val_df(model_name, training_data_dir, train_or_tune_prefix, as_grayscale=False, val_split=0.2):
    """

    :param model_name: str
    :param training_data_dir:  str - path to training data set, without the suffix '_train' or '_val'
    :param train_or_tune_prefix:
    :param as_grayscale:
    :param val_split:
    :return: DataFrame - with columns 'image_name', 'image_path',
                         true_label' as int, 'true_name' (str) and the predictions
                         'label' as int, 'name' (str)
    """
    _, training_data_dir_val, val_split = import_train_or_val_dir(training_data_dir, val_split)

    # Load the validation set as a DataFrame
    val_df, _ = load_val_df_with_names(
        train_prefix=None, model_name=None, seed=1234, subset='validation',
        training_data_dir=training_data_dir_val, val_split=val_split, return_label_dict=True)

    # Rename columns, because 'label' and 'name' are overwritten in apply_model_to_df
    val_df.rename(columns={'label': 'true_label', 'name': 'true_name'}, inplace=True)

    # Jasper van Leeuwen
    # This is the instance of 'apply_model_to_df' from pvnp_use.py
    val_df = apply_model_to_df(model_name, train_or_tune_prefix, val_df,
                               label_dict=None, batch_size=32, as_grayscale=as_grayscale)

    return val_df


def get_numbers_per_group_in_train_data(training_data_dir):
    """

    :param training_data_dir:
    :return: pd.Series - with index the names and values the numbers per group

    Acces the sizes same as a dict with the names as keys. e.g. train_group_sizes['trochophore']
    """
    df_train = load_train_df_per_model(training_data_dir)
    train_group_sizes = df_train.groupby(by='name').size()
    return train_group_sizes


def load_val_df(train_prefix, model_name='efficientnet_v2_240', subset='validation',
                seed=None, training_data_dir=None, val_split=None):
    """

    :param train_prefix: str - ignored if training_dict for train_prefix and model_name does not exist
    :param model_name: str - ignored if training_dict for train_prefix and model_name does not exist
    :param subset: str - should be 'validation', 'training' or 'full'
    :param seed: int, optional - required (and only used) if a training_dict does not exist for train_prefix
                                 and model_name
    :param val_split: float - in (0, 1), required (and only used) if a training_dict does not exist for train_prefix
                              and model_name
    :param training_data_dir: str - required (and only used) if a training_dict does not exist for train_prefix
                                    and model_name
    :param overwrite: bool
    :return: DataFrame - with columns 'image_name', 'image_path', 'label' (as int)
    """
    if subset not in ['validation', 'training', 'full']:
        raise ValueError(f"Value for subset {subset} not recognized. Options are: "
                         "'validation', 'training', 'full'")

    # pkl_name = f"{subset}_set_{train_prefix}.pkl"
    # if os.path.isfile(f"{PATH_TO_PKLS}/{pkl_name}") and not overwrite:
    #     return read_pickle(pkl_name)
    # else:

    try:
        train_dict = load_training_args(f"{model_name}_{train_prefix}")
    except FileNotFoundError:
        if not (bool(seed) & bool(training_data_dir)):
            raise ValueError(f"An existing train_dict for {model_name}_{train_prefix} was not found, so function "
                             f"arguments seed and training_data_dir should both be specified")
    else:
        seed = int(train_dict['seed'])

        # Jasper van Leeuwen
        # The dataframe(s) that I have been using use a column called "directory" rather than "data_dir"
        # This is the original line of code:
        # training_data_dir = train_dict['data_dir']
        # This is the new line of code:
        training_data_dir = train_dict['directory']
        val_split = float(train_dict['validation_split'])

    image_paths, labels, class_names = dataset_utils.index_directory(
        f"{PATH_TO_TRAINING_DATA}/{training_data_dir}", labels='inferred',
        formats=(".png", ".jpg"), class_names=None, seed=seed)

    if subset in ['validation', 'training']:
        image_paths, labels = dataset_utils.get_training_or_validation_split(
            image_paths, labels, val_split, subset=subset)

    image_names = map(lambda x: x.split(sep='/')[-1], image_paths)
    val_df = pd.DataFrame({'image_name': image_names,
                           'image_path': image_paths,
                           'label': labels})

    # to_pickle(val_df, pkl_name, verbose=0)
    # print(f"Saved {subset} set of {train_prefix} to {pkl_name}")

    return val_df


def load_val_df_with_names(train_prefix, model_name=None, subset='validation',
                seed=None, training_data_dir=None, val_split=None, return_label_dict=False):
    """

    :param train_prefix:
    :param model_name:
    :param return_label_dict:
    :return: DataFrame - with columns 'image_name', 'image_path', 'label' (int), 'name' (str)
    """
    val_df = load_val_df(train_prefix=train_prefix, model_name=model_name, seed=seed,
                         training_data_dir=training_data_dir, val_split=val_split, subset=subset)
    label_dict = load_label_dict(train_prefix=train_prefix, training_data_dir=training_data_dir)
    val_df['name'] = val_df['label'].apply(lambda x: label_dict[x])

    if return_label_dict:
        output = val_df, label_dict
    else:
        output = val_df

    return output


def load_label_dict(train_prefix=None, training_data_dir=None):
    """
    Load label dictionary with keys the integer labels (as integers) and as values the
    string names of the labels of the training data directory. Either loads the dictionary
    from a txt-file using load_saved_label_dict or reloads the dictionary by parsing the
    files in training_data_dir in the same way as is done in
    pvnp.import.import_plankton_images. So one of train_prefix or training_data_dir
    should be specified.

    :param train_prefix: str
    :param training_data_dir: str
    :return: dict - dictionary with keys the integer labels (as integers) and as values the
    string names of the labels
    """
    try:
        label_dict = load_saved_label_dict(train_prefix)
        label_dict = dict(zip([int(x) for x in label_dict.keys()], label_dict.values()))
    except FileNotFoundError:
        if training_data_dir:
            label_dict = pvnp_import.get_labeL_dict_for_training_directory(training_data_dir)
        else:
            raise ValueError(f"An existing label_dict for {train_prefix} was not found, so function "
                             f"arguments training_data_dir should be specified")
    return label_dict


def load_val_ds_per_model(model_name, train_prefix):
    model_run = f"{model_name}_{train_prefix}"
    train_dict = load_training_args(model_run)
    batch_size = int(train_dict['batch_size'])
    seed = int(train_dict['seed'])
    val_split = float(train_dict['validation_split'])

    # Somewhere on the way, the key of the training data directory was changed,
    # so to keep this function compatible an if-clause is used
    if 'data_dir' in train_dict.keys():
        dir = train_dict['data_dir']
    elif 'directory' in train_dict.keys():
        dir = train_dict['directory']

    img_size = model_dict[train_dict['model_name']]['img_size']

    val_ds, data_dict = pvnp_import.import_plankton_images(
        dir, img_size, batch_size, seed=seed,
        validation_split=val_split, subset="validation")

    return val_ds, data_dict


def save_history(history, train_prefix, round_prefix):
    """
    Saves db with metrics as columns to the path
    train_prefix/prefix_hist.pkl

    history: history-object
    train_prefix: str
    prefix: str
    """
    db = pd.DataFrame(history.history)
    to_pickle(db, (dest := f"{train_prefix}/{round_prefix}_hist.pkl"),
              dir=None, verbose=0)
    print(f"history of {round_prefix} saved to {dest}")


def load_history(train_name):
    """
    For training round, finds history files that were saved via
    save_history()

    Returns: dictionary with, for each training prefix, a pd.DataFrame
    with the saved metrics as columns and the metric values as entries.
    """
    history_paths = glob.glob(f"{PATH_TO_CHECKPOINTS}/{train_name}/*_hist.pkl")

    if not len(history_paths):
        raise Exception(f"No hist.pkl-file was found for {train_name} at {PATH_TO_CHECKPOINTS}")

    hist_dict = {}
    for history_path in history_paths:
        db = read_pickle(history_path, dir=None)
        prefix = history_path.split(sep='/')[-1].split(sep='_')[0]
        hist_dict[prefix] = db

    return hist_dict


def import_train_or_val_dir(data_dir, val_split=None):
    """
    For data_dir, it is assessed whether separate folders containing the training and validation
    data are present or if a split should be made within the single folder. Specifically,
    for folder 'data_dir' it is checked if folders named 'data_dir_train' or 'data_dir_training'
    and 'data_dir_val' or 'data_dir_validation' are present. If both are present and not
    multiple versions are found, then the name of the training folder, the validation folder and
    the adjusted val_split (=None) is returned. If multiple versions are found, an error is
    raised. If no folders were found, data_dir is returned as both the training folder and the
    validation folder and the val_split is unadjusted.

    The intended use is that for a base name of the data directory (e.g. 'delta_spring2022')
    you can apply this function to the data_dir and, if desired, the standard validation split
    and you get the names of the training and validation data and if necessary the adjusted
    validation split (because if one would apply a validation split on the separate folders,
    another split will be made).

    :param data_dir: str - path to training data set, without the suffix '_train' or '_val'
    :param val_split: float
    fraction of data to use as validation split. Only entered here in order to get the val_split
    adjusted to None (meaning no split will be made) for later processing in case the data
    is detected to be in separate folders.
    :return: str, str, float - training_data_dir, validation_data_dir, validation split
    """
    # Parse directories in parent directory of data_dir
    full_dir = f"{PATH_TO_TRAINING_DATA}/{data_dir}"
    parent_dir = os.path.dirname(full_dir)
    subdirs = os.listdir(parent_dir)  # subdirs contains '.DS_Store'. No problem for now

    def find_train(dir, orig_dir=data_dir):
        dir_split = dir.split(sep='_')
        if (dir_split[-1] in ['train', 'training']) and ('_'.join(dir_split[:-1]) == orig_dir):
            return True
        else:
            return False

    def find_val(dir, orig_dir=data_dir):
        dir_split = dir.split(sep='_')
        if (dir_split[-1] in ['val', 'validation']) and ('_'.join(dir_split[:-1]) == orig_dir):
            return True
        else:
            return False

    train_list = list(filter(find_train, subdirs))
    val_list = list(filter(find_val, subdirs))
    assert len(train_list) == len(val_list)
    if (length := len(train_list)) == 0:
        return data_dir, data_dir, val_split
    elif length == 1:
        return train_list[0], val_list[0], None
    else:
        raise AssertionError("Multiple possible files were found that can contain the"
                             f"training and/or validation data."
                             f"Found: {train_list}, {val_list}")


def save_dict(dictionary, path):
    """
    Save a dictionary to a txt.file with for each key in a separate line, followed by
    a double dot and then the value, e.g:
    key1: value1
    key2: value2
    ...

    Use load_args_dict to load back the original dictionary

    :param dictionary:
    :param path:
    :return:
    """
    with open(path, "w") as file:
        for key, val in dictionary.items():
            file.write(f"{key}: {str(val)} \n")


def save_training_args(args_dict, save_name):
    dest = f"{PATH_TO_CHECKPOINTS}/{save_name}/training_args.txt"
    # with open(dest, "w") as file:
    #     for key, val in args_dict.items():
    #         file.write(f"{key}: {str(val)} \n")
    save_dict(args_dict, path=dest)
    print(f"Trainings arguments saved to {dest}")


def load_training_args(cp_name):
    """

    """
    filename = f"{PATH_TO_CHECKPOINTS}/{cp_name}/training_args.txt"
    return load_args_dict(filename)


def save_tuner_args(args_dict, tuner_data_dir):
    dest = f"{PATH_TO_TUNER_DATA}/{tuner_data_dir}/best_hyperparameters.txt"
    save_dict(args_dict, path=dest)
    print(f"Best hyperparameters {args_dict} \n saved to {dest}")


def load_tuner_args(tuner_data_dir):
    args_dict = load_args_dict(f"{PATH_TO_TUNER_DATA}/{tuner_data_dir}/best_hyperparameters.txt")
    for key, item in args_dict.items():
        args_dict[key] = float(item)
    return args_dict


def load_args_dict(filename):
    with open(filename, 'r') as file:
        args_dict = {}
        for line in file:
            sep = line.find(': ')
            arg, val = line[:sep], line[sep+1:]
            args_dict[arg.strip()] = val.strip()

    return args_dict


def save_label_dict(train_prefix_base, model_name):
    """
    model_name is only needed here in order to use function load_val_ds_per_model()

    :param train_prefix_base:
    :param model_name:
    :return:
    """
    _, label_dict = load_val_ds_per_model(model_name, train_prefix_base)
    for key, val in label_dict.items():
        label_dict[key] = val['name']

    dest = f"{PATH_TO_CHECKPOINTS}/{train_prefix_base}_labels.txt"
    with open(dest, "w") as file:
        for key, val in label_dict.items():
            file.write(f"{key}: {str(val)} \n")
    print(f"Label dict {train_prefix_base} saved to {dest}")


def save_label_dict_new(train_prefix, training_dir):
    """

    :param train_prefix:
    :param training_dir:
    :return:
    """
    label_dict = pvnp_import.get_labeL_dict_for_training_directory(training_dir=training_dir)

    dest = f"{PATH_TO_CHECKPOINTS}/{train_prefix}_labels.txt"
    with open(dest, "w") as file:
        for key, val in label_dict.items():
            file.write(f"{key}: {str(val)} \n")
    print(f"Label dict {train_prefix} saved to {dest}")


def load_saved_label_dict(train_prefix_base):
    filename = f"{PATH_TO_CHECKPOINTS}/{train_prefix_base}_labels.txt"
    return load_args_dict(filename)


def load_and_stitch_history(model_name, *train_prefixes, train_round='best'):
    stitch_points = []
    for i, prefix in enumerate(train_prefixes):
        cp_name = f"{model_name}_{prefix}"
        if not i:
            db_history_stitched = load_history(cp_name)[train_round]
            continue

        db_history = load_history(cp_name)[train_round]
        stitch_points.append(len(db_history_stitched))
        db_history_stitched = pd.concat([db_history_stitched, db_history], ignore_index=True)

    return db_history_stitched, stitch_points


def get_best_acc_from_saved_model(model_name, train_prefix, train_round='best'):
    """
    Accuracy in [0, 1]

    :param model_name: str
    :param train_prefix: str
    :param train_round: str
    :return: float
    """
    return max(load_and_stitch_history(model_name, train_prefix, train_round=train_round)[0]['val_accuracy'])


if __name__ == "__main__":
    train_dir, val_dir, val_split = import_train_or_val_dir('alpha_spring2022', val_split=0.2)
    train_dir, val_dir, val_split = import_train_or_val_dir('delta_grayscale_spring2022', val_split=0.2)
    print(train_dir)
    print(val_dir)
    print(val_split)
    # import_train_or_val_dir('VLIZ_alpha2022_full')

    # load_val_ds_per_model('Xception', 'alpha_spring2022')
    # save_label_dict('alpha_spring2022', 'Xception')
    # save_label_dict_new('delta_spring2022_grayscale2', 'delta_grayscale_spring2022_train')
    pass


def get_total_length_with_augment(dataset, augmentation, val_split=None):
    """

    :param dataset: str - path to training data
    :param augmentation:
    :param val_split:
    :return:
    """
    total_length = len(load_train_df_per_model(dataset, val_split=val_split))
    if augmentation:
        total_length = total_length * AUGMENT_DICT[augmentation]

    return total_length
