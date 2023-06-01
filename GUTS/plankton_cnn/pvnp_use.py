#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy
import sys

import sklearn.metrics
from tensorflow import keras
import tensorflow as tf

# from plankton_cnn.pvnp_visualize import plot_confusion_matrix_for_val_df
# Jasper van Leeuwen
# This import has been commented out to prevent a circular import error when importing functions from
# pvnp_visualize.py together with apply_model_to_df() from this function. These functions were needed for a quick
# script to produce confusion matrices and plotting the convergence of the different runs.

import plankton_cnn.pvnp_models
import plankton_cnn.pvnp_save_and_load_utils
from plankton_cnn import pvnp_import
from db_utils.set_paths import *


def apply_model_to_df(model_name, train_prefix, df, batch_size=None,
                      label_dict=None, training_data_dir=None, as_grayscale=False,
                      apply_softmax=False, get_2nd_choice=False):
    """
    Apply model_name, trained in training round train_prefix to the images in
    DataFrame df.

    :param model_name: str
    :param train_prefix: str
    :param df: DataFrame - has to contain a column named 'image_path'
    :param as_grayscale: bool
    :param batch_size:
    :param apply_softmax: bool
    :param label_dict: dict - dict with keys the integer labels (as integers) and values the string names.
                              For all saved models both from regular training and from tuning, a label_dict
                              should exist at the disk. If present, this one will be used (recommended) and
                              argument for label_dict will be ignored.
    :param training_data_dir: str - not needed anymore, so will be ignored
    :param get_2nd_choice: bool - if True, also the softmax-value and predicted label of the second best
                                  predictions is added to the DataFrame
    :return: DataFrame df with new columns:
        - 'label' with model predictions
        - 'softmax' with softmax output in [0, 1] of predicted label
    """
    if not len(df):
        raise ValueError("DataFrame is empty")

    # Load input parameters for model and train_prefix
    model_run = f"{model_name}_{train_prefix}"
    img_size = plankton_cnn.pvnp_models.model_dict[model_name]['img_size']
    # img_size = (img_size, img_size)

    # If a training arguments file exist, we load that. Otherwise, the necessary parameters need
    # to be specified in the function call
    try:
        train_dict = plankton_cnn.pvnp_save_and_load_utils.load_training_args(model_run)
    except FileNotFoundError:
        if not batch_size:
            raise ValueError(f"An existing train_dict for {model_name}_{train_prefix} was not found, so function "
                             f"argument batch_size should be specified")
    else:
        batch_size = int(train_dict['batch_size'])

    # Load the label_dict
    if (base := train_prefix.split(sep='_')[0]) == 'alpha':
        dict_prefix = base
    else:
        dict_prefix = train_prefix

    try:
        label_dict = plankton_cnn.pvnp_save_and_load_utils.load_label_dict(dict_prefix)
    except FileNotFoundError:
        if not label_dict:
            raise ValueError(f"An existing label_dict for {train_prefix} was not found, so function "
                             f"argument label_dict should be specified")

    # Import data and predict
    ds = pvnp_import.import_images_from_df(df, img_size, batch_size, as_grayscale=as_grayscale)
    print(f"\nRunning model {model_run}")

    # Jasper van Leeuwen
    # In the line of code below, compile had to be set to False. Otherwise, the loaded model would throw an error.
    model = keras.models.load_model(f"{PATH_TO_MODELS}/{model_run}", compile=True)
    predictions = model.predict(ds)

    if apply_softmax:
        # In the more recent models that are imported, a final softmax-layer is added to the
        # model. For older models where this is not the case, a separate softmax needs
        # to be applied afterwards.
        predictions = scipy.special.softmax(predictions, axis=1)

    # Find maximum and 2nd maximum prediction and add both label and softmax
    # to val_df
    predictions_args = np.argsort(predictions, axis=1)
    predictions_val = np.sort(predictions, axis=1)

    df['label'] = predictions_args[:, -1]
    df['softmax'] = predictions_val[:, -1]

    # Add string-labels instead of numbers
    # df['label'] = df['label'].astype(str)
    df = df.replace({'label': label_dict})

    if get_2nd_choice:
        df['label_2nd'] = predictions_args[:, -2]
        df['softmax_2nd'] = predictions_val[:, -2]

        # df['label_2nd'] = df['label_2nd'].astype(str)
        df = df.replace({'label_2nd': label_dict})

    return df


def evaluate_model(model, test_dataset, checkpoint_path):
    model.load_weights(checkpoint_path)
    result = model.evaluate(test_dataset)
    names = model.metrics_names
    metrics = dict(zip(names, result))
    return metrics


def apply_threshold_and_2nd_grouping(df, threshold=0.7, groups_2nd_dict=None):
    """
    Apply threshold filtering to a separate label 'filter_0{threshold}' and apply
    grouping of related groups by their 'second choice', based on the grouping in
    groupings_2nd_dict. Note that both operations depend on each other and cannot
    be performed independently.


    :param df: DataFrame - requires column named 'label', 'softmax', 'label_2nd', 'softmax_2nd'
    :param threshold:
    :param groups_2nd_dict: dict - {'group_unid': [group_members]}
    :return: DataFrame with adapted labels
    """
    filter_name = f"filter_0{int(threshold * 100)}"

    if groups_2nd_dict:
        for group_unid, group_list in groups_2nd_dict.items():
            df.loc[(df['softmax'] < threshold) &
                        (df['label'].isin(group_list)) &
                        (df['label_2nd'].isin(group_list)), 'label'] = group_unid

        df.loc[(df['softmax'] < threshold) &
               (~df['label'].isin(groups_2nd_dict.keys())), 'label'] = filter_name
    else:
        df.loc[df["softmax"] < threshold, 'label'] = filter_name

    return df


def apply_beta_post(df, threshold=0.7):
    copepod_groups = ['copepod_calanoid', 'copepod_cyclopoid', 'copepod_cyclopoid_eggs',
                      'copepod_harpactacoid', 'copepod_harpactacoid_eggs',
                      'copepod_cyclopoid_eggs']
    rotifer_groups = ['rotifer_side_view', 'rotifer_polar_view', 'rotifer_eggs']
    groups_dict = {'copepod_unid': copepod_groups, 'rotifer_unid': rotifer_groups}

    df = apply_threshold_and_2nd_grouping(df, threshold=threshold,
                                          groups_2nd_dict=groups_dict)
    return df


if __name__ == '__main__':
    val_df = plankton_cnn.pvnp_save_and_load_utils.load_val_df(model_name="EfficientNetV2S", train_prefix="run_3_train_model")
    with tf.device('/gpu:0'):
        result = apply_model_to_df(model_name="EfficientNetV2S", train_prefix="run_3_train_model", df=val_df)
    print(result)


    pd.set_option("display.max_columns", None)
    sys.exit()


def calc_metric_for_evaluated_df(df, true_name_label='true_name', pred_name_label='pred_name'):
    """

    :param df: DataFrame - needs to contain the columns with names of true_name_label and pred_name_label
    :param true_name_label: str - name of column in df containing the true names
    :param pred_name_label: str - name of column in df containing the predicted names
    :return: tuple (float, float, float, float, float) - metrics in [0, 1]
    """
    # Calculate accuracy & balanced accuracy
    acc = sklearn.metrics.accuracy_score(df[true_name_label], df[pred_name_label])
    bal_acc = sklearn.metrics.balanced_accuracy_score(df[true_name_label], df[pred_name_label], sample_weight=None,
                                                      adjusted=False)
    prec = sklearn.metrics.precision_score(df[true_name_label], df[pred_name_label], average='weighted',
                                           zero_division=0)
    bal_prec = sklearn.metrics.precision_score(df[true_name_label], df[pred_name_label], average='macro',
                                               zero_division=0)
    f1 = sklearn.metrics.f1_score(df[true_name_label], df[pred_name_label], average='macro',
                                  zero_division=0)

    return acc, bal_acc, prec, bal_prec, f1


def calc_metric_per_group_for_evaluated_df(df, true_name_label='true_name', pred_name_label='pred_name'):
    """
    Calculate a new DataFrame with the precision and recall of the group for each group that
    appears in either true_name_label or in pred_name_label.

    If a group has zero predictions, precision for this group is not defined and is set to 0
    without warning whereas recall is 0 by definition. If on the other hand, a group has zero
    true names, then the recall is not defined and is set to 0 without warning, whereas
    precision is 0 by definition.

    :param df: DataFrame - needs to contain the columns with names of true_name_label and pred_name_label
    :param true_name_label: str - name of column in df containing the true names
    :param pred_name_label: str - name of column in df containing the predicted names
    :return: DataFrame - with columns:
                        - 'name': because a name in this column can be either from true_name_label
                                  or pred_name_label, we omit the wording 'true' and 'predicted in
                                  this column name
                        - 'recall' in [0, 1]
                        - 'precision' in [0, 1]
                        - n_true_name_label: number of occurrences of name in true_name_label
                        - n_pred_name_label: number of occurrences of name in pred_name_label
    """
    # We make a sorted array of all labels that appear in either true_name_label or pred_name_label
    names = np.union1d(df[true_name_label].unique(), df[pred_name_label].unique())
    names = np.sort(names)

    # We also add the number of occurrences per group for both the true names and predictions
    n_true_names = np.array([np.sum(df[true_name_label] == name) for name in names])
    n_pred_names = np.array([np.sum(df[pred_name_label] == name) for name in names])

    # By setting the argument 'labels', we guarantee that the order of the arrays matches
    # the order of the names.
    recall = sklearn.metrics.recall_score(df[true_name_label], df[pred_name_label],
                                          labels=names, average=None, zero_division=0)
    precision = sklearn.metrics.precision_score(df[true_name_label], df[pred_name_label],
                                                labels=names, average=None, zero_division=0)
    f1 = sklearn.metrics.f1_score(df[true_name_label], df[pred_name_label],
                                  labels=names, average=None, zero_division=0)

    return pd.DataFrame({'name': names,
                         'recall': recall,
                         'precision': precision,
                         'f1_score': f1,
                         f"n_{true_name_label}": n_true_names,
                         f"n_{pred_name_label}": n_pred_names})
