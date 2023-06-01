#!/usr/bin/env python3

import numpy
import sklearn.metrics
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import ImageGrid


# Jasper van Leeuwen
# Two imports had to be commented out in order to prevent circular import errors when importing a few of the
# visualization functions from this script.
# from db_utils.set_paths import *
from plankton_cnn.pvnp_save_and_load_utils import *
# from img_viewer.image_viewer import view_current_labeling
from plankton_cnn.pvnp_save_and_load_utils import load_and_stitch_history
from plankton_cnn.pvnp_use import calc_metric_per_group_for_evaluated_df


def show_subsample(batched_ds):
    """
    :param batched_ds:
    :return:
    """
    img_list = []
    for images, _ in batched_ds.take(1):
        for i in range(9):
            img_list.append(images[i])

    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.3, share_all=True)

    for ax, im in zip(grid, img_list):
        # ax.imshow(im.numpy().astype("uint8"))
        ax.imshow(im.numpy())
        ax.set_axis_off()
    plt.show()


def heatmap(data, text_labels, ax=None, show_cbar=None, contains_prec_recall=False,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels. If contains_prec_recall=True,
    then a separate color mask is calculated for the 'group-values' and the precision/recall-values.

    From: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    text_labels
        Either:
        - list or array of length N with the labels for the rows and columns
        - tuple of list or array of length N, with the labels for the x (1st) and y axis (2nd)
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    if contains_prec_recall:
        # Make a mask of the values of the groups
        groups_mask = np.zeros_like(data).astype(bool)
        groups_mask[:-1, :-1] = True

        # Precision/recall mask (except the corner). By default, corner value is plotted in white
        prec_recall_mask = ~groups_mask
        prec_recall_mask[-1, -1] = False

        for mask in [groups_mask, prec_recall_mask]:
            data_copy = data.copy()
            data_copy[~mask] = np.nan

            im = ax.imshow(data_copy, **kwargs, alpha=mask.astype(float))
            annotate_heatmap(im, use_mask=mask.astype(bool))
    else:
        im = ax.imshow(data, **kwargs)
        annotate_heatmap(im)

    # Create colorbar
    if show_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    if type(text_labels) == tuple:
        ax.set_xticklabels(text_labels[0])
        ax.set_yticklabels(text_labels[1])
    else:
        ax.set_xticklabels(text_labels)
        ax.set_yticklabels(text_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.0f}",
                     textcolors=("black", "white"),
                     threshold=None, use_mask=None, **textkw):
    """
    A function to annotate a heatmap.
    Adapted from:
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Based on use_mask we skip the annotation of data that is marked as False
            if (use_mask is not None) and (not use_mask[i, j]):
                continue

            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_learning_rate():
    from matplotlib import pyplot as plt

    def decayed_learning_rate(step, initial_learning_rate, decay_rate, decay_steps):
        return initial_learning_rate * decay_rate ** (step / decay_steps)

    steps = np.arange(0, 50)
    lr = [decayed_learning_rate(step, 0.001, 0.5, 50) for step in steps]

    fig, ax = plt.subplots()
    ax.scatter(steps, lr)
    plt.show()


def plot_history(ax1, ax2, db_history, stitch_point=None, num_rm=0):
    """
    Make 2 plots with on the x-axis the training epochs and on the y-axis the training and validation accuracy (top)
    and the training and validation loss (bottom). Data should be in a 'history-file' as saved during the training
    procedure.

    :param ax1:
    :param ax2:
    :param db_history:
    :param stitch_point:
    :param num_rm: int - if positive, a rolling mean among the surrounding num_rm points is plotted as well.
    :return:
    """
    acc = db_history['accuracy']
    val_acc = db_history['val_accuracy']
    loss = db_history['loss']
    val_loss = db_history['val_loss']
    epochs_range = np.arange(1, len(db_history) + 1).astype('uint8')

    # Accuracy
    ax1.plot(epochs_range, acc, label='Training acc.', color='tab:blue')
    ax1.plot(epochs_range, val_acc, label='Val. acc.', color='Orange')

    # Loss
    ax2.plot(epochs_range, loss, label='Training loss', color='tab:blue')
    ax2.plot(epochs_range, val_loss, label='Val. loss', color='Orange')

    # Rolling mean
    if num_rm > 0 and len(db_history) > 2. * num_rm:
        x_rm, acc_rm = rolling_mean(acc, num_rm)

        val_acc_rm, loss_rm, val_loss_rm = [
            rolling_mean(np.array(metric), num_rm)[1] for metric in [val_acc, loss, val_loss]]
        ax1.plot(x_rm, acc_rm, color='tab:blue', ls='--')
        label = f"Val. acc. (max.): {np.round(np.amax(val_acc_rm), 3)}"
        ax1.plot(x_rm, val_acc_rm, ls='--', color='Orange', label=label)

        ax2.plot(x_rm, loss_rm, color='tab:blue', ls='--')
        ax2.plot(x_rm, val_loss_rm, color='Orange', ls='--')

    # ax1.set_xticks(epochs_range)
    # ax1.set_xticklabels(epochs_range.astype('str'))
    ax1.set_xlabel('epoch')
    ax1.legend(loc='lower right')
    # ax1.set_title('Training and Validation Accuracy')

    ax2.set_xlabel('epoch')
    ax2.legend(loc='upper right')
    # ax2.set_title('Training and Validation Loss')

    if stitch_point:
        for ax in [ax1, ax2]:
            y_low = ax.get_ylim()[0]
            y_up = ax.get_ylim()[1]
            ax.vlines(stitch_point, y_low, y_up, colors='tab:gray',
                      linestyles='--')
            ax.set_ylim(y_low, y_up)

    return ax1, ax2


def plot_and_load_history(ax1, ax2, model_name, *train_prefix,
                          train_round='best'):
    db_history, stitch_points = load_and_stitch_history(model_name, *train_prefix,
                                                        train_round=train_round)
    ax1, ax2 = plot_history(ax1, ax2, db_history, stitch_point=stitch_points, num_rm=3)

    return ax1, ax2


def rolling_mean(varlist, num):
    rolling_mean = []
    rolling_mean_x = np.arange(num + 1, len(varlist) + 1 - num)

    for i in rolling_mean_x:
        el = np.mean(varlist[i - (num + 1):i + num])
        rolling_mean.append(el)

    return rolling_mean_x, np.array(rolling_mean)


def calc_confusion_matrix_for_val_df(val_df, true_name_label='true_name', pred_name_label='pred_name',
                                     add_precision_recall=False, return_names=True):
    """
    Calculate the confusion matrix by comparing the true names versus the predictions in val_df.
    Confusion matrix is calculated for each group that appears in either true_name_label or
    in pred_name_label and the output is sorted by the name of the group.

    If a group has zero predictions, precision for this group is not defined and is set to 0
    without warning whereas recall is 0 by definition. If on the other hand, a group has zero
    true names, then the recall is not defined and is set to 0 without warning, whereas
    precision is 0 by definition.

    :param val_df: DataFrame - needs to contain columns true_name_label and pred_name_label
    :param true_name_label: str
    :param pred_name_label: str
    :param add_precision_recall: bool
    :param return_names: bool
    :return: (confusion matrix, labels) - np.array (float), np.array (str)
    """

    # We make a sorted array of all labels that appear in either true_name_label or pred_name_label
    names = np.union1d(val_df[true_name_label].unique(), val_df[pred_name_label].unique())
    names = np.sort(names)

    # By setting the argument 'labels', we guarantee that the order of the arrays matches
    # the order of the names.
    conf_matrix = sklearn.metrics.confusion_matrix(val_df[true_name_label],
                                                   val_df[pred_name_label],
                                                   labels=names)

    if add_precision_recall:
        df_metrics = calc_metric_per_group_for_evaluated_df(val_df,
                                                            true_name_label=true_name_label,
                                                            pred_name_label=pred_name_label)

        # As long as the array names in this function is calculated in the same way as in
        # calc_metric_per_group_for_evaluated_df, the order should be the same
        if not np.all(np.array(df_metrics['name']) == names):
            raise ValueError("The order of labels in the confusion matrix and the calculated "
                             "precision and recall does not match. This should be checked.")

        precision = np.array(df_metrics['precision']) * 100.
        recall = np.array(df_metrics['recall']) * 100.

        conf_matrix = np.append(conf_matrix,
                                precision.reshape(1, len(precision)),
                                axis=0)
        conf_matrix = np.append(conf_matrix,
                                np.append(recall, 0).reshape(len(recall) + 1, 1),
                                axis=1)

    if return_names:
        return conf_matrix, names
    else:
        return conf_matrix


def plot_confusion_matrix_for_val_df(val_df, plot_name, true_name_label='true_name', pred_name_label='pred_name',
                                     add_precision_recall=False, figsize=None):
    """
    Calculate the confusion matrix of val_df using calc_confusion_matrix_for_val_df and
    plot as a table with a heatmap. If add_precision_recall = True, include an extra
    row/column for recall/precision (in %).

    :param val_df: DataFrame - needs to contain columns true_name_label and pred_name_label
    :param true_name_label: str
    :param pred_name_label: str
    :param add_precision_recall: bool
    :param figsize:
    :return: None
    """
    conf_matrix, labels = calc_confusion_matrix_for_val_df(val_df, true_name_label=true_name_label,
                                                           pred_name_label=pred_name_label,
                                                           add_precision_recall=add_precision_recall, return_names=True)

    # We need to update the text labels with precision and recall
    if add_precision_recall:
        labels = list(labels)
        labels_x, labels_y = labels.copy(), labels.copy()
        labels_x.append('recall (%)')
        labels_y.append('precision (%)')
        labels = (labels_x, labels_y)

    fig, ax = plt.subplots(figsize=figsize)
    heatmap(conf_matrix, text_labels=labels, ax=ax, cmap='Greys',
            contains_prec_recall=add_precision_recall)
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    fig.tight_layout()
    fig.suptitle(plot_name)
    plt.show()

    return


if __name__ == '__main__':
    pd.set_option("display.max_columns", None)
    pass
