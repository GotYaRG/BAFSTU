import tensorflow as tf

import pickle
import os
import pandas as pd
import matplotlib as mpl

from skimage import io
from matplotlib import pyplot as plt
from plankton_cnn.pvnp_visualize import plot_confusion_matrix_for_val_df, plot_history
from plankton_cnn.pvnp_use import apply_model_to_df
from plankton_cnn.pvnp_save_and_load_utils import load_val_df
from plankton_cnn.pvnp_save_and_load_utils import load_history


def unpickle_labeled_images():
    infile = open("../img_viewer/save_files/first_1k_redo", "rb")
    new_df = pickle.load(infile)
    infile.close()
    return new_df


def get_test_img_df(labeled_imgs):
    img_paths, img_labels = [], []
    for dir in os.listdir("../training_data/run_3_val"):
        for img_name in os.listdir(f"training_data/run_3_val/{dir}"):
            img_paths.append(f"training_data/run_3_val/{dir}/{img_name}")
            index = labeled_imgs["image_name"] == img_name
            img_labels.append(labeled_imgs[index]["label"][0])
    df_dict = {"image_path": img_paths, "true_name_label": img_labels}
    new_df = pd.DataFrame(df_dict)
    return new_df


def plateau_plotting(model_name, train_prefix):
    hist = load_history(f"{model_name}_{train_prefix}")

    fig, (ax1, ax2) = mpl.pyplot.subplots(1, 2)
    ax1, ax2 = plot_history(ax1, ax2, hist["r1"])
    fig.suptitle(f"{train_prefix}: Round 1")
    mpl.pyplot.show()


def reset_labels(train_prefix, val_df):
    with open(f"/home/gotya/PycharmProjects/tensorflow/ml_data/checkpoints/{train_prefix}_labels.txt", "r") as labels:
        labels_dict = {}
        for line in labels:
            line = line.strip(" \n").split(": ")
            labels_dict[line[0]] = line[1]
    new_labels = []
    for old_label in list(val_df["label"]):
        new_labels.append(labels_dict[str(old_label)])
    val_df.drop("label", axis=1, inplace=True)
    val_df["true_name"] = new_labels
    return val_df


def true_and_pred_labels(model_name, train_prefix):
    val_df = load_val_df(model_name=model_name, train_prefix=train_prefix)
    with tf.device('/gpu:0'):
        result = apply_model_to_df(model_name=model_name, train_prefix=train_prefix, df=val_df.copy())
    val_df = reset_labels(train_prefix, val_df)
    pd.to_pickle(result, "AdamW_finetuned_result.pickle")
    result = result.rename(columns={"label": "pred_name"})
    merged_df = pd.merge(result, val_df, how="inner", on=["image_path", "image_name"])
    return merged_df


def usable_unusable(df):
    labels_dict = {"digesting": "usable", "eating": "unusable", "gameto_genesis": "unusable",
                   "multiple": "unusable", "multiple_nuclei": "unusable", "not_noctiluca": "unusable",
                   "out_of_focus_or_frame": "unusable", "partial_empty": "unusable"}
    trues, falses = [], []
    for t, f in zip(df["true_name"], df["pred_name"]):
        trues.append(labels_dict[t])
        falses.append(labels_dict[f])
    df["true_name"] = trues
    df["pred_name"] = falses
    return df


def main():
    # Merged labels - Adamw
    #plateau_plotting(model_name="EfficientNetV2S", train_prefix="merged_classes_adam")
    #plot_confusion_matrix_for_val_df(true_and_pred_labels(model_name="EfficientNetV2S", train_prefix="merged_classes_adam"),
    #                                 "merged_classes_adam")

    # Merged classes - finetuning
    #plateau_plotting(model_name="EfficientNetV2S", train_prefix="merged_classes_finetuning")
    #plot_confusion_matrix_for_val_df(
    #    true_and_pred_labels(model_name="EfficientNetV2S", train_prefix="merged_classes_finetuning"),
    #    "merged_classes_finetuning")

    #plateau_plotting(model_name="EfficientNetV2S", train_prefix="merged_classes_adam")
    #plateau_plotting(model_name="EfficientNetV2S", train_prefix="merged_classes_adadelta")
    #plateau_plotting(model_name="EfficientNetV2S", train_prefix="merged_classes_nadam")
    #plateau_plotting(model_name="EfficientNetV2S", train_prefix="merged_classes_rmsprop")

    #plot_confusion_matrix_for_val_df(
    #    true_and_pred_labels(model_name="EfficientNetV2S", train_prefix="merged_classes_rmsprop"),
    #    "merged_classes_rmsprop")
    #print(true_and_pred_labels(model_name="EfficientNetV2S", train_prefix="merged_classes_rmsprop"))
    #plateau_plotting(model_name="EfficientNetV2S", train_prefix="merged_classesYogi")
    #plot_confusion_matrix_for_val_df(usable_unusable(true_and_pred_labels(model_name="EfficientNetV2S", train_prefix="merged_classes_Yogi_finetuning")), "merged_classes_Yogi_finetuning")

    #plateau_plotting(model_name="EfficientNetV2S", train_prefix="merged_classes_AdamW_finetuning")
    plot_confusion_matrix_for_val_df(usable_unusable(true_and_pred_labels(model_name="EfficientNetV2S", train_prefix="merged_classes_AdamW_finetuning")), "merged_classes_AdamW_finetuning")
    plot_confusion_matrix_for_val_df(true_and_pred_labels(model_name="EfficientNetV2S", train_prefix="merged_classes_AdamW_finetuning"), "merged_classes_AdamW_finetuning")

    #print(true_and_pred_labels(model_name="EfficientNetV2S", train_prefix="merged_classes_AdamW_finetuning"))

    """
    labels = true_and_pred_labels(model_name="EfficientNetV2S", train_prefix="merged_classes_Yogi_finetuning")
    for true_name, pred_name, image_name in zip(labels["true_name"], labels["pred_name"], labels["image_name"]):
        if true_name != pred_name:
            img = io.imread(f"training_data/merged_classes_train/{true_name}/{image_name}")
            fix, ax = plt.subplots(1, 1)
            ax.set_title(f"{true_name}, {pred_name}")
            ax.imshow(img)
            plt.show()
    """


main()
