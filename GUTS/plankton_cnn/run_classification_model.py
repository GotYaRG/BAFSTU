import os
import pandas as pd
from plankton_cnn.pvnp_use import apply_model_to_df


"""
Classification model
Not all images of N. scintillans can have their gut fullness assesed by the
gut fullness algorithm. In order to discern which images can be assessed, they
are classified by the classifier used in this script.
"""


def get_images_df(run_name, project_path):
    """
    Obtains the image names and paths of the images that need to be classified.

    :param run_name: The name of the current run of GUTS.

    :param project_path: The current absolute path of the project where the
    script is located.

    :return: A dataframe of images to classify, with their names and paths.
    """
    img_lib_path = f"{project_path}/Data/Imports/images/{run_name}"
    img_paths, img_names = [], []
    for img_name in os.listdir(img_lib_path):
        img_paths.append(f"{img_lib_path}/{img_name}")
        img_names.append(img_name)
    df_dict = {"image_path": img_paths, "image_name": img_names}
    new_df = pd.DataFrame(df_dict)
    return new_df


def run_classifier(run_name):
    """
    Houses the function calls needed to run the classification model on the
    images of this run of GUTS.

    :param run_name: The name of the current run of GUTS.
    """
    project_path = "/".join(os.path.abspath(__file__).split("\\")[:-2])
    df = get_images_df(run_name, project_path)
    predicted_labels = apply_model_to_df(model_name="EfficientNetV2S", train_prefix="merged_classes_AdamW_finetuning",
                                         df=df.copy())
    pd.to_pickle(predicted_labels, f"{project_path}/Data/Exports/classifier_output/{run_name}_classifications.pickle")


def main():
    run_classifier("")


if __name__ == "__main__":
    main()
