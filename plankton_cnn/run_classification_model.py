import os
import pandas as pd


from plankton_cnn.pvnp_use import apply_model_to_df


def get_images_df():
    img_lib_path = "C:/Users/GotYa/OneDrive - Hogeschool Leiden/School/BAFSTU/Python/manual_classifier/img_viewer/noctiluca_sample_Scharendijke2021"
    img_paths, img_names = [], []
    for img_name in os.listdir(img_lib_path):
        img_paths.append(f"{img_lib_path}/{img_name}")
        img_names.append(img_name)
    df_dict = {"image_path": img_paths, "image_name": img_names}
    new_df = pd.DataFrame(df_dict)
    return new_df


def main():
    df = get_images_df()
    predicted_labels = apply_model_to_df(model_name="EfficientNetV2S", train_prefix="merged_classes_AdamW_finetuning", df=df.copy())
    pd.to_pickle(predicted_labels, "predicted_labels.pickle")


if __name__ == "__main__":
    main()
