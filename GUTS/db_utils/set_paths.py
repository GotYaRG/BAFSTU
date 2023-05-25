#!/usr/bin/env python3

import os
import pandas as pd

"""
Import this script at the start of all scripts in the project
"""
pd.set_option("display.max_columns", None)

ON_CLUSTER = False

# All paths in entire project should be set relative to this path

#PATH_TO_PROJECT = '/home/gotya/PycharmProjects/tensorflow'
#PATH_TO_PROJECT = "C:/Users/GotYa\OneDrive - Hogeschool Leiden/School/BAFSTU/Python"
PATH_TO_PROJECT = "../"
os.chdir(PATH_TO_PROJECT)

# All important filenames and paths to folders within the project
# should be specified here.

# Databases
PATH_TO_EXTERNAL_STORAGE = '/volumes/Seagate Expansion Drive'
PATH_TO_DATABASES_RELATIVE = "databases"

# Use this one if databases are stored within path_to_project
PATH_TO_DATABASES = PATH_TO_DATABASES_RELATIVE

# Use this one if databases are stored on external drive
# PATH_TO_DATABASES = f"{PATH_TO_EXTERNAL_STORAGE}/{PATH_TO_DATABASES_RELATIVE}"

# if not os.path.isdir(PATH_TO_EXTERNAL_STORAGE):
#     raise Exception(f"Could not read {PATH_TO_EXTERNAL_STORAGE}. Is the external drive mounted?")

DB_NAME = "image_database.db"
DB_PATH = f"{PATH_TO_DATABASES}/{DB_NAME}"

# Archives
CPICS_ARCHIVE_PATH = f"{PATH_TO_DATABASES}/CPICS_archive"
TXT_CPICS_ARCHIVE_PATH = f"{CPICS_ARCHIVE_PATH}/txt_archives"
RBR_ARCHIVE_PATH = f"{PATH_TO_DATABASES}/RBR_archive"
ISIIS_ARCHIVE_FOLDER = "ISIIS_archive"
ISIIS_ARCHIVE_PATH = f"{PATH_TO_DATABASES}/{ISIIS_ARCHIVE_FOLDER}"
# ISIIS_ARCHIVE_PATH = f"{PATH_TO_EXTERNAL_STORAGE}/{ISIIS_ARCHIVE_FOLDER}"

PATH_TO_DEPL_LIST = f"{PATH_TO_DATABASES}/plankton_imaging_deployment_list.xlsx"

PATH_TO_PKLS = "temp_pkls"
PATH_TO_IMG_VIEWER_FILES = "img_viewer/save_files"

# The ML data
#PATH_TO_TRAINING_DATA = '/home/gotya/PycharmProjects/tensorflow/training_data'
#PATH_TO_TRAINING_DATA = "C:/Users/GotYa/OneDrive - Hogeschool Leiden/School/BAFSTU/Python/training_data"
PATH_TO_TRAINING_DATA = "../Data/Imports/"

#PATH_TO_ML_DATA = "/home/gotya/PycharmProjects/tensorflow/ml_data"
#PATH_TO_ML_DATA = "C:/Users/GotYa/OneDrive - Hogeschool Leiden/School/BAFSTU/Python/ml_data"
script_path = os.path.abspath(__file__)
script_path = script_path.split("\\")
script_path = script_path[:-2]
script_path = "/".join(script_path)

PATH_TO_ML_DATA = f"{script_path}/ml_data"

PATH_TO_MODELS = f"{PATH_TO_ML_DATA}/saved_models"
PATH_TO_CHECKPOINTS = f"{PATH_TO_ML_DATA}/checkpoints"
PATH_TO_TUNER_DATA = f"{PATH_TO_ML_DATA}/keras_tuner"
# PATH_TO_TUNER_DATA = "ml_data/keras_tuner"  # on HPC-cluster
PATH_TO_TENSORBOARD_LOGS = f"{PATH_TO_ML_DATA}/tb_logs"


def read_pickle(pkl_name, dir=PATH_TO_PKLS):
    if dir:
        pkl_name = f"{dir}/{pkl_name}"
    df = pd.read_pickle(pkl_name)
    return df


def to_pickle(df, pkl_name, dir=PATH_TO_PKLS, verbose=1):
    """

    :param df:
    :param pkl_name:
    :param dir: str - if None, pkl_name should be absolute
    :param verbose:
    :return:
    """
    if dir:
        pkl_name = f"{dir}/{pkl_name}"
    df.to_pickle(pkl_name)
    if verbose == 1:
        print(f"DataFrame saved to {pkl_name}")
