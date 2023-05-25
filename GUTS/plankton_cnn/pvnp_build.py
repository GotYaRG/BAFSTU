#!/usr/bin/env python3
import urllib.error

import numpy as np
import sys
import shutil
import datetime
import time
import statistics

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model
import tensorflow_hub as tf_hub

from db_utils.set_paths import *
import plankton_cnn.pvnp_import as pvnp_import
import plankton_cnn.pvnp_save_and_load_utils as pvnp_save_and_load_utils
from plankton_cnn.pvnp_models import model_dict, N_COLOR_CHANNELS
from plankton_cnn.pvnp_save_and_load_utils import save_label_dict


def import_and_save_model(model_name):
    """
    Function to store models from tensorflow-hub or locally constructed models, so they can be loaded from the local
    disk afterwards .Models need to be defined in model_dict in pvnp_models.py either as a string
    (a link to tensorflow-hub) or as a function (if locally defined). Function needs to be executed only once for
    every model.

    :param model_name: str - key of the model in model_dict
    :return: None
    """
    model_path = model_dict[model_name]["link"]
    img_size = model_dict[model_name]["img_size"]

    if isinstance(model_path, str):
    # Loading external model
        try:
            layer = tf_hub.KerasLayer(model_path, trainable=False)
        except urllib.error.HTTPError as error:
            print(error)
            print(f"{model_name} skipped")
        else:
            model = tf.keras.Sequential(layer)
            model.build([None, img_size, img_size, N_COLOR_CHANNELS])
            model.save(f"{PATH_TO_MODELS}/{model_name}")
            print(f"model from {model_path} saved to {PATH_TO_MODELS}/{model_name}")
    else:
    # Loading own model
        model = model_path
        model.build([None, img_size, img_size, N_COLOR_CHANNELS])
        model.save(f"{PATH_TO_MODELS}/{model_name}")
        print(f"model from {model_path} saved to {PATH_TO_MODELS}/{model_name}")


def save_new_models():
    """
    Loop over model_dict and save only the models that are not yet stored locally.

    Glob: model_dict
    """
    print("Start loading new models..")
    print(f"{len(model_dict)} models present in model_dict")
    counter = 0
    for model_name in model_dict.keys():
        print()
        try:
            load_model(model_name)
        except OSError:
            print(f"{model_name} not found at local storage")
            import_and_save_model(model_name)
            counter += 1
    print(f"{counter} models loaded and stored locally")


def load_model(model_name, model_path=PATH_TO_MODELS, verbose=True):
    """
    Import model_name from local storage. Each model is stored as a folder named 'model_name'.
    Raises OSError if SavedModel file does not exist at 'model_path'

    :param model_name: str - name of model
    :param model_path: str - path to folder with models.
    :param verbose: bool
    :return: TensorFlow model
    """
    model_local_path = f"{model_path}/{model_name}"
    model = keras.models.load_model(model_local_path)

    if verbose:
        print(f"Located model {model_name} at local storage")

    return model


def build_model(model_name, num_classes, classifier_activation='softmax', dropout=None,
                use_imagenet_weights=True):
    """
    TO DO: check how new, locally defined models are handled

    Load a generic model and construct it based on the specific requirements of the training data. Among others, a final
    dense layer is appended based on the number of classes. For the existing models, specific features such as
    preprocessing layers, final layers and kernel initializers are defined in accordance with the fully trained models
    as they exist in keras.applications.

    :param model_name: str - key of the model in model_dict
    :param num_classes: int - number of classes in the training data
    :param classifier_activation: name of the activation in the final dense layer. Any argument that can be passed
                                  to layers.Dense is accepted.
    :param dropout: bool - if True, a dropout-layer is added before the final layers. Only applies to tf-hub models!
    :param use_imagenet_weights: bool - if True, weights of the model is trained on ImageNet are used. If False,
                                        the model weights are newly initialized.
    :return: tf.model
    """
    model_path = model_dict[model_name]["link"]
    img_size = model_dict[model_name]["img_size"]

    if isinstance(model_path, str):
        # if hub-model at local storage
        model = load_model(model_name)

        if dropout:
            model.add(layers.Dropout(dropout))

        model.add(layers.Dense(num_classes, activation=classifier_activation))

    else:
        # if from Keras applications
        if use_imagenet_weights:
            weights = 'imagenet'
        else:
            weights = None

        base_model = model_path(include_top=False, weights=weights,
                                input_shape=(img_size, img_size, N_COLOR_CHANNELS))

        # check what 'training parameter' does in batch normalization layers.

        kernel_initializer = 'glorot_uniform'
        bias_initializer = 'zeros'

        if model_name == 'Xception':
            # Done
            preprocess = tf.keras.applications.xception.preprocess_input
            preds = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)

        elif model_name in ['VGG16', 'VGG19']:
            # Done
            if model_name == 'VGG16':
                preprocess = tf.keras.applications.vgg16.preprocess_input
            else:
                preprocess = tf.keras.applications.vgg19.preprocess_input

            preds = layers.Flatten(name='flatten')(base_model.output)
            preds = layers.Dense(4096, activation='relu', name='fc1')(preds)
            preds = layers.Dense(4096, activation='relu', name='fc2')(preds)

        elif model_name in ['ResNet50V2', 'ResNet101V2', 'ResNet152V2']:
            # Done
            preprocess = tf.keras.applications.resnet_v2.preprocess_input
            preds = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)

        elif model_name == 'InceptionV3':
            # Done
            preprocess = tf.keras.applications.inception_v3.preprocess_input
            preds = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)

        elif model_name == 'InceptionResNetV2':
            # Done
            preprocess = tf.keras.applications.inception_resnet_v2.preprocess_input
            preds = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)

        elif model_name == 'MobileNetV2':
            # Done
            preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
            preds = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)

        elif model_name in ['DenseNet121', 'DenseNet169', 'DenseNet201']:
            # Done
            preprocess = tf.keras.applications.densenet.preprocess_input
            preds = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)

        elif model_name in ['NASNetMobile', 'NASNetLarge']:
            # Done
            preprocess = tf.keras.applications.nasnet.preprocess_input
            preds = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)

        elif model_name in ['EfficientNetV2B0', 'EfficientNetV2B1',
                            'EfficientNetV2B2', 'EfficientNetV2B3',
                            'EfficientNetV2S']:
            preprocess = layers.Rescaling(1.)
            preds = layers.GlobalAveragePooling2D()(base_model.output)

            kernel_initializer = {"class_name": "VarianceScaling",
                                  "config": {"scale": 1. / 3.,
                                             "mode": "fan_out",
                                             "distribution": "uniform"}}
            bias_initializer = tf.constant_initializer(0)

        # Build the preprocessing model
        # The model-specific preprocessing layers require the model input to be in [0, 255]. Here we
        # assume the model input is in [0, 1] conform our import-function, so we add a rescaling layer.
        x = layers.Rescaling(255)(base_model.input)
        x = preprocess(x)
        input_model = Model(inputs=base_model.input, outputs=x,
                            name=f"Preprocessing-{model_name}")

        # Build the final classifier model
        preds = layers.Dense(num_classes,
                             activation=classifier_activation,
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             name='predictions')(preds)

        classifier_model = Model(inputs=base_model.output,
                                 outputs=preds,
                                 name='Classifier')

        # Build the complete model, using the preprocessing, base and classifier model
        y = input_model(input_model.input)
        y = base_model(y, training=False)
        # Setting training to False has to do with the batch normalization layers
        # see https://keras.io/guides/transfer_learning/#freezing-layers-understanding-the-trainable-attribute
        y = classifier_model(y)
        model = Model(inputs=input_model.input, outputs=y,
                      name=f"{model_name}-imnet")

    return model


def calc_class_weights(train_dict):
    """
    Calculate the weights of the groups in train_dict.

    The weights are such that they balance the length of the groups in order to
    obtain equal contribution to the network's loss function. I.e., by passing these weights
    to the training procedure, errors in rare groups get a larger contribution to the loss function
    whereas the contribution from errors in abundant groups becomes smaller. This prevents the network
    from fitting too much to the most abundant groups only.

    :param train_dict: dict - as returned by pvnp_import.import_plankton_images
    with the groups' integer labels as keys, as items dictionaries with key 'length' and 'name'
    :return: dict - with integer labels as keys, weights per group as items
    """
    mean_length = statistics.mean(map(
        lambda key: train_dict[key]['length'], train_dict.keys()))
    weights = list(map(
        lambda key: mean_length / train_dict[key]['length'], train_dict.keys()))
    return dict(zip(train_dict.keys(), weights))


def run_training_procedure(model_name, train_prefix_new, train_prefix_prev,
                           import_args, finetuning, optimizer, loss,
                           epochs, nr_rounds, perf_monitor, es_min_delta, es_patience, es_verbose,
                           fit_verbose,
                           use_imagenet_weights=True, classifier_activation=None, dropout=None):
    """
    Run the full training procedure: import the training data from the local storage, import a locally stored model,
    run the actual training and save the best model result and training history both as tensorboard logs and as pkl.
    Removes any existing saved files with the same train_prefix_new.

    :param model_name: str
    :param train_prefix_new: str - name of the training procedure that is used in the resulting saved files
    :param train_prefix_prev: str - if specified, load an already trained model to continue training. Only the model is
                                    loaded, any training parameters need to be specified again.
    :param import_args:
    :param finetuning: bool - if False, only the final dense layer(s) are trained, all others are 'frozen'
    :param optimizer: keras.optimizers-function
    :param loss: keras.losses-function
    :param epochs: int - number of training epochs
    :param nr_rounds: int - number of rounds. If > 1, multiple rounds with the same parameters are executed and the best
                            results are selected
    :param perf_monitor: str - should be one of ['loss', 'val_loss', 'accuracy', 'val_accuracy']. Metric that is used
                               to assess model performance
    :param es_patience: int - maximum number of epochs that can be pass without improvement before early stopping
    :param es_min_delta: float - minimum improvement in perf_monitor in order to prevent early stopping
    :param es_verbose:
    :param fit_verbose: int - Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    :param use_imagenet_weights: bool
    :param classifier_activation: str
    :param dropout: bool -
    :return: float - elapsed time during the fitting procedure

    """
    save_name = f"{model_name}_{train_prefix_new}"
    import_args['image_size'] = model_dict[model_name]["img_size"]

    # Load the training and validation data. Depending on the folder structure, the split is either made during the
    # import function or defined by the folders itself (i.e. training and validation data in separate folders)
    training_data_dir_train, training_data_dir_val, import_args['validation_split'] = \
        pvnp_save_and_load_utils.import_train_or_val_dir(import_args['directory'], import_args['validation_split'])

    import_args_mod = import_args.copy()
    import_args_mod.pop('directory')

    print(f"\nImporting from {training_data_dir_train} and {training_data_dir_val}")
    train_ds, train_dict = pvnp_import.import_plankton_images(
        training_data_dir_train, subset='training', **import_args_mod)
    val_ds, _ = pvnp_import.import_plankton_images(
        training_data_dir_val, subset='validation', **import_args_mod)

    class_weights = calc_class_weights(train_dict)

    print("For training:")
    for dic in train_dict.values():
        print(f" - using {dic['length']} files with label {dic['name']}")

    if train_prefix_prev:
        # Resume training from saved model
        model = keras.models.load_model((name := f"{PATH_TO_MODELS}/{model_name}_{train_prefix_prev}"))
        print(f"Loaded model {name}")
    else:
        # Load new model
        model = build_model(model_name, len(train_dict),
                            classifier_activation=classifier_activation,
                            dropout=dropout,
                            use_imagenet_weights=use_imagenet_weights)

    if finetuning:
        for layer in model.layers:  # Could also be the last xx layers
            layer.trainable = True
    else:
        for layer in model.layers[:-1]:
            layer.trainable = False

    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    save_name_cp = f"{PATH_TO_CHECKPOINTS}/{save_name}"
    if os.path.isdir(save_name_cp):
        shutil.rmtree(save_name_cp)
        print(f"Existing folder {save_name_cp} was removed")

    # Early stopping - fit procedure stops if es_monitor hasn't increased
    # over last xx epochs
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor=perf_monitor, min_delta=es_min_delta, patience=es_patience,
        verbose=es_verbose, mode="auto", restore_best_weights=True)

    # Loop over nr_rounds in order to reduce random variation
    initial_weights = model.get_weights()
    t_before = time.time()
    i, prev_max = 1, 0.
    while i <= nr_rounds:
        print(f"\n round nr. {i}")

        checkpoint_path = f"{save_name_cp}/r{i}_best_cp.ckpt"
        model.set_weights(initial_weights)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=True,
            save_best_only=True, monitor=perf_monitor, verbose=1)

        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                            callbacks=[cp_callback,
                                       es_callback,
                                       keras.callbacks.TensorBoard(f"{PATH_TO_TENSORBOARD_LOGS}/{save_name}_r{i}"),
                                       keras.callbacks.TerminateOnNaN()],
                            class_weight=class_weights,
                            verbose=fit_verbose)
        pvnp_save_and_load_utils.save_history(history, save_name_cp, f"r{i}")

        # select training round that got the best es_monitor
        if perf_monitor in ['loss', 'val_loss']:
            max_acc = min(history.history[perf_monitor])
            if max_acc < prev_max:
                prev_max, checkpoint_path_max, history_max = max_acc, checkpoint_path, history
                i_max = i
        elif perf_monitor in ['accuracy', 'val_accuracy']:
            max_acc = max(history.history[perf_monitor])
            if max_acc > prev_max:
                prev_max, checkpoint_path_max, history_max = max_acc, checkpoint_path, history
                i_max = i
        else:
            print(f"Could not determine best training round, because current es_monitor:")
            print(f"{perf_monitor} was not coded here, so we don't know whether to use min or max")
        i += 1
    t_elapsed = time.time() - t_before

    # Save model weights and whole model from the training round that got
    # the best validation accuracy
    print(f"\n Max. {perf_monitor} encountered: {prev_max} in round {i_max}")
    model.load_weights(checkpoint_path_max)
    train_prefix_max = checkpoint_path_max.replace(f"r{i_max}_", '')
    model.save_weights(train_prefix_max)
    print(f"model weights from round {i_max} saved to {train_prefix_max}")

    save_path = f"{PATH_TO_MODELS}/{save_name}"
    model.save(save_path)
    print(f"model was saved to {save_path}")

    pvnp_save_and_load_utils.save_history(history_max, save_name_cp, "best")

    return t_elapsed


if __name__ == '__main__':
    import tensorflow_addons as tfa
    from tensorflow import keras
    from keras import layers
    import keras_tuner

    name = "EfficientNetV2B0"
    model = build_model(name, 3, use_imagenet_weights=True)
    model.summary()

    submodel = model.get_layer('efficientnetv2-b0')
    submodel.trainable = False
    submodel.summary()
    # sys.exit()

    submodel = model.get_layer(name='Classifier')
    submodel.summary()
    sys.exit()

    # model = build_model("EfficientNetV2B0", 28, use_imagenet_weights=True)
    # sys.exit()

    df = pvnp_save_and_load_utils.load_val_df("beta_020_nu", subset='full')
    print(df)
    sys.exit()
    # save_new_models()
    # # load_training_args(f"{PATH_TO_CHECKPOINTS}/mobilenet_v2_050_160_test/training_args.txt")
    # sys.exit()

    model = load_model("efficientnet_v2_240")
    model.summary()
    # save_label_dict("alpha")
    # print(load_label_dict("alpha"))


"""
Structure outline:

# Global vars
dict{'my_model_name': {'link': link_to_model, 'img_size': img_size}}
with: - my_model_name = **net_v*_scale-ifappl_imgsize
      - model_local_path = {PATH_TO_MODELS}/my_model_name
n_color_channels

# Import and save
import_and_save_model(my_model_name) (only once for new models, without final dense layer)
    - save model to local destination

load_model()

build_model(my_model_name, num_classes, activation, dropout=None)
    - load model from local destination
    - add drop-out if desired
    - add final dense layer, dependent on num_classes, activation
    - return model

# Training
load plankton images(img_size)

checkpoint_path = my_model_name_<r_roundnr>_<training-project(alpha, beta, gamma, delta, eps...)>
fit_model(model, dataset, batch_size, checkpoint_path, n_epochs)
    - remove existing history-files with same training name
    - implement possibility to train in 2 rounds, i.e. last layer only and then repeat for full model
    - save model weights to checkpoint_path for all training rounds
    - save whole model to {PATH_TO_MODELS}/<model_name>_<training_project>
    - save history for each training round
    return none
"""


def test_if_script_continues(length_s, interval=10):
    """
    Quickly check if a script continues when computer is suspended, by printing current time to the terminal at regular
    intervals.

    :param length_s: int - duration of script in seconds
    :param interval: int - duration of interval at which current time is printed.
    :return:
    """
    print("\n Starting test_if_script_continues() \n")
    now = datetime.datetime.now
    start = now()

    counter, t_elapsed = 0, now() - start
    while t_elapsed < datetime.timedelta(seconds=length_s):
        t_now = now()
        t_elapsed = t_now - start
        print(t_now.strftime('%H:%M:%S'))
        time.sleep(interval)
        counter += 1

    print(f"time counted: {(counter * interval)} seconds")
    print(f"time elapsed: {t_elapsed.seconds} seconds")

    print("\n test_if_script_continues() finished")