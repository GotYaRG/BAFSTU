#!/usr/bin/env python3

import math
from functools import partial

import tensorflow_addons as tfa

import plankton_cnn.pvnp_save_and_load_utils
import plankton_cnn.pvnp_build as pvnp_build
from plankton_cnn.pvnp_save_and_load_utils import *


def run_training_general():
    """
    :param model_name: str
    Can be any model that appears in model_dict (see pvnp_models). I would recommend EfficientNetV2S for now.
    :param train_prefix_new: str
    This is the name that is used for all saved files regarding this training run. Choose it wisely (easy to recognise
    for yourself but not too long).
    :param train_prefix_prev:
    :param use_imagenet_weights: bool
    If True, use pretrained weights from the model trained on ImageNet for transfer learning (recommended). If False,
    the model is initialised with random weights.
    :param finetuning: bool
    If True, the whole model is trainable. If False, only the final layers are trainable and the rest of the model is
    'frozen'. If using transfer learning, it is standard practice to first train the model with finetuning = False,
    and then do some additional epochs with finetuning = True (and a very low learning rate)
    :param directory: str
    Directory of the training & validation data. Directory is expected to contain the images
    in a folder named with the corresponding label for every label
    :param batch_size: int
    during training, data goes into batches into the parameter optimisation. After each batch, the parameters are
    updated. Without going into details, 32 is a good default but for small training sets, it might help to use 16.
    :param seed: int
    random seed of the training/validation split. Always use one so you can repeat the data import without changing the
    training and validation sets. I always keep this at 1234.
    :param augment: str
    Can be 'simple' or 'all_rotations' or None.
    :param validation_split: float in [0, 1]
    If provided, the script automatically makes a (stratified) division between training and
    validation data, with the validation data the fraction specified here. Another possibility
    is to manually put the training and validation data in separate folders, in that case
    this argument is ignored.
    :param as_grayscale: bool
    if True, images are imported as grayscale instead of RGB. Keep at False unless there is an interesting reason to
    change this.
    :param epochs: int
    number of epochs the training continues within a training round. For now, 50 should do I think
    :param nr_rounds: int
    number of training rounds. If > 1, the exact training procedure is repeated with the model initialised again.
    After the final round, the best result is selected. The idea is to reduce randomness which is inherent to
    CNN-training, but the computational cost is high, so I wouldn't use >3, and to start with I think just 1 should do
    as well.
    :param perf_monitor: str
    should be one 'loss', 'accuracy', 'val_loss', 'val_accuracy'. No reason to use anything else than 'val_accuracy'
    :param es_patience: int
    'es' is for 'early stopping'. It means that after a certain amount of epochs the training round stops if the
    'perf_monitor' hasn't increased anymore. For now, 25 should do I think
    :param es_min_delta: float
    the minimum performance difference that counts as an improvement for the early stopping algorithm. I usually keep
    this at 0 (so any improvement counts)
    :param es_verbose:
    amount of information that is printed about the early stopping. No reason to change this I think
    :param fit_verbose:  int
    Verbosity mode during the model fit:
    0 = silent,
    1 = progress bar (fun if running a script directly in terminal or console),
    2 = one line per epoch (more useful if printing output to a file).
    :return:
    """

    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = "run_3_train_model"
        train_prefix_prev = None
        use_imagenet_weights = True
        finetuning = False

        # Training data arguments
        import_args = {
            'directory': "run_3_train",
            'batch_size': 32,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        # Define the learning rate and weight decay of the optimizer
        # I think for EfficientNetV2S, lr of 0.01 and wd of around 1e-5 should work.
        # if your model doesn't converge properly, then change the weight decay. You can also
        # easily change the optimizer, you can have a look in the tensorflow documentation
        # how this works (I would suggest 'Adam' as an alternative for SGDW)
        lr_start, wd_start = 0.01, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            # For finetuning I usually keep a fixed learning rate and weight decay because it's a limited number of
            # epochs anyway. Try lr_start = 0.001, wd_start between 1e-4 and 1e-6.
            lr = lr_start
            wd = wd_start
        else:
            # For training the final layers (opposed to finetuning) I found it useful to use a decay function
            # for the learning rate. It is an exponential decay function, such that the learning rate is halved every
            # epochs_half nr (with 25 a good default value, i.e. in this way the learning rate after 25 epochs
            # is 0.5 * lr_start

            # We need the total length of the training set for the decay function, so the number
            # of decay steps can be calculated taking into account the augmentations if applicable.
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tfa.optimizers.SGDW(learning_rate=lr, weight_decay=wd, nesterov=True, momentum=0.9, name="SGDW"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 50,
            'nr_rounds': 3,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def first_real_run():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = "run_3_train_model"
        train_prefix_prev = None
        use_imagenet_weights = True
        finetuning = False

        # Training data arguments
        import_args = {
            'directory': "run_3_train",
            'batch_size': 32,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.01, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tfa.optimizers.SGDW(learning_rate=lr, weight_decay=wd, nesterov=True, momentum=0.9, name="SGDW"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def first_real_run_gray():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = "run_3_train_model_gray"
        train_prefix_prev = None
        use_imagenet_weights = True
        finetuning = False

        # Training data arguments
        import_args = {
            'directory': "run_3_train",
            'batch_size': 32,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': True}

        lr_start, wd_start = 0.01, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tfa.optimizers.SGDW(learning_rate=lr, weight_decay=wd, nesterov=True, momentum=0.9, name="SGDW"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = "merged_classes"
        train_prefix_prev = None
        use_imagenet_weights = True
        finetuning = False

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 32,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.01, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tfa.optimizers.SGDW(learning_rate=lr, weight_decay=wd, nesterov=True, momentum=0.9, name="SGDW"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_gray():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = "merged_classes_gray"
        train_prefix_prev = None
        use_imagenet_weights = True
        finetuning = False

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 32,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': True}

        lr_start, wd_start = 0.01, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tfa.optimizers.SGDW(learning_rate=lr, weight_decay=wd, nesterov=True, momentum=0.9, name="SGDW"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_finetuning():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = "merged_classes_finetuning"
        train_prefix_prev = "merged_classes_adam"
        use_imagenet_weights = True
        finetuning = True

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 4,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.0001, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd, name="AdamW"), # tfa.optimizers.SGDW(learning_rate=lr, weight_decay=wd, nesterov=True, momentum=0.9, name="SGDW"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_adam():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = "merged_classes_adam"
        train_prefix_prev = None
        use_imagenet_weights = True
        finetuning = False

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 32,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.01, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:

            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd, name="AdamW"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_adadelta():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = f"merged_classes_adadelta"
        train_prefix_prev = None
        use_imagenet_weights = True
        finetuning = False

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 64,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.01, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)
            wd = 0.95

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tf.keras.optimizers.Adadelta(learning_rate=lr, weight_decay=wd, name="Adadelta"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 300,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_adagrad():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = f"merged_classes_adagrad"
        train_prefix_prev = None
        use_imagenet_weights = True
        finetuning = False

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 64,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.01, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)
            wd = 0.95

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tf.keras.optimizers.Adagrad(learning_rate=lr, weight_decay=wd, name="Adagrad"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_adamax():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = f"merged_classes_adamax"
        train_prefix_prev = None
        use_imagenet_weights = True
        finetuning = False

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 64,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.01, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)
            wd = 0.95

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tf.keras.optimizers.Adamax(learning_rate=lr, weight_decay=wd, name="Adamax"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_ftrl():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = f"merged_classes_ftrl"
        train_prefix_prev = None
        use_imagenet_weights = True
        finetuning = False

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 64,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.01, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)
            wd = 0.95

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tf.keras.optimizers.Ftrl(learning_rate=lr, weight_decay=wd, name="Ftrl"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_nadam():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = "merged_classes_nadam"
        train_prefix_prev = None
        use_imagenet_weights = True
        finetuning = False

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 64,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.01, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)
            wd = 0.95

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tf.keras.optimizers.Nadam(learning_rate=lr, weight_decay=wd, name="Nadam"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_rmsprop():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = "merged_classes_rmsprop"
        train_prefix_prev = None
        use_imagenet_weights = True
        finetuning = False

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 64,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.01, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)
            wd = 0.95

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tf.keras.optimizers.RMSprop(learning_rate=lr, weight_decay=wd, name="RMSprop"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_nadam_finetuning():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = "merged_classes_nadam_finetuning"
        train_prefix_prev = "merged_classes_nadam"
        use_imagenet_weights = True
        finetuning = True

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 4,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.0001, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)
            wd = 0.95

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tf.keras.optimizers.Nadam(learning_rate=lr, weight_decay=wd, name="Nadam"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_rmsprop_finetuning():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = "merged_classes_rmsprop_finetuning"
        train_prefix_prev = "merged_classes_rmsprop"
        use_imagenet_weights = True
        finetuning = True

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 4,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.0001, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)
            wd = 0.95

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tf.keras.optimizers.RMSprop(learning_rate=lr, weight_decay=wd, name="RMSprop"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_adam_finetuning():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = "merged_classes_adam_finetuning"
        train_prefix_prev = "merged_classes_adam"
        use_imagenet_weights = True
        finetuning = True

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 4,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.0001, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:

            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd, name="AdamW"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_rmsprop_s_augmentation():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = "merged_classes_rmsprop_s_augment"
        train_prefix_prev = None
        use_imagenet_weights = True
        finetuning = False

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 64,
            'seed': 1234,
            'augment': 'simple', # 'simple' or 'all_rotations'
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.01, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)
            wd = 0.95

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tf.keras.optimizers.RMSprop(learning_rate=lr, weight_decay=wd, name="RMSprop"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_rmsprop_f_augmentation():
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = "merged_classes_rmsprop_f_augment"
        train_prefix_prev = None
        use_imagenet_weights = True
        finetuning = False

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 16,
            'seed': 1234,
            'augment': 'all_rotations', # 'simple' or 'all_rotations'
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.01, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)
            wd = 0.95

        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': tf.keras.optimizers.RMSprop(learning_rate=lr, weight_decay=wd, name="RMSprop"),
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_tfa_optimizers(opt):
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = f"merged_classes{opt}"
        train_prefix_prev = None
        use_imagenet_weights = True
        finetuning = False

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 32,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.01, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)

        opt_dict = {"AdaBelief": tfa.optimizers.AdaBelief(learning_rate=lr, weight_decay=wd),
                    "AdamW": tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd),
                    "COCOB": tfa.optimizers.COCOB(),
                    #"ConditionalGradient": tfa.optimizers.ConditionalGradient(learning_rate=lr),
                    #"CyclicalLearningRate": tfa.optimizers.CyclicalLearningRate(learning_rate=lr, weight_decay=wd),
                    #"DecoupledWeightDecayExtension": tfa.optimizers.DecoupledWeightDecayExtension(weight_decay=wd),
                    #"ExponentialCyclicalLearningRate": tfa.optimizers.ExponentialCyclicalLearningRate(learning_rate=lr, weight_decay=wd),
                    #"LAMB": tfa.optimizers.LAMB(learning_rate=lr, weight_decay=wd),
                    #"LazyAdam": tfa.optimizers.LazyAdam(learning_rate=lr, decay=wd),
                    #"Lookahead": tfa.optimizers.Lookahead(learning_rate=lr, weight_decay=wd),
                    #"MovingAverage": tfa.optimizers.MovingAverage(learning_rate=lr, weight_decay=wd),
                    #"MultiOptimizer": tfa.optimizers.MultiOptimizer(learning_rate=lr, weight_decay=wd),
                    #"NovoGrad": tfa.optimizers.NovoGrad(learning_rate=lr, weight_decay=wd),
                    #"ProximalAdagrad": tfa.optimizers.ProximalAdagrad(learning_rate=lr),
                    #"RectifiedAdam": tfa.optimizers.RectifiedAdam(learning_rate=lr, weight_decay=wd),
                    #"SGDW": tfa.optimizers.SGDW(learning_rate=lr, weight_decay=wd),
                    #"SWA": tfa.optimizers.SWA(learning_rate=lr, weight_decay=wd),
                    #"Triangular2CyclicalLearningRate": tfa.optimizers.Triangular2CyclicalLearningRate(learning_rate=lr, weight_decay=wd),
                    #"TriangularCyclicalLearningRate": tfa.optimizers.TriangularCyclicalLearningRate(learning_rate=lr, weight_decay=wd),
                    "Yogi": tfa.optimizers.Yogi(learning_rate=lr)}
        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': opt_dict[opt],
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


def merged_classes_tfa_optimizers_finetuning(opt):
    # Initialize GPU
    with tf.device('/gpu:0'):

        # Model setup
        model_name = 'EfficientNetV2S'
        train_prefix_new = f"merged_classes_{opt}_finetuning"
        train_prefix_prev = f"merged_classes{opt}"
        use_imagenet_weights = True
        finetuning = True

        # Training data arguments
        import_args = {
            'directory': "merged_classes_train",
            'batch_size': 4,
            'seed': 1234,
            'augment': None,
            'validation_split': 0.18,
            'as_grayscale': False}

        lr_start, wd_start = 0.0001, 1e-5
        wd_start = lr_start * wd_start

        if finetuning:
            lr = lr_start
            wd = wd_start
        else:
            total_length = get_total_length_with_augment(import_args['directory'],
                                                         import_args['augment'],
                                                         import_args['validation_split'])

            epochs_half_nr = 25
            decay_fn = partial(keras.optimizers.schedules.ExponentialDecay,
                               decay_steps=(math.ceil(total_length / import_args["batch_size"]) * epochs_half_nr),
                               decay_rate=0.5)
            lr = decay_fn(initial_learning_rate=lr_start)
            wd = decay_fn(initial_learning_rate=wd_start)

        opt_dict = {"AdaBelief": tfa.optimizers.AdaBelief(learning_rate=lr, weight_decay=wd),
                    "AdamW": tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd),
                    "COCOB": tfa.optimizers.COCOB(),
                    #"ConditionalGradient": tfa.optimizers.ConditionalGradient(learning_rate=lr),
                    #"CyclicalLearningRate": tfa.optimizers.CyclicalLearningRate(learning_rate=lr, weight_decay=wd),
                    #"DecoupledWeightDecayExtension": tfa.optimizers.DecoupledWeightDecayExtension(weight_decay=wd),
                    #"ExponentialCyclicalLearningRate": tfa.optimizers.ExponentialCyclicalLearningRate(learning_rate=lr, weight_decay=wd),
                    #"LAMB": tfa.optimizers.LAMB(learning_rate=lr, weight_decay=wd),
                    #"LazyAdam": tfa.optimizers.LazyAdam(learning_rate=lr, decay=wd),
                    #"Lookahead": tfa.optimizers.Lookahead(learning_rate=lr, weight_decay=wd),
                    #"MovingAverage": tfa.optimizers.MovingAverage(learning_rate=lr, weight_decay=wd),
                    #"MultiOptimizer": tfa.optimizers.MultiOptimizer(learning_rate=lr, weight_decay=wd),
                    #"NovoGrad": tfa.optimizers.NovoGrad(learning_rate=lr, weight_decay=wd),
                    #"ProximalAdagrad": tfa.optimizers.ProximalAdagrad(learning_rate=lr),
                    #"RectifiedAdam": tfa.optimizers.RectifiedAdam(learning_rate=lr, weight_decay=wd),
                    #"SGDW": tfa.optimizers.SGDW(learning_rate=lr, weight_decay=wd),
                    #"SWA": tfa.optimizers.SWA(learning_rate=lr, weight_decay=wd),
                    #"Triangular2CyclicalLearningRate": tfa.optimizers.Triangular2CyclicalLearningRate(learning_rate=lr, weight_decay=wd),
                    #"TriangularCyclicalLearningRate": tfa.optimizers.TriangularCyclicalLearningRate(learning_rate=lr, weight_decay=wd),
                    "Yogi": tfa.optimizers.Yogi(learning_rate=lr)}
        fit_args = {
            'model_name': model_name,
            'train_prefix_new': train_prefix_new,
            'train_prefix_prev': train_prefix_prev,
            'classifier_activation': 'softmax',  # No reason to change this
            'use_imagenet_weights': use_imagenet_weights,
            'optimizer': opt_dict[opt],
            'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}

        fit_kwargs = {
            'epochs': 75,
            'nr_rounds': 1,
            'finetuning': finetuning,
            'perf_monitor': "val_accuracy",  # options should be: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            'es_min_delta': 0,
            'es_patience': 25,
            'es_verbose': 1,
            'fit_verbose': 2}

        t_elapsed = pvnp_build.run_training_procedure(import_args=import_args, **fit_args, **fit_kwargs)
        print("elapsed time in fit_model:", t_elapsed)

        all_args = dict(fit_args,
                        **import_args,
                        **{'learning_rate': lr, 'weight_decay': wd,
                           't_elapsed_sec': t_elapsed},
                        **fit_kwargs)
        plankton_cnn.pvnp_save_and_load_utils.save_training_args(all_args, f"{model_name}_{train_prefix_new}")

        save_label_dict_new(train_prefix_new, import_train_or_val_dir(import_args['directory'])[0])


if __name__ == '__main__':
    # merged_classes_adam()
    # Good performance 80 - 84%

    # merged_classes_adadelta()
    # Pretty bad, plateaud at around 55% after 150+ epochs

    # merged_classes_adagrad()
    # Mediocre performance

    # merged_classes_adamax()
    # Good performance but not as good as Adam

    # merged_classes_ftrl()
    # Mediocre performance

    # merged_classes_nadam()
    # Good performance, needs to be compared to adam, 80-82%

    # merged_classes_rmsprop()
    # Sparatic performance, but possibly better than both adam's, 83-84%


    #merged_classes_adam_finetuning()
    # lr_start, wd_start = 0.0001, 1e-5 undoubtedly works best
    # 83-84%

    #merged_classes_nadam_finetuning()
    # lr_start, wd_start = 0.0001, 1e-5 undoubtedly works best
    # 83-86%

    #merged_classes_rmsprop_finetuning()
    # lr_start, wd_start = 0.0001, 1e-5 undoubtedly works best
    # 85-87%

    #merged_classes_finetuning()

    #merged_classes_rmsprop_f_augmentation()
    #merged_classes_rmsprop_s_augmentation()
    #merged_classes_rmsprop()

    full_opt_list = ["AdaBelief", "AdamW", "COCOB",
                     "ConditionalGradient", "CyclicalLearningRate",
                     "DecoupledWeightDecayExtension",
                     "ExponentialCyclicalLearningRate", "LAMB",
                     "LazyAdam", "Lookahead", "MovingAverage", "MultiOptimizer",
                     "NovoGrad", "ProximalAdagrad", "RectifiedAdam", "SGDW",
                     "SWA", "Triangular2CyclicalLearningRate",
                     "TriangularCyclicalLearningRate", "Yogi"]

    opt_list = ["Yogi"]

    # AdaBelief
    # 0.808917224407196 - 0.8216560482978821 - 0.8471337556838989
    # No substantial improvement when finetuning, 0.8216560482978821

    # AdamW
    # 0.8280254602432251 * 2 - 0.8343949317932129
    # Big improvement when finetuning, 0.8853503465652466

    # COCOB
    # 0.8280254602432251 * 2 - 0.8471337556838989
    # Inexplicable decrease in performance when finetuning, 0.03184713423252106

    # ProximalAdagrad - not good
    # 0.6942675113677979 - 0.7197452187538147

    # RectifiedAdam
    # 0.8025477528572083 - 0.808917224407196 Also exact same Acc

    # SGDW - also not good
    # 0.5732483863830566 - 0.5859872698783875

    # Yogi
    # 0.8343949317932129 Also exact same acc...
    # Big improvement when finetuning, 0.8726114630699158

    for opt in opt_list:
        print(f"ALERT\nALERT\nALERT\n{opt}\nALERT\nALERT\nALERT\n")
        merged_classes_tfa_optimizers_finetuning(opt)

    sys.exit()
