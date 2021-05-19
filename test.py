import argparse
import os
import pandas as pd
from dataset import load_dataset
from models.unet import unet
from models.unetpp import unetpp
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Average, Conv3D
from tensorflow.keras.models import Model
from config import vol_x, vol_y, vol_z, n_channels, train_samples_total, train_samples, learning_rate, beta_1, beta_2, \
    epsilon, decay, validation_samples, n_filters, dataset_output_path
from metrics import dice_coe, dice_loss
from helpers import print_section, calculated_steps_per_epoch, generate_checkpoint_path, save_test_images, \
    generate_model_image_path, create_results_csv, generate_path_prefix
from callbacks import get_callbacks, TimeHistory
from postprocess import compare_segmentations
from datetime import datetime
from models.helpers import handle_input_fusion


def get_model(model_type, multi_modal, perform_early_fusion, pe_block, inputA, inputB, cascade=False):
    if model_type == "unet":
        return unet(multi_modal, perform_early_fusion, pe_block, inputA, inputB, cascade)
    elif model_type == "unet++":
        return unetpp(multi_modal, perform_early_fusion, pe_block, inputA, inputB, cascade)


def create_model(model_type, dataset, validation_dataset, test_dataset, callbacks, batch_size, num_epochs, multi_modal,
                 augmented, perform_test_only, perform_early_fusion, perform_late_fusion, pe_block, path_prefix):
    time_callback = TimeHistory()
    input_shape = (vol_x, vol_y, vol_z, n_channels)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        print_section("Creating and compiling " + model_type + " model")

        inputs_bmode, inputs_pd = handle_input_fusion(multi_modal, input_shape, batch_size)

        if multi_modal and perform_late_fusion:
            # Treat as separate single modal models
            model_bmode = get_model(model_type, False, False, pe_block, inputs_bmode, None, True)
            model_pd = get_model(model_type, False, False, pe_block, inputs_pd, None, True)

            output_conv = Average()([model_bmode, model_pd])
            output = Conv3D(1, 1, activation='sigmoid', padding='same')(output_conv)

            model = Model(inputs=[inputs_bmode, inputs_pd], outputs=[output])
        else:
            model = get_model(model_type, multi_modal, perform_early_fusion, pe_block, inputs_bmode, inputs_pd)

        model.compile(
            optimizer=Adam(lr=learning_rate,
                           beta_1=beta_1,
                           beta_2=beta_2,
                           epsilon=epsilon,
                           decay=decay),
            loss=dice_loss,
            metrics=["binary_crossentropy", dice_coe])

        print_section("Generating model summary")
        model.summary()

        print_section("Generate model graph image")
        plot_model(
            model,
            to_file=generate_model_image_path(path_prefix),
            show_shapes=True, show_layer_names=True,
            rankdir='LR', expand_nested=False, dpi=192
        )

        print_section("Training model")
        num_train_samples = train_samples_total if augmented else train_samples
        model.fit(dataset,
                  epochs=num_epochs,
                  verbose=1,
                  shuffle=True,
                  steps_per_epoch=calculated_steps_per_epoch(num_train_samples, batch_size),
                  validation_data=validation_dataset,
                  validation_steps=calculated_steps_per_epoch(validation_samples, batch_size),
                  validation_freq=1,
                  callbacks=[time_callback] + callbacks)

        print_section("Loading model weights")
        model.load_weights(
            generate_checkpoint_path(path_prefix))
        t_time = sum(time_callback.times)

        print_section('Testing model')
        imgs_mask_test = model.predict(test_dataset, batch_size=batch_size, verbose=1)

        print_section('Saving predictions')
        pred_label_list = save_test_images(imgs_mask_test, path_prefix)

        print_section('Evaluating against ground truth')
        test_df = pd.read_csv(os.path.join(dataset_output_path, 'test.csv'))
        ground_truth_label_list = test_df['label'].values.tolist()
        perf_metrics = compare_segmentations(pred_label_list, ground_truth_label_list)

        print_section('Saving results')
        create_results_csv(path_prefix, perf_metrics, list(zip(pred_label_list, ground_truth_label_list)),
                           t_time)

        print_section("Printing stats")
        print("Each epoch time", time_callback.times)
        print("Total Time Taken (s)", sum(time_callback.times))


def test_model(model_type, test_dataset, batch_size, num_epochs, multi_modal, augmented,
               perform_test_only, perform_early_fusion, perform_late_fusion, pe_block, path_prefix, weights):
    time_callback = TimeHistory()
    input_shape = (vol_x, vol_y, vol_z, n_channels)


    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        print_section("Creating and compiling " + model_type + " model")

        inputs_bmode, inputs_pd = handle_input_fusion(multi_modal, input_shape, batch_size)

        if multi_modal and perform_late_fusion:
            # Treat as separate single modal models
            model_bmode = get_model(model_type, False, False, pe_block, inputs_bmode, None, True)
            model_pd = get_model(model_type, False, False, pe_block, inputs_pd, None, True)

            output_conv = Average()([model_bmode, model_pd])
            output = Conv3D(1, 1, activation='sigmoid', padding='same')(output_conv)

            model = Model(inputs=[inputs_bmode, inputs_pd], outputs=[output])
        else:
            model = get_model(model_type, multi_modal, perform_early_fusion, pe_block, inputs_bmode, inputs_pd)

        model.compile(
            optimizer=Adam(lr=learning_rate,
                           beta_1=beta_1,
                           beta_2=beta_2,
                           epsilon=epsilon,
                           decay=decay),
            loss=dice_loss,
            metrics=["binary_crossentropy", dice_coe])

        print_section("Generating model summary")
        model.summary()


        if not perform_test_only:
            print_section("Generate model graph image")
            plot_model(
                model,
                to_file=generate_model_image_path(path_prefix),
                show_shapes=True, show_layer_names=True,
                rankdir='LR', expand_nested=False, dpi=192
            )

            print_section("Training model")
            num_train_samples = train_samples_total if augmented else train_samples
            model.fit(dataset,
                      epochs=num_epochs,
                      verbose=1,
                      shuffle=True,
                      steps_per_epoch=calculated_steps_per_epoch(num_train_samples, batch_size),
                      validation_data=validation_dataset,
                      validation_steps=calculated_steps_per_epoch(validation_samples, batch_size),
                      validation_freq=1,
                      callbacks=[time_callback] + callbacks)

        print_section("Loading model weights")
        model.load_weights(
            generate_checkpoint_path(weights))
        t_time = 0.0

        print_section('Testing model')
        imgs_mask_test = model.predict(test_dataset, batch_size=batch_size, verbose=1)

        print_section('Saving predictions')
        pred_label_list = save_test_images(imgs_mask_test, path_prefix)

        print_section('Evaluating against ground truth')
        test_df = pd.read_csv(os.path.join(dataset_output_path, 'test.csv'))
        ground_truth_label_list = test_df['label'].values.tolist()
        perf_metrics = compare_segmentations(pred_label_list, ground_truth_label_list)

        print_section('Saving results')
        create_results_csv(path_prefix, perf_metrics, list(zip(pred_label_list, ground_truth_label_list)),
                           t_time)


if __name__ == "__main__":
    print_section("Setting configuration options")
    parser = argparse.ArgumentParser(description="Process configuration")
    parser.add_argument("--model", required=True, type=str, action="store")
    parser.add_argument("--batch_size", required=True, type=int, action="store")
    parser.add_argument("--num_epochs", required=True, type=int, action="store")
    parser.add_argument("--multi_modal", required=False, type=bool, action="store")
    parser.add_argument("--augment", required=False, type=bool, action="store")
    parser.add_argument("--test_only", required=False, type=bool, action="store")
    parser.add_argument("--old_weights", required=True, type=str, action="store")
    parser.add_argument("--early_fusion", required=False, type=bool, action="store")
    parser.add_argument("--late_fusion", required=False, type=bool, action="store")
    parser.add_argument("--pe_block", required=False, type=bool, action="store")
    args = parser.parse_args()

    augment = False
    multi_modal = False
    test_only = False
    early_fusion = False
    late_fusion = False
    pe_block = False

    if args.multi_modal:
        multi_modal = True
    if args.augment:
        augment = True
    if args.test_only:
        test_only = True
    if args.early_fusion:
        early_fusion = True
    if args.late_fusion:
        late_fusion = True
    if args.pe_block:
        pe_block = True

    print_section("Setting GPU Settings")
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    if not args.test_only:
        print_section("Loading training, validation and test datasets")
    else:
        print_section("****** test only case ******")
        print_section("Loading test dataset")
    modal_type = "multi_modal" if multi_modal else "bmode"

    if not args.test_only:
        train_dataset = load_dataset("train", args.batch_size, args.num_epochs, modal_type, augment)
        validation_dataset = load_dataset("validation", args.batch_size, args.num_epochs, modal_type, augment)
        test_dataset = load_dataset("test", args.batch_size, args.num_epochs, modal_type, augment)

        dt = str(datetime.today().strftime('%Y-%m-%d___%H-%M-%S'))
        path_prefix = generate_path_prefix(args.model, args.batch_size, args.num_epochs, n_filters, multi_modal,
                                           augment,
                                           early_fusion, late_fusion, pe_block, dt)

        print_section("Creating model callbacks")
        callbacks = get_callbacks(path_prefix)

        print_section("Creating model on multiple GPUs")
        create_model(args.model, train_dataset, validation_dataset, test_dataset,
                     callbacks, args.batch_size, args.num_epochs, multi_modal, augment, test_only, early_fusion,
                     late_fusion,
                     pe_block, path_prefix)
    else:
        test_dataset = load_dataset("test", args.batch_size, args.num_epochs, modal_type, augment)

        dt = str(datetime.today().strftime('%Y-%m-%d___%H-%M-%S'))
        path_prefix = generate_path_prefix(args.model, args.batch_size, args.num_epochs, n_filters, multi_modal,
                                           augment,
                                           early_fusion, late_fusion, pe_block, dt)

        print_section("Model will be using saved weight parameters: {}{}".format(args.old_weights, '.hdf5'))
        #print_section("Creating model callbacks")
        #callbacks = get_callbacks(path_prefix)

        #print_section("Creating model on multiple GPUs")
        test_model(args.model, test_dataset, args.batch_size, args.num_epochs, multi_modal, augment, test_only, early_fusion,
                   late_fusion, pe_block, path_prefix, args.old_weights)

