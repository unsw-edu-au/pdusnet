import math, os, csv
from config import checkpoint_path, csv_log_path, log_path, image_dir_path, dataset_output_path, \
    augmented_dataset_output_path, results_path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def print_section(message):
    print('-' * 50)
    print(message, '...')
    print('-' * 50)


def calculated_steps_per_epoch(num_samples, batch_size):
    return int(math.ceil(1. * num_samples / batch_size))


def generate_path_prefix(model, batch_size, epochs, n_filters, multi_modal, augmented, early_fusion, late_fusion,
                         pe_block, dt, stage=None):
    aug = "augmented" if augmented else "non_augmented"
    stage_str = "_stage_" + str(stage) if stage else ""
    pe_block_str = "pe_block_all" if pe_block else ""
    fusion_type = ""
    if multi_modal:
        fusion_type = "multi_stage_fusion"
    if multi_modal and early_fusion:
        fusion_type = "early_fusion"
    elif multi_modal and late_fusion:
        fusion_type = "late_fusion"
    return model + "_e" + str(epochs) + "_b" + str(batch_size) + "_nf" + str(
        n_filters[0]) + "_" + aug + "_" + pe_block_str + "_" + fusion_type + "_" + dt + stage_str


def generate_model_image_path(path_prefix):
    return os.path.join(image_dir_path, path_prefix + ".png")


def generate_csv_log_path(path_prefix):
    return os.path.join(csv_log_path, path_prefix + ".csv")


def generate_checkpoint_path(path_prefix):
    return os.path.join(checkpoint_path, path_prefix + ".hdf5")


def generate_tensorboard_path(path_prefix):
    return os.path.join(log_path, path_prefix)


def save_test_images(imgs_mask_test, path_prefix):
    imgs_mask_test = np.around(imgs_mask_test, decimals=0)
    imgs_mask_test = (imgs_mask_test * 1.).astype(np.uint8)
    pred_dir = os.path.join('preds', path_prefix)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    pred_list = []
    for k, img in enumerate(imgs_mask_test):
        pred_path = os.path.join(pred_dir, str(k) + ".nii.gz")
        img = np.squeeze(img, axis=-1)
        img_vol = sitk.GetImageFromArray(img)
        sitk.WriteImage(img_vol, pred_path)
        pred_list.append(pred_path)
    return pred_list


def create_dataset_csv(dataset_type, volume_list, augmented):
    folder = augmented_dataset_output_path if augmented else dataset_output_path
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, dataset_type + ".csv")
    with open(file_path, 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['bmode', 'pd', 'label'])
        for row in volume_list:
            csv_out.writerow(row)


def create_results_csv(path_prefix, perf_metrics, pred_ground_truth_tuple_list, total_time):
    if len(perf_metrics) != len(pred_ground_truth_tuple_list):
        return

    if not os.path.exists(results_path):
        os.mkdir(results_path)
    file_path = os.path.join(results_path, path_prefix + ".csv")

    with open(file_path, 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['Prediction', 'Ground Truth', 'DSC', 'Jaccard', 'Hausdorff', 'Mean Surface Distance', 'Total Time (s)'])
        for i in range(len(pred_ground_truth_tuple_list)):
            row = list()
            row.extend(list(pred_ground_truth_tuple_list[i]))
            row.extend(list(perf_metrics[i]))
            row.append(total_time)
            print(tuple(row))
            csv_out.writerow(tuple(row))


def get_record_path(dataset_type, i=None, augmented=False):
    folder = os.path.join(augmented_dataset_output_path, dataset_type) if augmented else os.path.join(
        dataset_output_path, dataset_type)
    if not os.path.exists(folder):
        os.makedirs(folder)
    if i != None:
        return os.path.join(folder, str(i) + '.tfrecords')
    else:
        return folder


def generate_slices(dataset_type, i, bmode, pd, label, augmented=False):
    folder = os.path.join(augmented_dataset_output_path, "images") if augmented else os.path.join(dataset_output_path,
                                                                                                  "images")
    if not os.path.exists(folder):
        os.makedirs(folder)

    sub_dir = os.path.join(folder, dataset_type)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    mask = np.ma.masked_where(label == 0, label)
    y_mins = np.where(mask == np.amin(mask))[1]
    mask_min = y_mins[0]
    mask_max = y_mins[-1]

    plt.imshow(bmode[:, (mask_min + mask_max) // 2, :], cmap="gray")
    plt.imshow(mask[:, (mask_min + mask_max) // 2, :], cmap="autumn", alpha=0.3)

    plt.savefig(os.path.join(sub_dir, str(i) + ".png"))
    plt.close()


def generate_nifti(dataset_type, i, bmode, pd, label, augmented=False):
    folder = os.path.join(augmented_dataset_output_path, "volumes") if augmented else os.path.join(dataset_output_path,
                                                                                                   "volumes")
    if not os.path.exists(folder):
        os.makedirs(folder)

    sub_dir = os.path.join(folder, dataset_type)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    child_dir = os.path.join(sub_dir, str(i))
    if not os.path.exists(child_dir):
        os.makedirs(child_dir)

    bmode_vol = sitk.GetImageFromArray(bmode)
    pd_vol = sitk.GetImageFromArray(pd)
    label_vol = sitk.GetImageFromArray(label)

    sitk.WriteImage(bmode_vol, os.path.join(child_dir, str(i) + "_bmode.nii.gz"))
    sitk.WriteImage(pd_vol, os.path.join(child_dir, str(i) + "_pd.nii.gz"))
    sitk.WriteImage(label_vol, os.path.join(child_dir, str(i) + "_label.nii.gz"))
