from statistics import mean, variance, stdev
from eval_metrics import dc, jc, hd, asd
import SimpleITK as sitk
import numpy as np

def calculate_overlap(seg_file_path, truth_file_path):
    pred_img = sitk.ReadImage(seg_file_path, sitk.sitkUInt8)
    truth_img = sitk.ReadImage(truth_file_path, sitk.sitkUInt8)

    # Post-process
    pred_img = dilate_img(pred_img)
    pred_img = erode_img(pred_img)

    pred_data = sitk.GetArrayFromImage(pred_img)
    truth_data = sitk.GetArrayFromImage(truth_img)
    return [dc(pred_data, truth_data), jc(pred_data, truth_data), hd(pred_data, truth_data), asd(pred_data, truth_data)]

def calculate_vol(seg_file_path):
    img = sitk.ReadImage(seg_file_path)
    seg_data = sitk.GetArrayFromImage(img)
    return np.sum([seg_data == 1]) * np.prod(np.array(img.GetSpacing()))


def erode_img(img_orig):
    erode = sitk.BinaryErodeImageFilter()
    erode.SetKernelRadius(5)
    erode.SetForegroundValue(1)
    erode.SetKernelType(sitk.sitkCross)
    registered_img = erode.Execute(img_orig)
    return sitk.Cast(registered_img, sitk.sitkUInt8)


def dilate_img(img_orig):
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetKernelRadius(5)
    dilate.SetForegroundValue(1)
    registered_img = dilate.Execute(img_orig)
    return sitk.Cast(registered_img, sitk.sitkUInt8)

def compare_segmentations(pred_list, ground_truth_list):
    if len(pred_list) != len(ground_truth_list):
        return
    pred_tuples = list(zip(pred_list, ground_truth_list))
    dc_all = []
    jc_all = []
    hd_all = []
    asd_all = []
    for tuple in pred_tuples:
        eval = calculate_overlap(tuple[0], tuple[1])
        dc_all.append(eval[0])
        jc_all.append(eval[1])
        hd_all.append(eval[2])
        asd_all.append(eval[3])
        # print("Evaluating", tuple)
        # print(eval)

    mean_dc = mean(dc_all)
    mean_jc = mean(jc_all)
    mean_hd = mean(hd_all)
    mean_asd = mean(asd_all)

    print("Average Dice Coefficient: " + str(mean_dc) + " +- " + str(stdev(dc_all, mean_dc)))
    print("Average Jaccard Index: " + str(mean_jc) + " +- " + str(stdev(jc_all, mean_jc)))
    print("Average Hausdorff Distance: " + str(mean_hd) + " +- " + str(stdev(hd_all, mean_hd)))
    print("Average Surface Distance: " + str(mean_asd) + " +- " + str(stdev(asd_all, mean_asd)))

    return list(zip(dc_all, jc_all, hd_all, asd_all))
