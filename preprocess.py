import SimpleITK as sitk
import numpy as np
from scipy.spatial.transform import Rotation as R
from dltk.io.preprocessing import whitening

"""
img: simpleitk input image
angle: radian angle to rotate around the z axis
size: voxel size for resampled data
"""


def rotate_image(img, angle, size=[64, 64, 64], is_label=False):
    rotation_center = (0, 0, 0)
    rotation = sitk.VersorTransform(R.from_euler('Z', angle).as_quat(), rotation_center)

    rigid_versor = sitk.VersorRigid3DTransform()
    rigid_versor.SetRotation(rotation.GetVersor())
    rigid_versor.SetCenter(rotation_center)

    out_origin, out_size, out_spacing = get_output_parameters(img, rigid_versor, size)

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetTransform(rigid_versor)

    if is_label:
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample_filter.SetInterpolator(sitk.sitkBSpline)

    resample_filter.SetSize(size)
    resample_filter.SetOutputOrigin(out_origin)
    resample_filter.SetOutputSpacing(out_spacing)

    resample_filter.SetOutputDirection(img.GetDirection())
    if is_label:
        resample_filter.SetOutputPixelType(sitk.sitkUInt8)
    else:
        resample_filter.SetOutputPixelType(sitk.sitkFloat32)
    resample_filter.SetDefaultPixelValue(0.0)

    output_img = resample_filter.Execute(img)

    if is_label:
        return sitk.GetArrayFromImage(output_img)
    else:
        return whitening(sitk.GetArrayFromImage(output_img))


"""
img: simpleitk input image
axes: 1 for no flip, -1 for a flip of array of (int, 3) 
size: voxel size for resampled data
"""


def flip_image(img, axes=[1, -1, 1], size=[64, 64, 64], is_label=False):
    out_origin, out_size, out_spacing = get_output_parameters(img, sitk.Transform(3, sitk.sitkIdentity), size)

    rotation_center = (0, 0, 0)
    rotation = sitk.VersorTransform(np.array([0., 0., 0., 1.]), rotation_center)

    rigid_versor = sitk.VersorRigid3DTransform()
    rigid_versor.SetRotation(rotation.GetVersor())
    rigid_versor.SetCenter(rotation_center)

    rigid_versor.SetMatrix([axes[0], 0, 0, 0, axes[1], 0, 0, 0, axes[2]])

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetTransform(rigid_versor)
    if is_label:
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample_filter.SetInterpolator(sitk.sitkBSpline)

    resample_filter.SetSize(size)
    resample_filter.SetOutputOrigin(img.GetOrigin())
    resample_filter.SetOutputSpacing(out_spacing)

    resample_filter.SetOutputDirection(img.GetDirection())
    if is_label:
        resample_filter.SetOutputPixelType(sitk.sitkUInt8)
    else:
        resample_filter.SetOutputPixelType(sitk.sitkFloat32)
    resample_filter.SetDefaultPixelValue(0.0)

    output_img = resample_filter.Execute(img)

    if is_label:
        return sitk.GetArrayFromImage(output_img)
    else:
        return whitening(sitk.GetArrayFromImage(output_img))


"""
given an image and a transform, provide the transformed bounds 

returns: output_origin, size and spacing based on a given transform    

output_origin : the origin of the image given a transform
output_spacing: the spacing given the size input set at 64 voxels as a default.
output_size   : the size given the input image spacing

"""


def get_output_parameters(image, transform, size=[64, 64, 64]):
    # origin and maximum of the transformed image.
    x0, y0, z0 = image.GetOrigin()
    x1, y1, z1 = image.TransformIndexToPhysicalPoint(image.GetSize())

    trans_pts = []
    for x in (x0, x1):
        for y in (y0, y1):
            for z in (z0, z1):
                trans_pt = transform.GetInverse().TransformPoint((x, y, z))
                trans_pts.append(trans_pt)

    min_arr = np.array(trans_pts).min(axis=0)
    max_arr = np.array(trans_pts).max(axis=0)

    output_origin = min_arr
    output_size = np.round(((max_arr - min_arr) / image.GetSpacing())).astype(int)
    output_spacing = ((max_arr - min_arr) / size).astype(float)
    # print(output_size)
    return output_origin, output_size.tolist(), output_spacing.tolist()


"""
Pre-process and augment data (if defined)
Returns a list of all pre-processed/augmented data volumes as tuples
"""


def preprocess(volume_list, augment_data=False):
    preprocessed_volumes = []
    for volume_tuple in volume_list:
        bmode, pd, label = load_volumes(volume_tuple)
        preprocessed_volumes.append((bmode, pd, label))
        if augment_data:
            preprocessed_volumes += augment(volume_tuple)
    return preprocessed_volumes


"""
Augments tuple of volumes (BMode, PD, Label) and returns a list of all augmented volumes (as tuples)
TO-DO: Currently just returns an array of the same volumes as a tuple array but should insert logic here
NOTE: It should not return the input volumes in the return array since its already added to the full volume list
      ONLY append the augmentations
"""


def augment(volume_tuple):
    bmode, pd, label = sitk.ReadImage(volume_tuple[0], sitk.sitkFloat32), sitk.ReadImage(volume_tuple[1],  sitk.sitkFloat32), sitk.ReadImage(volume_tuple[2],  sitk.sitkUInt8)
    augmented_tuples = []
    size = [64, 64, 64]

    # initial go - -20 to +20 degrees (5 deg increment) no zero
    # angles = array([-0.34906585, -0.26179939, -0.17453293, -0.08726646,  0.08726646, 0.17453293,  0.26179939,  0.34906585])
    # now with more angles (-40 + 40) in 4 degree increments... to get to ~27 we augmented with prior
    angles = np.linspace(-np.pi / 18, np.pi / 18, 11)
    angles = angles[angles != 0]

    for rad in angles:
        augmented_tuples.append((rotate_image(bmode, rad, size), rotate_image(pd, rad, size),
                                 rotate_image(label, rad, size, True)))

    axes_flip = [-1, 1, 1], [1, -1, 1], [-1, -1, 1]
    for a in axes_flip:
        augmented_tuples.append(
            (flip_image(bmode, a, size), flip_image(pd, a, size), flip_image(label, a, size, True)))

    return augmented_tuples


def load_volumes(volume_tuple):
    bmode, pd, label = sitk.ReadImage(volume_tuple[0], sitk.sitkFloat32), sitk.ReadImage(volume_tuple[1], sitk.sitkFloat32), sitk.ReadImage(volume_tuple[2], sitk.sitkUInt8)
    bmode_vol = sitk.GetArrayFromImage(bmode)
    pd_vol = sitk.GetArrayFromImage(pd)
    label_vol = sitk.GetArrayFromImage(label)
    return whitening(bmode_vol), whitening(pd_vol), label_vol
