from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.backend import sum, flatten


def dice_coe(y_true, y_pred):
    smooth = 1.
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)


def dice_loss(y_true,y_pred):
    return -dice_coe(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coe(y_true, y_pred)
