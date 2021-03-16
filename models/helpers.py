from tensorflow.keras.layers import Conv3D, Dropout, Input, BatchNormalization, MaxPooling3D, concatenate, Multiply, Add
from tensorflow import reshape
from config import dropout_rate

from tensorflow_addons.layers import AdaptiveAveragePooling3D

"""
Standard ConvBlock in UNet.
Includes BatchNormalisation and Dropout.
"""
def conv_block(input, n_filters, project_excite=False, k_size=3):
    x = Conv3D(n_filters, k_size, activation='relu', padding='same')(input)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv3D(n_filters, k_size, activation='relu', padding='same')(x)

    if project_excite:
       x = project_excite_block(x, n_filters)

    return x


def project_excite_block(input, num_channels, reduction_ratio=2):
    """
    keras implementation of 3D extensions of Project & Excite block based on pytorch implementation provided by Anne-Marie Richman  (https://github.com/arickm) available at https://github.com/ai-med/squeeze_and_excitation/blob/master/squeeze_and_excitation/squeeze_and_excitation_3D.py

    :param input: X shape = (batch_size, D,H,W, num_channels)
    :param num_channels: No of input channels
    :param reduction_ratio: By how much should the num_channels should be reduced
    """
    num_channels_reduced = num_channels // reduction_ratio

    batch_size, D, H, W, num_channels = input.shape

    # Project:
    # Average along channels and different axes
    squeeze_tensor_w = AdaptiveAveragePooling3D((1, 1, W), data_format="channels_last")(input)
    squeeze_tensor_h = AdaptiveAveragePooling3D((1, H, 1), data_format="channels_last")(input)
    squeeze_tensor_d = AdaptiveAveragePooling3D((D, 1, 1), data_format="channels_last")(input)

    # tile tensors to original size and add:
    final_squeeze_tensor = Add()([reshape(squeeze_tensor_w, [batch_size, 1, 1, W, num_channels]),
    reshape(squeeze_tensor_h, [batch_size, 1, H, 1, num_channels]),
    reshape(squeeze_tensor_d, [batch_size, D, 1, 1, num_channels])])

    # Excitation:
    x = Conv3D(filters=num_channels_reduced, kernel_size=1, activation="relu")(final_squeeze_tensor)
    x = Conv3D(filters=num_channels, kernel_size=1, activation="sigmoid")(x)

    x = Multiply()([input, x])
    return x

"""
Handle multi modal input and return both modalities if running multi-modal config.
"""
def handle_input_fusion(multi_modal, input_shape, batch_size):
    input_bmode = Input(shape=input_shape, batch_size=batch_size)
    input_pd = None
    if multi_modal:
        input_pd = Input(shape=input_shape, batch_size=batch_size)
    return input_bmode, input_pd


"""
Multi-modal fusion encoder block. Performs a concatenation at each encoder stage.
"""
def fusion_encoder_block(n_filters, multi_modal, project_excite, inputsA, inputsB=None):
    conv = conv_block(inputsA, n_filters, project_excite)
    pool_b = None

    if multi_modal:
        conv_b = conv_block(inputsB, n_filters, project_excite)
        pool_b = MaxPooling3D(pool_size=(2, 2, 2))(conv_b)
        conv = concatenate([conv, conv_b], axis=-1)

    pool = MaxPooling3D(pool_size=(2, 2, 2))(conv)
    return conv, pool, pool_b
