from tensorflow.keras.backend import set_image_data_format
from tensorflow.keras.layers import Conv3D, concatenate, Conv3DTranspose, Average
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from config import deep_supervision, n_filters
from models.helpers import conv_block, fusion_encoder_block


"""
UNet++ [Zhou et.al, 2018]
Total params (single-modal): 6,315,905 (NUM_FILTERS: 16 -> 256)
Total params (multi-modal early-fusion): 6,330,068 (NUM_FILTERS: 16 -> 256)
Total params (multi-modal early-fusion PE_BLOCK): 6,447,556 (NUM_FILTERS: 16 -> 256)
Total params (multi modal multi-stage-fusion): 9,047,876 (NUM_FILTERS: 16 -> 256)
Total params (multi modal multi-stage-fusion PE_BLOCK):  9,187,332 (NUM_FILTERS: 16 -> 256)
Total params (multi modal late-fusion): 12,639,089 (NUM_FILTERS: 16 -> 256)
Total params (multi modal late-fusion PE_BLOCK): 12,873,761 (NUM_FILTERS: 16 -> 256)
"""


def unetpp(multi_modal, early_fusion, project_excite, inputs_bmode, inputs_pd, cascade=False):
    set_image_data_format('channels_last')

    multi_stage_fusion = multi_modal and not early_fusion

    # Input
    if multi_modal and early_fusion:
        conv, pool, pool_b = fusion_encoder_block(n_filters[0] // 2, multi_modal, project_excite, inputs_bmode, inputs_pd)
        conv0_0, pool0, pool0_b = fusion_encoder_block(n_filters[0], multi_stage_fusion, project_excite, conv, None)
    else:
        conv0_0, pool0, pool0_b = fusion_encoder_block(n_filters[0], multi_stage_fusion, project_excite, inputs_bmode,
                                                     inputs_pd)

    # Fused Encode Block 1
    conv1_0, pool1, pool1_b = fusion_encoder_block(n_filters[1], multi_stage_fusion, project_excite, pool0, pool0_b)

    # Dense Conv Block 0_1
    up0_1 = concatenate(
        [Conv3DTranspose(n_filters[0], 2, strides=(2, 2, 2), padding='same')(conv1_0), conv0_0], axis=-1)
    conv0_1 = conv_block(up0_1, n_filters[0])

    # Fused Encode Block 2
    conv2_0, pool2, pool2_b = fusion_encoder_block(n_filters[2], multi_stage_fusion, project_excite, pool1, pool1_b)

    # Dense Conv Block 1_1
    up1_1 = concatenate(
        [Conv3DTranspose(n_filters[1], 2, strides=(2, 2, 2), padding='same')(conv2_0), conv1_0], axis=-1)
    conv1_1 = conv_block(up1_1, n_filters[1])

    # Dense Conv Block 0_2
    up0_2 = concatenate(
        [Conv3DTranspose(n_filters[0], 2, strides=(2, 2, 2), padding='same')(conv1_1), conv0_1], axis=-1)
    conv0_2 = conv_block(up0_2, n_filters[0])

    # Fused Encode Block 3
    conv3_0, pool3, pool3_b = fusion_encoder_block(n_filters[3], multi_stage_fusion, project_excite, pool2, pool2_b)

    # Dense Conv Block 2_1
    up2_1 = concatenate(
        [Conv3DTranspose(n_filters[2], 2, strides=(2, 2, 2), padding='same')(conv3_0), conv2_0], axis=-1)
    conv2_1 = conv_block(up2_1, n_filters[2])

    # Dense Conv Block 1_2
    up1_2 = concatenate(
        [Conv3DTranspose(n_filters[1], 2, strides=(2, 2, 2), padding='same')(conv2_1), conv1_1], axis=-1)
    conv1_2 = conv_block(up1_2, n_filters[1])

    # Dense Conv Block 0_3
    up0_3 = concatenate(
        [Conv3DTranspose(n_filters[0], 2, strides=(2, 2, 2), padding='same')(conv1_2), conv0_2], axis=-1)
    conv0_3 = conv_block(up0_3, n_filters[0])

    # Bottom Conv Block
    conv4_0 = conv_block(pool3, n_filters[4], project_excite)

    # Decode stage 1
    up3_1 = concatenate(
        [Conv3DTranspose(n_filters[3], 2, strides=(2, 2, 2), padding='same')(conv4_0), conv3_0], axis=-1)
    conv3_1 = conv_block(up3_1, n_filters[3], project_excite)

    # Decode stage 2
    up2_2 = concatenate(
        [Conv3DTranspose(n_filters[2], 2, strides=(2, 2, 2), padding='same')(conv3_1), conv2_1], axis=-1)
    conv2_2 = conv_block(up2_2, n_filters[2], project_excite)

    # Decode stage 3
    up1_3 = concatenate(
        [Conv3DTranspose(n_filters[1], 2, strides=(2, 2, 2), padding='same')(conv2_2), conv1_2], axis=-1)
    conv1_3 = conv_block(up1_3, n_filters[1], project_excite)

    # Decode stage 4
    up0_4 = concatenate(
        [Conv3DTranspose(n_filters[0], 2, strides=(2, 2, 2), padding='same')(conv1_3), conv0_3], axis=-1)
    conv0_4 = conv_block(up0_4, n_filters[0], project_excite)

    if cascade:
        return Average()([conv0_1, conv0_2, conv0_3, conv0_4])

    # Average all segmentation branches from top row
    output_1 = Conv3D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))(conv0_1)
    output_2 = Conv3D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))(conv0_2)
    output_3 = Conv3D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))(conv0_3)
    output_4 = Conv3D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))(conv0_4)

    # If training with deep_supervision, average all skip connection convolution layers in the top row
    if deep_supervision:
        output = Average()([output_1, output_2, output_3, output_4])
    else:
        output = output_4

    if multi_modal:
        model = Model(inputs=[inputs_bmode, inputs_pd], outputs=[output])
    else:
        model = Model(inputs=[inputs_bmode], outputs=[output])

    return model