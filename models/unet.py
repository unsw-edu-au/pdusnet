from tensorflow.keras.backend import set_image_data_format
from tensorflow.keras.layers import Conv3D, concatenate, Conv3DTranspose
from tensorflow.keras.models import Model
from config import n_filters
from models.helpers import conv_block, fusion_encoder_block


"""
3D U-Net [Ronnerberger et.al, 2015]
Total params (single-modal): 5,647,857 (NUM_FILTERS: 16 -> 256)
Total params (multi modal early-fusion): 5,658,321 (NUM_FILTERS: 16 -> 256)
Total params (multi modal early-fusion PE_BLOCK): 5,768,633 (NUM_FILTERS: 16 -> 256)
Total params (multi modal multi-stage-fusion): 8,290,113 (NUM_FILTERS: 16 -> 256)
Total params (multi modal multi-stage-fusion PE_BLOCK): 8,422,393 (NUM_FILTERS: 16 -> 256)
Total params (multi modal late-fusion): 11,295,697 (NUM_FILTERS: 16 -> 256)
Total params (multi modal late-fusion PE_BLOCK): 11,516,017 (NUM_FILTERS: 16 -> 256)
"""


def unet(multi_modal, early_fusion, project_excite, inputs_bmode, inputs_pd, cascade=False):
    set_image_data_format('channels_last')

    multi_stage_fusion = multi_modal and not early_fusion

    # Input
    if multi_modal and early_fusion:
        conv, pool, pool_b = fusion_encoder_block(n_filters[0] // 2, multi_modal, project_excite, inputs_bmode, inputs_pd)
        conv1, pool1, pool1_b = fusion_encoder_block(n_filters[0], multi_stage_fusion, project_excite, conv, None)
    else:
        conv1, pool1, pool1_b = fusion_encoder_block(n_filters[0], multi_stage_fusion, project_excite, inputs_bmode,
                                                       inputs_pd)

    # Fused Encode Block 2
    conv2, pool2, pool2_b = fusion_encoder_block(n_filters[1], multi_stage_fusion, project_excite, pool1, pool1_b)

    # Fused Encode Block 3
    conv3, pool3, pool3_b = fusion_encoder_block(n_filters[2], multi_stage_fusion, project_excite, pool2, pool2_b)

    # Fused Encode Block 4
    conv4, pool4, pool4_b = fusion_encoder_block(n_filters[3], multi_stage_fusion, project_excite, pool3, pool3_b)

    # Bottom stage
    conv5 = conv_block(pool4, n_filters[4], project_excite)

    # Decode stage 1
    up6 = concatenate(
        [Conv3DTranspose(n_filters[3], 2, strides=(2, 2, 2), padding='same')(conv5), conv4], axis=-1)
    conv6 = conv_block(up6, n_filters[3], project_excite)

    # Decode stage 2
    up7 = concatenate(
        [Conv3DTranspose(n_filters[2], 2, strides=(2, 2, 2), padding='same')(conv6), conv3], axis=-1)
    conv7 = conv_block(up7, n_filters[2], project_excite)

    # Decode stage 3
    up8 = concatenate(
        [Conv3DTranspose(n_filters[1], 2, strides=(2, 2, 2), padding='same')(conv7), conv2], axis=-1)
    conv8 = conv_block(up8, n_filters[1], project_excite)

    # Decode stage 4
    up9 = concatenate(
        [Conv3DTranspose(n_filters[0], 2, strides=(2, 2, 2), padding='same')(conv8), conv1], axis=-1)
    conv9 = conv_block(up9, n_filters[0], project_excite)

    if cascade:
        return conv9

    # Output
    conv10 = Conv3D(1, 1, activation='sigmoid', padding='same')(conv9)

    if multi_modal:
        model = Model(inputs=[inputs_bmode, inputs_pd], outputs=[conv10])
    else:
        model = Model(inputs=[inputs_bmode], outputs=[conv10])

    return model
