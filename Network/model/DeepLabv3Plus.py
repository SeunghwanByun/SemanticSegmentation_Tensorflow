import cv2 as cv
import tensorflow as tf
import numpy as np
import scipy.misc
from datetime import timedelta
import os
import time
from glob import glob

from utils_for_LiDAR2RGBImage import *

# 학습에 필요한 설정값들을 지정
KEEP_PROB = 0.1
MAX_ITERATION = 1e-2
NUM_OF_CLASSESS = 2
IMAGE_SHAPE_KITTI = (128, 480)
# IMAGE_SHAPE_KITTI = (160, 576)
# IMAGE_SHAPE_KITTI = (192, 704)
# IMAGE_SHAPE_KITTI = (384, 1280)
# IMAGE_SHAPE_KITTI = (713, 1280)
BATCH_SIZE = 1
EPOCHS = 300
LEARNING_RATE = 1e-4

DATA_DIR = "data_road"

def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optinally add activation after BN
        Implements right "SAME" padding for even kernel sizes

    :param x: input tensor
    :param filters: num of filters in pointwise convolution
    :param prefix: prefix before name
    :param stride: stride at depthwise conv
    :param kernel_size: kernel size for depthwise convolution
    :param rate: atrous rate for depthwise convolution
    :param depth_activation: flag to use activation between depthwise & pointwise convs
    :param epsilon: epsilon to use in BN layer
    :return:
    """
    if stride == 1:
        depth_padding = 'SAME'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = Zero_Padding(x, (pad_beg, pad_end))
        depth_padding = 'VALID'

    if not depth_activation:
        x = ReLU(x)
    x = tf.nn.depthwise_conv2d(x, filters, strides=stride, rate=rate, padding=depth_padding, name=prefix + '_depthwise')
    x = Batch_Normalization(x)

    if depth_activation:
        x = ReLU(x)

    x = Conv2D_Block(x, filters, filter_height=1, filter_width=1, stride=1, padding='SAME', batch_normalization=True,
                     relu=True, name=prefix+"_pointwise_BN")

    return x

def Xception_block(x, depth_list, prefix, skip_connection_type, stride, rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
    :param x: input tensor
    :param depth_list: number of filters in each SepConv layer. len(depth_list) == 3
    :param prefix: prefix before name
    :param skip_connection_type: one of {'conv', 'sum', 'none'}
    :param stride: stride at last depthwise conv
    :param rate: atrous rate for depthwise convolution
    :param depth_activation: flag to use activation between depthwise & pointwise convs
    :param return_skip: flag to return additional tensor after 2 SepConvs for decoder
    :return:
    """
    residual = x
    for i in range(3):
        residual = SepConv_BN(residual, depth_list[i], prefix+'separable_conv{}'.format(i+1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = Conv2D_Block(x, depth_list[-1], filter_height=1, filter_width=1, stride=stride, padding='SAME',
                                batch_normalization=True, name=prefix+"_shortcut")
        outputs = tf.add(shortcut, residual, name=prefix+'add_conv')
    elif skip_connection_type == 'sum':
        outputs = tf.add(residual, x, name=prefix+'add_sum')
    elif skip_connection_type == 'none':
        outputs = residual

    if return_skip:
        return outputs, skip
    else:
        return outputs

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _inverted_res_block(x, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = x.shape[-1].value
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)

    inputs = x
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        x = Conv2D_Block(x, expansion * in_channels, filter_height=1, filter_width=1, stride=1, padding='SAME',
                         batch_normalization=True, relu=True, name=prefix+'_expand')
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    x = tf.nn.depthwise_conv2d(x, filter=[3, 3, x.shape[-1], ], strides=stride, padding='SAME', rate=rate, name=prefix+'depthwise')
    x = ReLU(x)

    ## Project
    x = Conv2D_Block(x, pointwise_filters, filter_height=1, filter_width=1, stride=1, padding='SAME', batch_normalization=True, name=prefix+'project')

    if skip_connection:
        return tf.add(x, inputs)

    return x


def DeepLabV3Plus(x, input_shape, keep_prob, num_classess, backbone='mobilenetv2', out_stride=16, alpha=0.1, activation=None):
    """
        Instantiates the DeepLabV3+ architecture.
    :param x:
    :param input_shape:
    :param num_classess:
    :param out_stride:
    :param alpha:
    :param activation:
    :return:
    """
    if backbone == 'xception':
        if out_stride == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = (2, 4)
            atrous_rate = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rate = (6, 12, 18)

        entry_conv1 = Conv2D_Block(x, 32, filter_height=3, filter_width=3, stride=2, batch_normalization=True, relu=True, name='entry_flow_conv1_1_BN')

        entry_conv1_2 = Conv2D_Block(entry_conv1, 64, filter_height=3, filter_width=3, stride=1, batch_normalization=True, relu=True, name='entry_flow_conv1_2_BN')

        xblock_entry_1 = Xception_block(entry_conv1_2, [128, 128, 128], 'entry_flow_block1', skip_connection_type='conv', stride=2, depth_activation=False)

        xblock_entry_2, skip1 = Xception_block(xblock_entry_1, [256, 256, 256], 'entry_flow_block2', skip_connection_type='conv', stride=2, depth_activation=False, return_skip=True)

        xblock_entry_3 = Xception_block(xblock_entry_2, [728, 728, 728], 'entry_flow_block3', skip_connection_type='conv', stride=entry_block3_stride, depth_activation=False)

        for i in range(16):
            xblock_mid = Xception_block(xblock_entry_3, [728, 728, 728], 'middle_flow_unit_{}'.format(i+1), skip_connection_type='sum', stride=1, rate=middle_block_rate, depth_activation=False)

        xblock_exit_1 = Xception_block(xblock_mid, [728, 728, 1024], 'exit_flow_block1', skip_connection_type='conv', stride=1, rate=exit_block_rates[0], depth_activation=False)

        extracted_feature = Xception_block(xblock_exit_1, [1536, 1536, 2048], 'exit_flow_block2', skip_connection_type='none', stride=1, rate=exit_block_rates[1], depth_activation=True)

    else:
        out_stride = 8
        first_block_filters = _make_divisible(32 * alpha, 8)
        x = Conv2D_Block(x, first_block_filters, filter_height=3, filter_width=3, stride=2, padding='SAME', batch_normalization=True, relu=True, name='Conv')

        x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0, skip_connection=False)

        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1, skip_connection=False)

        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2, skip_connection=True)

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=3, skip_connection=False)

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4, skip_connection=True)

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5, skip_connection=True)

        # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=6, skip_connection=False)

        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=7, skip_connection=True)

        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=8, skip_connection=True)

        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=9, skip_connection=True)


        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=10, skip_connection=False)

        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=11, skip_connection=True)

        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=12, skip_connection=True)


        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2, expansion=6, block_id=13, skip_connection=False)

        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4, expansion=6, block_id=14, skip_connection=True)

        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4, expansion=6, block_id=15, skip_connection=True)


        extracted_feature = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4, expansion=6, block_id=16, skip_connection=False)


    # end of feature extractor

    # branching for Atrous Spatia Pyramid Pooling

    # Image Featrue branch
    b4 = Global_Avg_Pool(extracted_feature, name='b4')

    # from (b_size, channels)->(b_size, 1, 1, channels)
    # b4 = Lambda(tf.expand_dims(x, dim=1))(b4)
    # b4 = Lambda(tf.expand_dims(x, dim=1))(b4)
    b4 = tf.expand_dims(b4, dim=1)
    b4 = tf.expand_dims(b4, dim=1)

    b4 = Conv2D_Block(b4, 256, filter_height=1, filter_width=1, stride=1, padding='SAME', batch_normalization=True, relu=True, name='image_pooling')

    # Upsampling. have to use compat because of the option align_corners
    size_before = tf.shape(extracted_feature)
    b4 = Resize_Bilinear(b4, size_before[1:3], name='Upsampling')

    # simple 1x1
    b0 = Conv2D_Block(x, 256, filter_height=1, filter_width=1, stride=1, padding='SAME', batch_normalization=True, relu=True, name='aspp0')

    # there are only 2 branches in mobilenetV2. not sure why
    if backbone == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1', rate=atrous_rate[0], depth_activation=True, epsilon=1e-5)

        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2', rate=atrous_rate[1], depth_activation=True, epsilon=1e-5)

        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3', rate=atrous_rate[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        concatenated = Concat([b4, b0, b1, b2, b3], axis=-1, name="concatenation")
    else:
        concatenated = Concat([b4, b0], axis=-1, name="concatenation")


    reduce_channels = Conv2D_Block(concatenated, 256, filter_height=1, filter_width=1, stride=1, padding='SAME', batch_normalization=True, relu=True, name='concat_projection')
    reduce_channels = Dropout(reduce_channels, keep_prob=keep_prob)

    # DeepLab v3+ Decoder

    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block
        size_before2 = tf.shape(reduce_channels)
        dec_up1 = Resize_Bilinear(reduce_channels, size_before2[1:3], name="Upsampling2")

        dec_skip1 = Conv2D_Block(skip1, 48, filter_height=1, filter_width=1, stride=1, padding='SAME', batch_normalization=True, relu=True, name="feature_projection0")

        dec_concatenated = Concat([dec_up1,  dec_skip1], axis=-1, name="concatenation_decoder0")
        dec_block1 = SepConv_BN(dec_concatenated, 256, 'decoder_conv0', depth_activation=True, epsilon=1e-5)
        dec_block2 = SepConv_BN(dec_block1, 256, 'decoder_conv1', depth_activation=True, epsilon=1e-5)

    last_layer = Conv2D_Block(dec_block2, num_classess, filter_height=1, filter_width=1, stride=1, padding='SAME', name='Last_layer')
    size_before3 = tf.shape(input_shape)
    outputs = Resize_Bilinear(last_layer, size_before3[1:3], name="Upsampling3")

    # Ensure that the model takes into account
    # any potential predecessors of 'input_tensor'
