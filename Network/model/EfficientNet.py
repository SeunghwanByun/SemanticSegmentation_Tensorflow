# Copyright 2019 The TensorFlow Authors, Pavel Yakubovskiy, Björn Barz. All Rights Reserved.
#
# https://github.com/qubvel/efficientnet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.
[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""
# Code of this model implementation is mostly written by
# Björn Barz ([@Callidior](https://github.com/Callidior))

import cv2 as cv
import tensorflow as tf
import numpy as np
import scipy.misc
from datetime import timedelta
import os
import time
import string
from glob import glob

import collections
import functools
import math

from absl import logging
import six
from six.moves import xrange

# from condconv import condconv_layers

from Utils_EfficientNet import *

# 학습에 필요한 설정값들을 지정
# KEEP_PROB = 0.8
# MAX_ITERATION = 1e-2
# NUM_OF_CLASSESS = 2
IMAGE_SHAPE_KITTI = (160, 576)
# IMAGE_SHAPE_KITTI = (192, 704)
# IMAGE_SHAPE_KITTI = (384, 1280)
# IMAGE_SHAPE_KITTI = (713, 1280)
BATCH_SIZE = 1
EPOCHS = 50
# LEARNING_RATE = 1e-4

DATA_DIR = "data_road"

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])

# defaults will be a public argument for namedtuple in Python3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None, ) * len(BlockArgs._fields)

DEFAULT_BLOCK_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=7, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for dense kernels.

    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out', distribution='uniform').

    It is written out explicitly here for clarity.

    :param shape: shape of variable
    :param dtype: dtype of variable
    :param partition_info: unused
    :return: an initialization for the variables
    """
    del partition_info
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


def superpixel_kernel_initializer(shape, dtype='float32', partition_info=None):
    """Initializes superpixel kernels.

    This is inspired by space-to-depth transformation that is mathematically
    equivalent before and after the transformation. But we do the space-to-depth
    via a convolution. Moreover, we make the layer trainable instead of direct
    transform, we can initialization it this way so that the model can learn not
    to do anything but keep it mathematically equivalent, when improving
    performance.


    :param shape: shape of variable
    :param dtype: dtype of variable
    :param partition_info: unused
    :return: an initialization for the variable
    """
    del partition_info
    # use input depth to make superpixel kernel.
    depth = shape[-2]
    filters = np.zeros([2, 2, depth, 4 * depth], dtype=dtype)
    i = np.arange(2)
    j = np.arange(2)
    k = np.arange(depth)
    mesh = np.array(np.meshgrid(i, j, k)).T.reshape(-1, 3).T
    filters[mesh[0],
    mesh[1],
    mesh[2],
    4*mesh[2] + 2 * mesh[0] + mesh[1]] = 1
    return filters

def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


def MBConvBlock(inputs, block_args, activation, drop_rate=None, prefix='', ):
    """ Mobile Inverted Residual Bottleneck."""
    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 3

    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = Conv2D_Block(inputs, filters, filter_height=1, filter_width=1, stride=1, padding='SAME', name=prefix + 'expand_conv')
        x = Batch_Normalization(x, axis=bn_axis, name=prefix+'expand_bn')
        if activation == 'relu':
            x = tf.nn.relu(x)
        if activation == 'swish':
            x = tf.nn.swish(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = tf.nn.depthwise_conv2d(x, filter=[block_args.kernel_size, block_args.kernel_size, x.shape[-1], ], strides=block_args.strides,
                               padding='SAME', name=prefix + 'dwconv')
    x = Batch_Normalization(x, axis=bn_axis, name=prefix + 'bn')
    if activation == 'relu':
        x = tf.nn.relu(x)
    if activation == 'swish':
        x = tf.nn.swish(x)

    # Squeeze and Excitations phase
    if has_se:
        num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))
        se_tensor = Global_Avg_Pool(x, name=prefix + 'se_squeeze')

        target_shape = (1, 1, filters)
        se_tensor = tf.reshape(se_tensor, target_shape, name=prefix+'se_reshape')
        se_tensor = Conv2D_Block(se_tensor, num_reduced_filters, filter_height=1, filter_width=1, stride=1,
                                 padding='SAME', activation=activation, name=prefix + 'se_reduce')
        se_tensor = Conv2D_Block(se_tensor, filters, filter_height=1, filter_width=1, padding='SAME',
                                 activation='sigmoid', name=prefix + 'se_expand')
        x = tf.math.multiply(x, se_tensor, name=prefix + 'se_excite')

    # Output phase
    x = Conv2D_Block(x, block_args.output_filters, filter_height=1, filter_width=1, padding='SAME',
                     name=prefix + 'project_conv')
    x = Batch_Normalization(x, axis=bn_axis, name=prefix + 'project_bn')

    if block_args.id_skip and all(
        s == 1 for s in block_args.output_strides
    ) and block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            x = Dropout(x, keep_prob=drop_rate, noise_shape=(None, 1, 1, 1), name=prefix+'drop')
        x = tf.math.add(x, inputs, name=prefix+'add')

    return x


def EfficientNet(inputs,
                 width_coefficient,
                 depth_coefficient,
                 default_resolution,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCK_ARGS,
                 model_name='efficientnet',
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classess=2,
                 **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at '~/.keras/keras.json'.

    :param inputs:
    :param width_coefficient: float, scaling coefficient for network width.
    :param depth_coefficient: float, scaling coefficientnet for network depth.
    :param default_resolution: int, default input image size.
    :param dropout_rate: float, dropout rate before final classifier layer.
    :param drop_connect_rate: float, drouput rate at skip connections.
    :param depth_divisor: int.
    :param block_args: A list of BlockArgs to construct block modules.
    :param model_name: string, model_name.
    :param include_top: whether to include the fully-connected
                layers at the top of the network.
    :param weights: one of 'None' (random initialization),
                'imagenet' (pre-training on ImageNet),
                or the path to the weights file to be loaded.
    :param input_tensor: optional Keras tensor
                (i.e. output of 'layers.Input()')
                to use as image input for the model.
    :param input_shape:optional shape tuple, only to be specified
                if 'include_top' is False.
                It should have exactly 3 inputs channels.
    :param pooling: optional pooling mode for feature extraction
                when 'include_top' is 'False'.
                - 'None' means that the output of the model will be
                the 4D tensor output of the last convolutional layer.
                - 'avg' means that global average pooling will be
                applied to the output of the last convolutional layer, and thus
                the output of the model will be a 2D tensor.
                - 'max' means that global max pooling wil be applied.
    :param classess: optional number of classess to classify images into, only to be specified if 'include_top'
                is True, and if no 'weights' argument is specified.
    :param kwargs:
    :return: model
    """

    bn_axis = 3
    activation = 'swish'

    # Build stem
    x = inputs
    num_filters = round_filters(32, width_coefficient, depth_divisor)
    x = Conv2D_Block(x, num_filters, filter_height=3, filter_width=3, stride=2, padding='SAME', name='stem_conv')
    x = Batch_Normalization(x, axis=bn_axis, name='stem_bn')
    x = tf.nn.swish(x, name='stem_activation')

    # Build blocks
    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters, width_coefficient, depth_divisor),
            output_filters=round_filters(block_args.output_filters, width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = MBConvBlock(x, block_args, activation=activation, drop_rate=drop_rate, prefix='bock{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            # pylint: disable=protected-access
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1])
            # pylint: enable=protected-access
            for bidx in xrange(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(idx + 1, string.ascii_lowercase[bidx + 1])
                x = MBConvBlock(x, block_args, activation=activation, drop_rate=drop_rate, prefix=block_prefix)

                block_num += 1

    # Build top
    x = Conv2D_Block(x, round_filters(1280, width_coefficient, depth_divisor), filter_height=1, filter_width=1, padding='SAME', name='top_conv')
    x = Batch_Normalization(x, axis=bn_axis, name='top_bn')
    x = tf.nn.swish(x, name='top_activation')

    if include_top:
        x = Global_Avg_Pool(x)
        if dropout_rate and dropout_rate > 0:
            x = tf.nn.dropout(x, dropout_rate, name='top_dropout')
        x = tf.layers.dense(x, classess, activation=tf.nn.softmax, name='probs')
    else:
        if pooling == 'avg':
            x = Global_Avg_Pool(x)
        elif pooling == 'max':
            x = Global_Max_Pool(x)

    return x

def EfficientNetB0(
        inputs,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2,
        **kwargs
):
    return EfficientNet(
        inputs,
        1.0, 1.0, 224, 0.2,
        model_name='efficientnet-b0',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB1(
        inputs,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2,
        **kwargs
):
    return EfficientNet(
        inputs,
        1.0, 1.1, 240, 0.2,
        model_name='efficientnet-b1',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB2(
        inputs,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2,
        **kwargs
):
    return EfficientNet(
        inputs,
        1.1, 1.2, 260, 0.3,
        model_name='efficientnet-b2',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB3(
        inputs,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2,
        **kwargs
):
    return EfficientNet(
        inputs,
        1.2, 1.4, 300, 0.3,
        model_name='efficientnet-b3',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB4(
        inputs,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2,
        **kwargs
):
    return EfficientNet(
        inputs,
        1.4, 1.8, 380, 0.4,
        model_name='efficientnet-b4',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB5(
        inputs,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2,
        **kwargs
):
    return EfficientNet(
        inputs,
        1.6, 2.2, 456, 0.4,
        model_name='efficientnet-b5',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB6(
        inputs,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2,
        **kwargs
):
    return EfficientNet(
        inputs,
        1.8, 2.6, 528, 0.5,
        model_name='efficientnet-b6',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetB7(
        inputs,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2,
        **kwargs
):
    return EfficientNet(
        inputs,
        2.0, 3.1, 600, 0.5,
        model_name='efficientnet-b7',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )


def EfficientNetL2(
        inputs,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2,
        **kwargs
):
    return EfficientNet(
        inputs,
        4.3, 5.3, 800, 0.5,
        model_name='efficientnet-l2',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, classes=classes,
        **kwargs
    )