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
# 나는 일단 이 코드를 KITTI road 데이터셋을 통해 Semantic Segmentation을 하는 코드로 변경시킬 것.
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
import csv
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
KEEP_PROB = 0.8
MAX_ITERATION = 1e-2
NUM_OF_CLASSESS = 2
IMAGE_SHAPE_KITTI = (160, 576)
BATCH_SIZE = 1
EPOCHS = 50
# EPOCHS = 1
LEARNING_RATE = 1e-4

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
    filter_shape = [block_args.kernel_size, block_args.kernel_size, x.shape[-1], 1]
    filter = tf.get_variable(prefix+'_filter', shape=filter_shape, dtype=tf.float32)
    x = tf.nn.depthwise_conv2d(x, filter=filter, strides=[1, block_args.strides[0], block_args.strides[1], 1],
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

        target_shape = (1, 1, filters, 8)
        
        se_tensor = tf.expand_dims(se_tensor, dim=1)
        se_tensor = tf.expand_dims(se_tensor, dim=1)
        se_tensor = Conv2D_Block(se_tensor, num_reduced_filters, filter_height=1, filter_width=1, stride=1,
                                 padding='SAME', activation=activation, name=prefix + 'se_reduce')
        se_tensor = Conv2D_Block(se_tensor, filters, filter_height=1, filter_width=1, padding='SAME',
                                 activation='sigmoid', name=prefix + 'se_expand')
        x = tf.math.multiply(x, se_tensor, name=prefix + 'se_excite')

        # 약간 좀 다른 부분이 있음.

    # Output phase
    x = Conv2D_Block(x, block_args.output_filters, filter_height=1, filter_width=1, padding='SAME',
                     name=prefix + 'project_conv')
    x = Batch_Normalization(x, axis=bn_axis, name=prefix + 'project_bn')

    if block_args.id_skip and all(
        s == 1 for s in block_args.strides
    ) and block_args.input_filters == block_args.output_filters:
    #     if drop_rate and (drop_rate > 0):
    #         x = Dropout(x, keep_prob=drop_rate, noise_shape=[1], name=prefix+'drop')
        x = tf.math.add(x, inputs, name=prefix+'add')

    return x


def EfficientNet(inputs,
                 keep_prob,
                 width_coefficient,
                 depth_coefficient,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCK_ARGS,
                 include_top=True,
                 pooling=None,
                 classes=2):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at '~/.keras/keras.json'.

    :param inputs:
    :param width_coefficient: float, scaling coefficient for network width.
    :param depth_coefficient: float, scaling coefficientnet for network depth.
    :param dropout_rate: float, dropout rate before final classifier layer.
    :param drop_connect_rate: float, drouput rate at skip connections.
    :param depth_divisor: int.
    :param block_args: A list of BlockArgs to construct block modules.
    :param include_top: whether to include the fully-connected
                layers at the top of the network.
    :param pooling: optional pooling mode for feature extraction
                when 'include_top' is 'False'.
                - 'None' means that the output of the model will be
                the 4D tensor output of the last convolutional layer.
                - 'avg' means that global average pooling will be
                applied to the output of the last convolutional layer, and thus
                the output of the model will be a 2D tensor.
                - 'max' means that global max pooling wil be applied.
    :param classes: optional number of classess to classify images into, only to be specified if 'include_top'
                is True, and if no 'weights' argument is specified.
    :param kwargs:
    :return: model
    """

    bn_axis = 3
    activation = 'swish'

    # Build stem
    x = inputs
    num_filters = round_filters(32, width_coefficient, depth_divisor)
    # print("num_filters:", num_filters)
    # print("x:", x.shape)
    x = Conv2D_Block(x, num_filters, filter_height=3, filter_width=3, stride=2, padding='SAME', name='stem_conv')
    # print("x2:", x.shape)
    x = Batch_Normalization(x, axis=bn_axis, name='stem_bn')
    # print("x3:", x.shape)
    # x = tf.nn.swish(x, name='stem_activation')
    # x = tf.nn.sigmoid(x, name='stem_activation')

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
    # x = tf.nn.swish(x, name='top_activation')

    return x

def EfficientNetB0(
        inputs,
        keep_prob,
        include_top=True,
        classes=2,
):
    return EfficientNet(
        inputs, keep_prob,
        1.0, 1.0, 224, 0.2,
        include_top=include_top,
        classes=classes
)


def EfficientNetB1(
        inputs,
        keep_prob,
        include_top=True,
        classes=2,
):
    return EfficientNet(
        inputs, keep_prob,
        1.0, 1.1, 240, 0.2,
        include_top=include_top,
        classes=classes
    )


def EfficientNetB2(
        inputs,
        keep_prob,
        include_top=True,
        classes=2,
):
    return EfficientNet(
        inputs, keep_prob,
        1.1, 1.2, 260, 0.3,
        include_top=include_top,
        classes=classes,
    )


def EfficientNetB3(
        inputs,
        keep_prob,
        include_top=True,
        classes=2,
):
    return EfficientNet(
        inputs, keep_prob,
        1.2, 1.4, 300, 0.3,
        include_top=include_top,
        classes=classes,
    )


def EfficientNetB4(
        inputs,
        keep_prob,
        include_top=True,
        classes=2,
):
    return EfficientNet(
        inputs, keep_prob,
        1.4, 1.8, 380, 0.4,
        include_top=include_top,
        classes=classes,
    )


def EfficientNetB5(
        inputs,
        keep_prob,
        include_top=True,
        classes=2,
):
    return EfficientNet(
        inputs, keep_prob,
        1.6, 2.2, 456, 0.4,
        include_top=include_top,
        classes=classes,
    )


def EfficientNetB6(
        inputs,
        keep_prob,
        include_top=True,
        classes=2,
):
    return EfficientNet(
        inputs, keep_prob,
        1.8, 2.6, 528, 0.5,
        include_top=include_top,
        classes=classes,
    )


def EfficientNetB7(
        inputs,
        keep_prob,
        include_top=True,
        classes=2,
):
    return EfficientNet(
        inputs, keep_prob,
        2.0, 3.1, 600, 0.5,
        include_top=include_top,
        classes=classes,
    )


def EfficientNetL2(
        inputs,
        keep_prob,
        include_top=True,
        classes=2,
):
    return EfficientNet(
        inputs, keep_prob,
        4.3, 5.3, 800, 0.5,
        include_top=include_top,
        classes=classes,
    )

def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False):
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
        x = Zero_Padding(x, paddings=1, name=prefix+'zero_paddings')
        depth_padding = 'VALID'

    if not depth_activation:
        x = ReLU(x)

    # Atrous separable convolution contains atruos depthwise & pointwise convolution
    filter_shape = [kernel_size, kernel_size, x.get_shape()[-1], 1]
    filter = tf.get_variable(prefix+'_filter', shape=filter_shape, dtype=tf.float32)

    x = tf.nn.depthwise_conv2d(x, filter=filter, strides=[1, stride, stride, 1], rate=[rate, rate], padding=depth_padding, name=prefix+'_depthwise')

    x = Batch_Normalization(x)

    if depth_activation:
        x = ReLU(x)

    x = Conv2D_Block(x, filters, filter_height=1, filter_width=1, stride=1, padding='SAME', batch_normalization=True, activation='relu', name=prefix+"_pointwise_BN")

    return x

def EfficientSeg(inputs, keep_prob, phi, num_classes):
    """Efficientnet + Semantic Segmentation

    :param inputs:
    :return:
    """
    atrous_rate = (6, 12, 18)

    feature = 0

    # Encoder : feature extraction from Efficientnet-B0-7
    if phi == 'b0':
        feature = EfficientNetB0(inputs, keep_prob)
    elif phi == 'b1':
        feature = EfficientNetB1(inputs, keep_prob)
    elif phi == 'b2':
        feature = EfficientNetB2(inputs, keep_prob)
    elif phi == 'b3':
        feature = EfficientNetB3(inputs, keep_prob)
    elif phi == 'b4':
        feature = EfficientNetB4(inputs, keep_prob)
    elif phi == 'b5':
        feature = EfficientNetB5(inputs, keep_prob)
    elif phi == 'b6':
        feature = EfficientNetB6(inputs, keep_prob)
    else:
        feature = EfficientNetB7(inputs, keep_prob)

    # Decoder
    b4 = Global_Avg_Pool(feature, name='b4')
    b4 = tf.expand_dims(b4, dim=1)
    b4 = tf.expand_dims(b4, dim=1)
    b4 = Conv2D_Block(b4, 256, filter_height=1, filter_width=1, stride=1, padding='SAME', batch_normalization=True, activation='relu', name='image_pooling')

    size_before = tf.shape(feature)
    b4 = Resize_Bilinear(b4, size_before[1:3], name='upsampling_after_gap')

    # Simple 1x1
    b0 = Conv2D_Block(feature, 256, filter_height=1, filter_width=1, stride=1, padding='SAME', batch_normalization=True, activation='relu', name='aspp0')

    # rate 6 (12)
    b1 = SepConv_BN(feature, 256, 'aspp1', rate=atrous_rate[0], depth_activation=True)

    # rate 12 (24)
    b2 = SepConv_BN(feature, 256, 'aspp2', rate=atrous_rate[1], depth_activation=True)

    # rate 18 (36)
    b3 = SepConv_BN(feature, 256, 'aspp3', rate=atrous_rate[2], depth_activation=True)

    concatenated = Concat([b4, b0, b1, b2, b3], axis=-1, name='concatenation')

    concat_conv = Conv2D_Block(concatenated, 256, filter_height=1, filter_width=1, stride=1, padding='SAME', batch_normalization=True, activation='relu', name='concatenation_conv')
    concat_conv = Dropout(concat_conv, keep_prob=keep_prob)

    size_before2 = tf.shape(feature) * 2
    dec_up1 = Resize_Bilinear(concat_conv, size_before2[1:3], name='upsampling2')

    dec_block1 = SepConv_BN(dec_up1, 256, 'decoder_conv0', depth_activation=True)
    dec_block2 = SepConv_BN(dec_block1, 256, 'decoder_conv1', depth_activation=True)

    last_layer = Conv2D_Block(dec_block2, num_classes, filter_height=1, filter_width=1, stride=1, padding='SAME', name='last_layer')

    size_before3 = tf.shape(inputs)
    output = Resize_Bilinear(last_layer, size_before3[1:3], name='output')

    return output

def run():
    # GPU
    tf.debugging.set_log_device_placement(True)
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if gpu:
        try:
            tf.config.experimental.set_memory_growth(gpu[0], True)
        except RuntimeError as e:
            print(e)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Training 데이터 셋을 불러옴
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")
    RGB_IMAGE = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 3],
                               name="input")
    RGB_LABEL = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 2],
                                name="prediction")

    # Network 선언
    logits = EfficientSeg(RGB_IMAGE, keep_probability, 'b1', NUM_OF_CLASSESS)

    # Tensorboard를 위한 summary들을 지정
    tf.summary.image('input_image', RGB_IMAGE, max_outputs=2)
    # 손실 함수를 선언하고 손실 함수에 대한 summary들을 지정
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=RGB_LABEL))

    tf.summary.scalar('loss', loss)

    # 옵티마이저를 선언하고 파라미터를 한 스텝 업데이트하는 train_step 연산을 정의
    optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE)
    # train_step = optimizer.minimize(loss)

    # Constant to scale sum of gradient
    const = tf.compat.v1.constant(1 / BATCH_SIZE * 3)

    # Get all trainable variables
    t_vars = tf.compat.v1.trainable_variables()

    # Create a copy of all trainable variables with '0' as initial values
    accum_tvars = [tf.compat.v1.Variable(tf.compat.v1.zeros_like(t_var.initialized_value()), trainable=False) for t_var
                   in t_vars]

    # Create a op to initialize all accums vars
    zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvars]

    # Compute gradients for a batch
    batch_grads_vars = optimizer.compute_gradients(loss, t_vars)

    # Collect the (scaled by const) batch gradient into accumulated vars
    accum_ops = [accum_tvars[i].assign_add(tf.scalar_mul(const, batch_grad_var[0])) for i, batch_grad_var in
                     enumerate(batch_grads_vars)]

    # Apply accums gradients
    train_step = optimizer.apply_gradients(
        [(accum_tvars[i], batch_grad_var[1]) for i, batch_grad_var in enumerate(batch_grads_vars)])

    # Tensorboard를 위한 summary를 하나로 merge
    print("Setting up summary up")
    summary_op = tf.summary.merge_all()

    # training 데이터와 validation 데이터 개수를 불러옴.
    training_labels_count = len(glob(os.path.join(DATA_DIR, 'training/gt_image_2/*_road_*.png')))
    training_images_count = len(glob(os.path.join(DATA_DIR, 'training/image_2/*.png')))
    training_projection_count = len(glob(os.path.join(DATA_DIR, 'training/projection/*.png')))
    testing_images_count = len(glob(os.path.join(DATA_DIR, 'testing/image_2/*.png')))
    validating_labels_count = len(glob(os.path.join(DATA_DIR, 'validating/gt_image_2/*_road_*.png')))
    validating_images_count = len(glob(os.path.join(DATA_DIR, 'validating/image_2/*.png')))
    validating_projection_count = len(glob(os.path.join(DATA_DIR, 'validating/projection/*.png')))

    assert not (training_images_count == training_labels_count == testing_images_count == 0), \
        'Kitti dataset not found. Extract Kitti dataset in {}'.format(DATA_DIR)
    assert training_images_count == 259, 'Expected 259 training images, found {} images.'.format(
        training_images_count)  # 289
    assert training_labels_count == 259, 'Expected 259 training labels, found {} labels.'.format(
        training_labels_count)  # 289
    assert training_projection_count == 259, 'Expected 259 training projection images, found {} images.'.format(
        training_projection_count)
    assert testing_images_count == 290, 'Expected 290 testing images, found {} images.'.format(
        testing_images_count)
    assert validating_labels_count == 30, 'Expected 30 validating images, found {} images.'.format(
        validating_images_count)
    assert validating_labels_count == 30, 'Expected 30 validating images, found {} labels.'.format(
        validating_labels_count)
    assert validating_projection_count == 30, 'Expected 30 validaing projection images, found {} images.'.format(
        validating_projection_count)

    # training 데이터를 불러옴
    get_batches_fn = gen_batch_function(os.path.join(DATA_DIR, 'training'), IMAGE_SHAPE_KITTI)

    # 세션을 염
    sess = tf.Session(config=config)
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # 학습된 파라미터를 저장하기 위한 tf.train.Saver()
    # tensorboard summary들을 저장하기 위한 tf.summary.FileWriter를 선언
    print("Setting up Saver...")
    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter("logs/", sess.graph)

    # 변수들을 초기화하고 저장된 ckpt 파일이 있으면 저장된 파라미터를 불러옴
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("./model")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        sess.run(tf.global_variables_initializer())

    csv_path = os.path.join(DATA_DIR, "loss.csv")
    csvfile = open(csv_path, "w", newline="")
    start = time.time()  # 시작 시간 저장
    for epoch in range(EPOCHS):
        s_time = time.time()
        costs = []
        # 학습 데이터를 불러오고 feed_dict에 데이터를 지정
        # for images, img_labels, _, lid_labels in get_batches_fn(batch_size=BATCH_SIZE):
        for images, img_labels in get_batches_fn(batch_size=BATCH_SIZE):
            feed_dict = {RGB_IMAGE: images, RGB_LABEL: img_labels, keep_probability: 0.8}

            # Initialize the accumulated grads
            sess.run(zero_ops)
            for i in range(len(images)):
                sess.run(accum_ops, feed_dict=feed_dict)

            # train_step을 실행해서 파라미터를 한 스텝 업데이트 함
            _, cost = sess.run([train_step, loss], feed_dict=feed_dict)

            costs.append(cost)

            # Tensorboard 를 위한 sess.run()
            summary = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step=sess.run(global_step))

        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(costs)

        print("[Epoch: {0}/{1} Time: {2}]".format(epoch + 1, EPOCHS, str(timedelta(seconds=(time.time() - s_time)))))

    print("Time: ", time.time() - start)  # 현재 시각 - 시작 시간 = 실행 시간
    print("Training Successfully")

    # 훈련이 끝나고 학습된 파라미터 저장
    saver.save(sess, './model/Efficientnet.ckpt', global_step=global_step)

    # 훈련이 끝나고 테스트 데이터 셋으로 테스트
    output_dir = os.path.join(DATA_DIR, 'output')
    mask_dir = os.path.join(DATA_DIR, 'mask')
    print("Training Finished. Saving test images to: {}".format(output_dir))
    image_output = gen_test_output(sess, logits, keep_probability, RGB_IMAGE, os.path.join(DATA_DIR, 'validating'),
                                   IMAGE_SHAPE_KITTI)

    total_processing_time = 0
    for name, mask, image, processing_time in image_output:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        scipy.misc.imsave(os.path.join(mask_dir, name), mask)
        total_processing_time += processing_time

    print("Average processing time is : ", total_processing_time / 30)


if __name__ == '__main__':
    run()
