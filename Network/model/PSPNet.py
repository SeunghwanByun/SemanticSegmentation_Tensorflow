import cv2 as cv
import tensorflow as tf
import numpy as np
import scipy.misc
from datetime import timedelta
import os
import time
from glob import glob

from Utils import *

# 학습에 필요한 설정값들을 지정
KEEP_PROB = 0.8
MAX_ITERATION = 1e-2
NUM_OF_CLASSESS = 2
IMAGE_SHAPE_KITTI = (160, 576)
# IMAGE_SHAPE_KITTI = (192, 704)
# IMAGE_SHAPE_KITTI = (384, 1280)
# IMAGE_SHAPE_KITTI = (713, 1280)
BATCH_SIZE = 1
EPOCHS = 50
LEARNING_RATE = 1e-4

DATA_DIR = "data_road"

def PSPNet50(x, num_classes):
    # Network 구현
    print("Network Build Start...")
    # Stage 1
    conv1 = Zero_Padding(x, paddings=3, name="conv1")
    conv1 = Conv2D_Block(conv1, 64, 7, 7, 2, 'VALID', True, True, name="conv1")
    conv1 = Zero_Padding(conv1, paddings=1, name="conv1")

    pool1 = Max_Pooling(conv1, filter_height=3, filter_width=3, stride=2, name="pool1")

    # Stage 2
    conv2_1_1 = Conv2D_Block(pool1, 64, 1, 1, 1, 'VALID', True, True, name="conv2_1_1")
    conv2_1_2 = Conv2D_Block(conv2_1_1, 64, 3, 3, 1, 'SAME', True, True, name="conv2_1_2")
    conv2_1_3 = Conv2D_Block(conv2_1_2, 256, 1, 1, 1, 'VALID', True, name="conv2_1_3")
    shortcut1 = Conv2D_Block(pool1, 256, 1, 1, 1, 'VALID', True, name="shortcut1")
    conv2_1 = tf.add(conv2_1_3, shortcut1, name="conv2_1")
    conv2_1 = tf.nn.relu(conv2_1)

    conv2_2_1 = Conv2D_Block(conv2_1, 64, 1, 1, 1, 'VALID', True, True, name="conv2_2_1")
    conv2_2_2 = Conv2D_Block(conv2_2_1, 64, 3, 3, 1, 'SAME', True, True, name="conv2_2_2")
    conv2_2_3 = Conv2D_Block(conv2_2_2, 256, 1, 1, 1, 'VALID', True, name="conv2_2_3")
    conv2_2 = tf.add(conv2_2_3, conv2_1, name="conv2_2")
    conv2_2 = tf.nn.relu(conv2_2)

    conv2_3_1 = Conv2D_Block(conv2_2, 64, 1, 1, 1, 'VALID', True, True, name="conv2_3_1")
    conv2_3_2 = Conv2D_Block(conv2_3_1, 64, 3, 3, 1, 'SAME', True, True, name="conv2_3_2")
    conv2_3_3 = Conv2D_Block(conv2_3_2, 256, 1, 1, 1, 'SAME', True, name="conv2_3_3")
    conv2_3 = tf.add(conv2_3_3, conv2_2, name="conv2_3")
    conv2_3 = tf.nn.relu(conv2_3)

    # Stage 3
    conv3_1_1 = Conv2D_Block(conv2_3, 128, 1, 1, 2, 'VALID', True, True, name="conv3_1")
    conv3_1_2 = Conv2D_Block(conv3_1_1, 128, 3, 3, 1, 'SAME', True, True, name="conv3_2")
    conv3_1_3 = Conv2D_Block(conv3_1_2, 512, 1, 1, 1, 'VALID', True, name="conv3_3")
    shortcut2 = Conv2D_Block(conv2_3, 512, 1, 1, 2, 'VALID', True, name="shortcut2")
    conv3_1 = tf.add(conv3_1_3, shortcut2, name="conv3_1")
    conv3_1 = tf.nn.relu(conv3_1)

    conv3_2_1 = Conv2D_Block(conv3_1, 128, 1, 1, 1, 'VALID', True, True, name="conv3_2_1")
    conv3_2_2 = Conv2D_Block(conv3_2_1, 128, 3, 3, 1, 'SAME', True, True, name="conv3_2_2")
    conv3_2_3 = Conv2D_Block(conv3_2_2, 512, 1, 1, 1, 'VALID', True, name="conv3_2_3")
    conv3_2 = tf.add(conv3_2_3, conv3_1, name="conv3_2")
    conv3_2 = tf.nn.relu(conv3_2)

    conv3_3_1 = Conv2D_Block(conv3_2, 128, 1, 1, 1, 'VALID', True, True, name="conv3_3_1")
    conv3_3_2 = Conv2D_Block(conv3_3_1, 128, 3, 3, 1, 'SAME', True, True, name="conv3_3_2")
    conv3_3_3 = Conv2D_Block(conv3_3_2, 512, 1, 1, 1, 'VALID', True, name="conv3_3_3")
    conv3_3 = tf.add(conv3_3_3, conv3_1, name="conv3_3")
    conv3_3 = tf.nn.relu(conv3_3)

    conv3_4_1 = Conv2D_Block(conv3_3, 128, 1, 1, 1, 'VALID', True, True, name="conv3_4_1")
    conv3_4_2 = Conv2D_Block(conv3_4_1, 128, 3, 3, 1, 'SAME', True, True, name="conv3_4_2")
    conv3_4_3 = Conv2D_Block(conv3_4_2, 512, 1, 1, 1, 'VALID', True, name="conv3_4_3")
    conv3_4 = tf.add(conv3_4_3, conv3_1, name="conv3_4")
    conv3_4 = tf.nn.relu(conv3_4)

    # Stage 4
    conv4_1_1 = Conv2D_Block(conv3_4, 256, 1, 1, 2, 'VALID', True, True, name="conv4_1_1")
    conv4_1_2 = Conv2D_Block(conv4_1_1, 256, 3, 3, 1, 'SAME', True, True, name="conv4_1_2")
    conv4_1_3 = Conv2D_Block(conv4_1_2, 1024, 1, 1, 1, 'VALID', True, name="conv4_1_3")
    shortcut3 = Conv2D_Block(conv3_4, 1024, 1, 1, 2, 'VALID', True, name="shortcut3")
    conv4_1 = tf.add(conv4_1_3, shortcut3, name="conv4_1")
    conv4_1 = tf.nn.relu(conv4_1)

    conv4_2_1 = Conv2D_Block(conv4_1, 256, 1, 1, 1, 'VALID', True, True, name="conv4_2_1")
    conv4_2_2 = Conv2D_Block(conv4_2_1, 256, 3, 3, 1, 'SAME', True, True, name="conv4_2_2")
    conv4_2_3 = Conv2D_Block(conv4_2_2, 1024, 1, 1, 1, 'VALID', True, name="conv4_2_3")
    conv4_2 = tf.add(conv4_2_3, conv4_1, name="conv4_2")
    conv4_2 = tf.nn.relu(conv4_2)

    conv4_3_1 = Conv2D_Block(conv4_2, 256, 1, 1, 1, 'VALID', True, True, name="conv4_3_1")
    conv4_3_2 = Conv2D_Block(conv4_3_1, 256, 3, 3, 1, 'SAME', True, True, name="conv4_3_2")
    conv4_3_3 = Conv2D_Block(conv4_3_2, 1024, 1, 1, 1, 'VALID', True, name="conv4_3_3")
    conv4_3 = tf.add(conv4_3_3, conv4_2, name="conv4_3")
    conv4_3 = tf.nn.relu(conv4_3)

    conv4_4_1 = Conv2D_Block(conv4_3, 256, 1, 1, 1, 'VALID', True, True, name="conv4_4_1")
    conv4_4_2 = Conv2D_Block(conv4_4_1, 256, 3, 3, 1, 'SAME', True, True, name="conv4_4_2")
    conv4_4_3 = Conv2D_Block(conv4_4_2, 1024, 1, 1, 1, 'VALID', True, name="conv4_4_3")
    conv4_4 = tf.add(conv4_4_3, conv4_3, name="conv4_4")
    conv4_4 = tf.nn.relu(conv4_4)

    conv4_5_1 = Conv2D_Block(conv4_4, 256, 1, 1, 1, 'VALID', True, True, name="conv4_5_1")
    conv4_5_2 = Conv2D_Block(conv4_5_1, 256, 3, 3, 1, 'SAME', True, True, name="conv4_5_2")
    conv4_5_3 = Conv2D_Block(conv4_5_2, 1024, 1, 1, 1, 'VALID', True, name="conv4_5_3")
    conv4_5 = tf.add(conv4_5_3, conv4_4, name="conv4_5")
    conv4_5 = tf.nn.relu(conv4_5)

    conv4_6_1 = Conv2D_Block(conv4_5, 256, 1, 1, 1, 'VALID', True, True, name="conv4_6_1")
    conv4_6_2 = Conv2D_Block(conv4_6_1, 256, 3, 3, 1, 'SAME', True, True, name="conv4_6_2")
    conv4_6_3 = Conv2D_Block(conv4_6_2, 1024, 1, 1, 1, 'VALID', True, name="conv4_6_3")
    conv4_6 = tf.add(conv4_6_3, conv4_5, name="conv4_6")
    conv4_6 = tf.nn.relu(conv4_6)

    # Stage 5
    conv5_1_1 = Conv2D_Block(conv4_6, 512, 1, 1, 2, 'VALID', True, True, name="conv5_1_1")
    print("conv5_1_1.get_shape()", conv5_1_1.get_shape())
    conv5_1_2 = Conv2D_Block(conv5_1_1, 512, 3, 3, 1, 'SAME', True, True, name="conv5_1_2")
    conv5_1_3 = Conv2D_Block(conv5_1_2, 2048, 1, 1, 1, 'VALID', True, name="conv5_1_3")
    shortcut4 = Conv2D_Block(conv4_6, 2048, 1, 1, 2, 'VALID', True, name="shortcut4")
    conv5_1 = tf.add(conv5_1_3, shortcut4, name="conv5_1")
    conv5_1 = tf.nn.relu(conv5_1)

    conv5_2_1 = Conv2D_Block(conv5_1, 512, 1, 1, 1, 'VALID', True, True, name="conv5_2_1")
    conv5_2_2 = Conv2D_Block(conv5_2_1, 512, 3, 3, 1, 'SAME', True, True, name="conv5_2_2")
    conv5_2_3 = Conv2D_Block(conv5_2_2, 2048, 1, 1, 1, 'VALID', True, name="conv5_2_3")
    conv5_2 = tf.add(conv5_2_3, conv5_1, name="conv5_2")
    conv5_2 = tf.nn.relu(conv5_2)

    conv5_3_1 = Conv2D_Block(conv5_2, 512, 1, 1, 1, 'VALID', True, True, name="conv5_3_1")
    conv5_3_2 = Conv2D_Block(conv5_3_1, 512, 3, 3, 1, 'SAME', True, True, name="conv5_3_2")
    conv5_3_3 = Conv2D_Block(conv5_3_2, 2048, 1, 1, 1, 'VALID', True, name="conv5_3_3")
    conv5_3 = tf.add(conv5_3_3, conv5_1, name="conv5_3")
    conv5_3 = tf.nn.relu(conv5_3)

    encoder_output = Conv2D_Block(conv5_3, 512, 1, 1, 1, 'SAME', name='encoder_output')

    print("Encoder Build Finished...")
    shape = tf.shape(x)[1:3]
    encoder_interp = Resize_Bilinear(encoder_output, shape, name="conv5_3_interp")

    pool1 = Avg_Pooling(encoder_output, filter_height=5, filter_width=18, stride_height=5, stride_width=18,
                               name="conv5_3_pool1")
    pool1_conv = Conv2D_Block(pool1, 128, filter_height=1, filter_width=1, stride=1,
                                      batch_normalization=True, relu=True, name="conv5_3_pool1_conv")
    pool1_interp = Resize_Bilinear(pool1_conv, shape, name="conv5_3_pool1_interp")

    pool2 = Avg_Pooling(encoder_output, filter_height=3, filter_width=9, stride_height=3, stride_width=9,
                                name="conv5_3_pool2")
    pool2_conv = Conv2D_Block(pool2, 128, filter_height=1, filter_width=1, stride=1,
                                      batch_normalization=True, relu=True, name="conv5_3_pool2_conv")
    pool2_interp = Resize_Bilinear(pool2_conv, shape, name="conv5_3_pool2_interp")

    pool3 = Avg_Pooling(encoder_output, filter_height=2, filter_width=6, stride_height=2, stride_width=6,
                                name="conv5_3_pool3")
    pool3_conv = Conv2D_Block(pool3, 128, filter_height=1, filter_width=1, stride=1,
                                      batch_normalization=True, relu=True, name="conv5_3_pool3_conv")
    pool3_interp = Resize_Bilinear(pool3_conv, shape, name="conv5_3_pool3_interp")

    pool6 = Avg_Pooling(encoder_output, filter_height=1, filter_width=3, stride_height=1, stride_width=3,
                                name="conv5_3_pool6")

    pool6_conv = Conv2D_Block(pool6, 128, filter_height=1, filter_width=1, stride=1,
                                      batch_normalization=True, relu=True, name="conv5_3_pool6_conv")
    pool6_interp = Resize_Bilinear(pool6_conv, shape, name="conv5_3_pool6_interp")

    concatenation = Concat([encoder_interp, pool1_interp, pool2_interp, pool3_interp, pool6_interp], axis=-1,
                           name="concatenation")

    conv5_4 = Conv2D_Block(concatenation, 128, batch_normalization=True, relu=True, name="conv5_4")
    conv6 = Conv2D_Block(conv5_4, num_classes, filter_height=1, filter_width=1, stride=1, name="conv6")

    prediction = tf.argmax(conv6, dimension=3, name="prediction")
    print("Builde Network done...")

    return tf.expand_dims(prediction, dim=3), conv6

def PSPNet101(x, num_classes):
    # Network based on ResNet-101 : PSPNet101
    print("Build Network start...")
    # Stage1
    conv1_1_3x3_s2 = Conv2D_Block(x, 64, stride=2, padding='SAME', batch_normalization=True, relu=True, name="conv1_1_3x3_s2")
    conv1_2_3x3 = Conv2D_Block(conv1_1_3x3_s2, 64, padding='SAME', batch_normalization=True, relu=True, name="conv1_1_3x3")
    conv1_3_3x3 = Conv2D_Block(conv1_2_3x3, 128, padding='SAME', batch_normalization=True, relu=True, name="conv1_3_3x3")
    pool1_3x3_s2 = Max_Pooling(conv1_3_3x3, filter_height=3, filter_width=3, padding='SAME', name="pool1_3x3_s2")
    conv2_1_1x1_proj = Conv2D_Block(pool1_3x3_s2, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                    batch_normalization=True, relu=False, name="conv2_1_1x1_proj")

    # Stage2
    conv2_1_1x1_reduce = Conv2D_Block(pool1_3x3_s2, 64, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                    batch_normalization=True, relu=True, name="conv2_1_1x1_reduce")
    padding1 = Zero_Padding(conv2_1_1x1_reduce, paddings=1, name="padding1")
    conv2_1_3x3 = Conv2D_Block(padding1, 64, padding='VALID', batch_normalization=True, relu=True, name="conv2_1_3x3")
    conv2_1_1x1_increase = Conv2D_Block(conv2_1_3x3, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, relu=False, name="conv2_1_1x1_increase")

    print("conv2_1_proj", conv2_1_1x1_proj.shape)
    print("conv2_1_1x1_increase", conv2_1_1x1_increase.shape)
    conv2_1 = tf.add(conv2_1_1x1_proj, conv2_1_1x1_increase, name="conv2_1")
    conv2_1_relu = tf.nn.relu(conv2_1, name="conv2_1_relu")
    conv2_2_1x1_reduce = Conv2D_Block(conv2_1_relu, 64, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv2_2_1x1_reduce")
    padding2 = Zero_Padding(conv2_2_1x1_reduce, paddings=1, name="padding2")
    conv2_2_3x3 = Conv2D_Block(padding2, 64, padding='VALID', batch_normalization=True, relu=True, name="conv2_2_3x3")
    conv2_2_1x1_increase = Conv2D_Block(conv2_2_3x3, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, relu=False, name="conv2_2_1x1_increase")

    print("conv2_1_relu", conv2_1_relu.shape)
    print("conv2_2_1x1_increase", conv2_2_1x1_increase.shape)
    conv2_2 = tf.add(conv2_1_relu, conv2_2_1x1_increase, name="conv2_2")
    conv2_2_relu = tf.nn.relu(conv2_2, name="conv2_2_relu")
    conv2_3_1x1_reduce = Conv2D_Block(conv2_2_relu, 64, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv2_3_1x1_reduce")
    padding3 = Zero_Padding(conv2_3_1x1_reduce, paddings=1, name="padding3")
    conv2_3_3x3 = Conv2D_Block(padding3, 64, padding='VALID', batch_normalization=True, relu=True, name="conv2_3_3x3")
    conv2_3_1x1_increase = Conv2D_Block(conv2_3_3x3, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, relu=False, name="conv2_3_1x1_increase")

    conv2_3 = tf.add(conv2_2_relu, conv2_3_1x1_increase, name="conv2_3")
    conv2_3_relu = tf.nn.relu(conv2_3, name="conv2_3_relu")
    conv3_1_1x1_proj = Conv2D_Block(conv2_3_relu, 512, filter_height=1, filter_width=1, stride=2, padding='VALID',
                                    batch_normalization=True, relu=False, name="conv3_1_1x1_proj")

    # Stage 3
    conv3_1_1x1_reduce = Conv2D_Block(conv2_3_relu, 128, filter_height=1, filter_width=1, stride=2, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv3_1_1x1_reduce")
    padding4 = Zero_Padding(conv3_1_1x1_reduce, paddings=1, name="padding4")
    conv3_1_3x3 = Conv2D_Block(padding4, 128, padding='VALID', batch_normalization=True, relu=True, name="conv3_1_3x3")
    conv3_1_1x1_increase = Conv2D_Block(conv3_1_3x3, 512, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, relu=False, name="conv3_1_1x1_increase")
    print("conv3_1_1x1_proj", conv3_1_1x1_proj.shape)
    print("conv3_1_1x1_increase", conv3_1_1x1_increase.shape)
    conv3_1 = tf.add(conv3_1_1x1_proj, conv3_1_1x1_increase, name="conv3_1")
    conv3_1_relu = tf.nn.relu(conv3_1, name="conv3_1_relu")
    conv3_2_1x1_reduce = Conv2D_Block(conv3_1_relu, 128, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv3_2_1x1_reduce")
    padding5 = Zero_Padding(conv3_2_1x1_reduce, paddings=1, name="padding5")
    conv3_2_3x3 = Conv2D_Block(padding5, 128, padding='VALID', batch_normalization=True, relu=True, name="conv3_2_3x3")
    conv3_2_1x1_increase = Conv2D_Block(conv3_2_3x3, 512, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, relu=False, name="conv3_2_1x1_increase")

    print("conv3_1_relu", conv3_1_relu.shape)
    print("conv3_2_1x1_increase", conv3_2_1x1_increase.shape)
    conv3_2 = tf.add(conv3_1_relu, conv3_2_1x1_increase, name="conv3_2")
    conv3_2_relu = tf.nn.relu(conv3_2, name="conv3_2_relu")
    conv3_3_1x1_reduce = Conv2D_Block(conv3_2_relu, 128, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv3_3_1x1_reduce")
    padding6 = Zero_Padding(conv3_3_1x1_reduce, paddings=1, name="padding6")
    conv3_3_3x3 = Conv2D_Block(padding6, 128, padding='VALID', batch_normalization=True, relu=True, name="conv3_3_3x3")
    conv3_3_1x1_increase = Conv2D_Block(conv3_3_3x3, 512, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, relu=False, name="conv3_3_1x1_increase")

    print("conv3_2_relu.shape", conv3_2_relu.shape)
    print("conv3_3_1x1_increase", conv3_3_1x1_increase.shape)
    conv3_3 = tf.add(conv3_2_relu, conv3_3_1x1_increase, name="conv3_3")
    conv3_3_relu = tf.nn.relu(conv3_3, name="conv3_3_relu")
    conv3_4_1x1_reduce = Conv2D_Block(conv3_3_relu, 128, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv3_4_1x1_reduce")
    padding7 = Zero_Padding(conv3_4_1x1_reduce, paddings=1, name="padding7")
    conv3_4_3x3 = Conv2D_Block(padding7, 128, padding='VALID', batch_normalization=True, relu=True, name="conv3_4_3x3")
    conv3_4_1x1_increase = Conv2D_Block(conv3_4_3x3, 512, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, relu=False, name="conv3_4_1x1_increase")

    conv3_4 = tf.add(conv3_3_relu, conv3_4_1x1_increase, name="conv3_4")
    conv3_4_relu = tf.nn.relu(conv3_4, name="conv3_4_relu")
    conv4_1_1x1_proj = Conv2D_Block(conv3_4_relu, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                    batch_normalization=True, relu=False, name="conv4_1_1x1_proj")

    # Stage 4
    conv4_1_1x1_reduce = Conv2D_Block(conv3_4_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv4_1_1x1_reduce")
    padding8 = Zero_Padding(conv4_1_1x1_reduce, paddings=2, name="padding8")
    conv_4_1_3x3 = Atrous_Conv2D_Block(padding8, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                       name="conv_4_1_3X3")
    conv4_1_1x1_increase = Conv2D_Block(conv_4_1_3x3, 1024, filter_height=1, filter_width=1, stride=1,
                                        batch_normalization=True, name="conv4_1_1x1_increase")

    conv4_1 = tf.add(conv4_1_1x1_proj, conv4_1_1x1_increase, name="conv4_1")
    conv4_1_relu = tf.nn.relu(conv4_1, name="conv4_1_relu")
    conv4_2_1x1_reduce = Conv2D_Block(conv4_1_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv4_2_1x1_reduce")
    padding9 = Zero_Padding(conv4_2_1x1_reduce, paddings=2, name="padding9")
    conv4_2_3x3 = Atrous_Conv2D_Block(padding9, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                      name="conv4_2_3x3")
    conv4_2_1x1_increase = Conv2D_Block(conv4_2_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, name="conv4_2_1x1_increase")

    conv4_2 = tf.add(conv4_1_relu, conv4_2_1x1_increase, name="conv4_2")
    conv4_2_relu = tf.nn.relu(conv4_2, name="conv4_2_relu")
    conv4_3_1x1_reduce = Conv2D_Block(conv4_2_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                       batch_normalization=True, relu=True, name="conv4_3_1x1_reduce")
    padding10 = Zero_Padding(conv4_3_1x1_reduce, paddings=2, name="padding10")
    conv4_3_3x3 = Atrous_Conv2D_Block(padding10, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                      name="conv4_3_3x3")
    conv4_3_1x1_increase = Conv2D_Block(conv4_3_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, name="conv4_3_1x1_increase")

    conv4_3 = tf.add(conv4_2_relu, conv4_3_1x1_increase, name="conv4_3")
    conv4_3_relu = tf.nn.relu(conv4_3, name="conv4_3_relu")
    conv4_4_1x1_reduce = Conv2D_Block(conv4_3_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv4_4_1x1_reduce")
    padding11 = Zero_Padding(conv4_4_1x1_reduce, paddings=2, name="padding11")
    conv4_4_3x3 = Atrous_Conv2D_Block(padding11, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                      name="conv4_4_3x3")
    conv4_4_1x1_increase = Conv2D_Block(conv4_4_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, name="conv4_4_1x1_increase")

    conv4_4 = tf.add(conv4_3_relu, conv4_4_1x1_increase, name="conv4_4")
    conv4_4_relu = tf.nn.relu(conv4_4, name="conv4_4_relu")
    conv4_5_1x1_reduce = Conv2D_Block(conv4_4_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv4_5_1x1_reduce")
    padding12 = Zero_Padding(conv4_5_1x1_reduce, paddings=2, name="padding12")
    conv4_5_3x3 = Atrous_Conv2D_Block(padding12, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                      name="conv4_5_3x3")
    conv4_5_1x1_increase = Conv2D_Block(conv4_5_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, name="conv4_5_1x1_increase")

    conv4_5 = tf.add(conv4_4_relu, conv4_5_1x1_increase, name="conv4_5")
    conv4_5_relu = tf.nn.relu(conv4_5, name="conv4_5_relu")
    conv4_6_1x1_reduce = Conv2D_Block(conv4_5_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv4_6_1x1_reduce")
    padding13 = Zero_Padding(conv4_6_1x1_reduce, paddings=2, name="padding13")
    conv4_6_3x3 = Atrous_Conv2D_Block(padding13, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                      name="conv4_6_3x3")
    conv4_6_1x1_increase = Conv2D_Block(conv4_6_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, name="conv4_6_1x1_increase")

    conv4_6 = tf.add(conv4_5_relu, conv4_6_1x1_increase, name="conv4_5")
    conv4_6_relu = tf.nn.relu(conv4_6, name="conv4_6_relu")
    conv4_7_1x1_reduce = Conv2D_Block(conv4_6_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv4_7_1x1_reduce")
    padding14 = Zero_Padding(conv4_7_1x1_reduce, paddings=2, name="padding14")
    conv4_7_3x3 = Atrous_Conv2D_Block(padding14, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                      name="conv4_7_3x3")
    conv4_7_1x1_increase = Conv2D_Block(conv4_7_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, name="conv4_7_1x1_increase")

    conv4_7 = tf.add(conv4_6_relu, conv4_7_1x1_increase, name="conv4_7")
    conv4_7_relu = tf.nn.relu(conv4_7, name="conv4_7_relu")
    conv4_8_1x1_reduce = Conv2D_Block(conv4_7_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv4_8_1x1_reduce")
    padding15 = Zero_Padding(conv4_8_1x1_reduce, paddings=2, name="padding15")
    conv4_8_3x3 = Atrous_Conv2D_Block(padding15, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                      name="conv4_8_3x3")
    conv4_8_1x1_increase = Conv2D_Block(conv4_8_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, name="conv4_8_1x1_increase")

    conv4_8 = tf.add(conv4_7_relu, conv4_8_1x1_increase, name="conv4_8")
    conv4_8_relu = tf.nn.relu(conv4_8, name="conv4_8_relu")
    conv4_9_1x1_reduce = Conv2D_Block(conv4_8_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv4_9_1x1_reduce")
    padding16 = Zero_Padding(conv4_9_1x1_reduce, paddings=2, name="padding16")
    conv4_9_3x3 = Atrous_Conv2D_Block(padding16, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                      name="conv4_9_3x3")
    conv4_9_1x1_increase = Conv2D_Block(conv4_9_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, name="conv4_9_1x1_increase")

    conv4_9 = tf.add(conv4_8_relu, conv4_9_1x1_increase, name="conv4_9")
    conv4_9_relu = tf.nn.relu(conv4_9, name="conv4_9_relu")
    conv4_10_1x1_reduce = Conv2D_Block(conv4_9_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                       batch_normalization=True, relu=True, name="conv4_10_1x1_reduce")
    padding17 = Zero_Padding(conv4_10_1x1_reduce, paddings=2, name="padding17")
    conv4_10_3x3 = Atrous_Conv2D_Block(padding17, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                       name="conv4_10_3x3")
    conv4_10_1x1_increase = Conv2D_Block(conv4_10_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                         batch_normalization=True, name="conv4_10_1x1_increase")

    conv4_10 = tf.add(conv4_9_relu, conv4_10_1x1_increase, name="conv4_10")
    conv4_10_relu = tf.nn.relu(conv4_10, name="conv4_10_relu")
    conv4_11_1x1_reduce = Conv2D_Block(conv4_10_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                       batch_normalization=True, relu=True, name="conv4_10_1x1_reduce")
    padding18 = Zero_Padding(conv4_11_1x1_reduce, paddings=2, name="padding18")
    conv4_11_3x3 = Atrous_Conv2D_Block(padding18, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                       name="conv4_11_3x3")
    conv4_11_1x1_increase = Conv2D_Block(conv4_11_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                         batch_normalization=True, name="conv4_11_1x1_increase")

    conv4_11 = tf.add(conv4_10_relu, conv4_11_1x1_increase, name="conv4_11")
    conv4_11_relu = tf.nn.relu(conv4_11, name="conv4_11_relu")
    conv4_12_1x1_reduce = Conv2D_Block(conv4_11_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                       batch_normalization=True, relu=True, name="conv4_12_1x1_reduce")
    padding19 = Zero_Padding(conv4_12_1x1_reduce, paddings=2, name="padding19")
    conv4_12_3x3 = Atrous_Conv2D_Block(padding19, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                       name="conv4_12_3x3")
    conv4_12_1x1_increase = Conv2D_Block(conv4_12_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                         batch_normalization=True, name="conv4_12_1x1_increase")

    conv4_12 = tf.add(conv4_11_relu, conv4_12_1x1_increase, name="conv4_12")
    conv4_12_relu = tf.nn.relu(conv4_12, name="conv4_12_relu")
    conv4_13_1x1_reduce = Conv2D_Block(conv4_12_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                       batch_normalization=True, relu=True, name="conv4_13_1x1_reduce")
    padding20 = Zero_Padding(conv4_13_1x1_reduce, paddings=2, name="padding20")
    conv4_13_3x3 = Atrous_Conv2D_Block(padding20, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                       name="conv4_13_3x3")
    conv4_13_1x1_increase = Conv2D_Block(conv4_13_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                         batch_normalization=True, name="conv4_13_1x1_increase")

    conv4_13 = tf.add(conv4_12_relu, conv4_13_1x1_increase, name="conv4_13")
    conv4_13_relu = tf.nn.relu(conv4_13, name="conv4_13_relu")
    conv4_14_1x1_reduce = Conv2D_Block(conv4_13_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                       batch_normalization=True, relu=True, name="conv4_14_1x1_reduce")
    padding21 = Zero_Padding(conv4_14_1x1_reduce, paddings=2, name="padding21")
    conv4_14_3x3 = Atrous_Conv2D_Block(padding21, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                       name="conv4_14_3x3")
    conv4_14_1x1_increase = Conv2D_Block(conv4_14_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                         batch_normalization=True, name="conv4_14_1x1_increase")

    conv4_14 = tf.add(conv4_13_relu, conv4_14_1x1_increase, name="conv4_14")
    conv4_14_relu = tf.nn.relu(conv4_14, name="conv4_14_relu")
    conv4_15_1x1_reduce = Conv2D_Block(conv4_14_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                       batch_normalization=True, relu=True, name="conv4_15_1x1_reduce")
    padding22 = Zero_Padding(conv4_15_1x1_reduce, paddings=2, name="padding22")
    conv4_15_3x3 = Atrous_Conv2D_Block(padding22, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                       name="conv4_15_3x3")
    conv4_15_1x1_increase = Conv2D_Block(conv4_15_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                         batch_normalization=True, name="conv4_15_1x1_increase")

    conv4_15 = tf.add(conv4_14_relu, conv4_15_1x1_increase, name="conv4_15")
    conv4_15_relu = tf.nn.relu(conv4_15, name="conv4_15_relu")
    conv4_16_1x1_reduce = Conv2D_Block(conv4_15_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                       batch_normalization=True, relu=True, name="conv4_16_1x1_reduce")
    padding23 = Zero_Padding(conv4_16_1x1_reduce, paddings=2, name="padding23")
    conv4_16_3x3 = Atrous_Conv2D_Block(padding23, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                       name="conv4_16_3x3")
    conv4_16_1x1_increase = Conv2D_Block(conv4_16_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                         batch_normalization=True, name="conv4_16_1x1_increase")

    conv4_16 = tf.add(conv4_15_relu, conv4_16_1x1_increase, name="conv4_16")
    conv4_16_relu = tf.nn.relu(conv4_16, name="conv4_16_relu")
    conv4_17_1x1_reduce = Conv2D_Block(conv4_16_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                       batch_normalization=True, relu=True, name="conv4_17_1x1_reduce")
    padding24 = Zero_Padding(conv4_17_1x1_reduce, paddings=2, name="padding24")
    conv4_17_3x3 = Atrous_Conv2D_Block(padding24, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                       name="conv4_17_3x3")
    conv4_17_1x1_increase = Conv2D_Block(conv4_17_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                         batch_normalization=True, name="conv4_17_1x1_increase")

    conv4_17 = tf.add(conv4_16_relu, conv4_17_1x1_increase, name="conv4_17")
    conv4_17_relu = tf.nn.relu(conv4_17, name="conv4_17_relu")
    conv4_18_1x1_reduce = Conv2D_Block(conv4_17_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                       batch_normalization=True, relu=True, name="conv4_18_1x1_reduce")
    padding25 = Zero_Padding(conv4_18_1x1_reduce, paddings=2, name="padding25")
    conv4_18_3x3 = Atrous_Conv2D_Block(padding25, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                       name="conv4_18_3x3")
    conv4_18_1x1_increase = Conv2D_Block(conv4_18_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                         batch_normalization=True, name="conv4_18_1x1_increase")

    conv4_18 = tf.add(conv4_17_relu, conv4_18_1x1_increase, name="conv4_18")
    conv4_18_relu = tf.nn.relu(conv4_18, name="conv4_18_relu")
    conv4_19_1x1_reduce = Conv2D_Block(conv4_18_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                       batch_normalization=True, relu=True, name="conv4_19_1x1_reduce")
    padding26 = Zero_Padding(conv4_19_1x1_reduce, paddings=2, name="padding26")
    conv4_19_3x3 = Atrous_Conv2D_Block(padding26, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                       name="conv4_19_3x3")
    conv4_19_1x1_increase = Conv2D_Block(conv4_19_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                         batch_normalization=True, name="conv4_19_1x1_increase")

    conv4_19 = tf.add(conv4_18_relu, conv4_19_1x1_increase, name="conv4_19")
    conv4_19_relu = tf.nn.relu(conv4_19, name="conv4_19_relu")
    conv4_20_1x1_reduce = Conv2D_Block(conv4_19_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                       batch_normalization=True, relu=True, name="conv4_20_1x1_reduce")
    padding27 = Zero_Padding(conv4_20_1x1_reduce, paddings=2, name="padding27")
    conv4_20_3x3 = Atrous_Conv2D_Block(padding27, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                       name="conv4_20_3x3")
    conv4_20_1x1_increase = Conv2D_Block(conv4_20_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                         batch_normalization=True, name="conv4_20_1x1_increase")

    conv4_20 = tf.add(conv4_19_relu, conv4_20_1x1_increase, name="conv4_20")
    conv4_20_relu = tf.nn.relu(conv4_20, name="conv4_20_relu")
    conv4_21_1x1_reduce = Conv2D_Block(conv4_20_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                       batch_normalization=True, relu=True, name="conv4_21_1x1_reduce")
    padding28 = Zero_Padding(conv4_21_1x1_reduce, paddings=2, name="padding28")
    conv4_21_3x3 = Atrous_Conv2D_Block(padding28, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                       name="conv4_21_3x3")
    conv4_21_1x1_increase = Conv2D_Block(conv4_21_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                         batch_normalization=True, name="conv4_21_1x1_increase")

    conv4_21 = tf.add(conv4_20_relu, conv4_21_1x1_increase, name="conv4_21")
    conv4_21_relu = tf.nn.relu(conv4_21, name="conv4_21_relu")
    conv4_22_1x1_reduce = Conv2D_Block(conv4_21_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                       batch_normalization=True, relu=True, name="conv4_22_1x1_reduce")
    padding29 = Zero_Padding(conv4_22_1x1_reduce, paddings=2, name="padding29")
    conv4_22_3x3 = Atrous_Conv2D_Block(padding29, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                       name="conv4_22_3x3")
    conv4_22_1x1_increase = Conv2D_Block(conv4_22_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                         batch_normalization=True, name="conv4_22_1x1_increase")

    conv4_22 = tf.add(conv4_21_relu, conv4_22_1x1_increase, name="conv4_22")
    conv4_22_relu = tf.nn.relu(conv4_22, name="conv4_22_relu")
    conv4_23_1x1_reduce = Conv2D_Block(conv4_22_relu, 256, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                       batch_normalization=True, relu=True, name="conv4_23_1x1_reduce")
    padding30 = Zero_Padding(conv4_23_1x1_reduce, paddings=2, name="padding30")
    conv4_23_3x3 = Atrous_Conv2D_Block(padding30, 256, dilation=2, padding='VALID', batch_normalization=True, relu=True,
                                       name="conv4_23_3x3")
    conv4_23_1x1_increase = Conv2D_Block(conv4_23_3x3, 1024, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                         batch_normalization=True, name="conv4_23_1x1_increase")

    conv4_23 = tf.add(conv4_22_relu, conv4_23_1x1_increase, name="conv4_23")
    conv4_23_relu = tf.nn.relu(conv4_23, name="conv4_23_relu")
    conv5_1_1x1_proj = Conv2D_Block(conv4_23_relu, 2048, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                    batch_normalization=True, name="conv5_1_1x1_proj")

    # Stage 5
    conv5_1_1x1_reduce = Conv2D_Block(conv4_23, 512, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv5_1_1x1_reduce")
    padding31 = Zero_Padding(conv5_1_1x1_reduce, paddings=4, name="padding31")
    conv5_1_3x3 = Atrous_Conv2D_Block(padding31, 512, dilation=4, padding='VALID', batch_normalization=True, relu=True,
                                      name="conv5_1_3x3")
    conv5_1_1x1_increase = Conv2D_Block(conv5_1_3x3, 2048, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, name="conv5_1_1x1_increase")

    conv5_1 = tf.add(conv5_1_1x1_proj, conv5_1_1x1_increase, name="conv5_1")
    conv5_1_relu = tf.nn.relu(conv5_1, name="conv5_1_relu")
    conv5_2_1x1_reduce = Conv2D_Block(conv5_1_relu, 512, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv5_2_1_1x1_reduce")
    padding32 = Zero_Padding(conv5_2_1x1_reduce, paddings=4, name="padding32")
    conv5_2_3x3 = Atrous_Conv2D_Block(padding32, 512, dilation=4, padding='VALID', batch_normalization=True, relu=True,
                                      name="conv5_2_3x3")
    conv5_2_1x1_increase = Conv2D_Block(conv5_2_3x3, 2048, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, name="conv5_2_1x1_increase")

    conv5_2 = tf.add(conv5_1_relu, conv5_2_1x1_increase, name="conv5_2")
    conv5_2_relu = tf.nn.relu(conv5_2, name="conv5_2_relu")
    conv5_3_1x1_reduce = Conv2D_Block(conv5_2_relu, 512, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                      batch_normalization=True, relu=True, name="conv5_3_1_1x1_reduce")
    padding33 = Zero_Padding(conv5_3_1x1_reduce, paddings=4, name="padding33")
    conv5_3_3x3 = Atrous_Conv2D_Block(padding33, 512, dilation=4, padding='VALID', batch_normalization=True, relu=True,
                                      name="conv5_3_3x3")
    conv5_3_1x1_increase = Conv2D_Block(conv5_3_3x3, 2048, filter_height=1, filter_width=1, stride=1, padding='VALID',
                                        batch_normalization=True, name="conv5_3_1x1_increase")

    conv5_3 = tf.add(conv5_2_relu, conv5_3_1x1_increase, name="conv5_3")
    conv5_3_relu = tf.nn.relu(conv5_3, name="conv5_3_relu")

    shape = tf.shape(conv5_3_relu)[1:3]
    shape_input = tf.shape(x)[1:3]
    # shape = tf.shape(x)[1:3]
    conv5_3_relu_interp = Resize_Bilinear(conv5_3_relu, shape_input, name="conv5_3_relu_interp")

    # conv5_3_pool1 = Avg_Pooling(conv5_3_relu, filter_height=90, filter_width=90, stride=90, name="conv5_3_pool1")
    conv5_3_pool1 = Avg_Pooling(conv5_3_relu, filter_height=20, filter_width=72, stride_height=20, stride_width=72, name="conv5_3_pool1")
    conv5_3_pool1_conv = Conv2D_Block(conv5_3_pool1, 512, filter_height=1, filter_width=1, stride=1, batch_normalization=True,
                                      relu=True, name="conv5_3_pool1_conv")
    # conv5_3_pool1_interp = Resize_Bilinear(conv5_3_pool1_conv, shape, name="conv5_3_pool1_interp")
    conv5_3_pool1_interp = Resize_Bilinear(conv5_3_pool1_conv, shape_input, name="conv5_3_pool1_interp")

    # conv5_3_pool2 = Avg_Pooling(conv5_3_relu, filter_height=45, filter_width=45, stride=45, name="conv5_3_pool2")
    conv5_3_pool2 = Avg_Pooling(conv5_3_relu, filter_height=10, filter_width=36, stride_height=10, stride_width=36, name="conv5_3_pool2")
    conv5_3_pool2_conv = Conv2D_Block(conv5_3_pool2, 512, filter_height=1, filter_width=1, stride=1, batch_normalization=True,
                                      relu=True, name="conv5_3_pool2_conv")
    # conv5_3_pool2_interp = Resize_Bilinear(conv5_3_pool2_conv, shape, name="conv5_3_pool2_interp")
    conv5_3_pool2_interp = Resize_Bilinear(conv5_3_pool2_conv, shape_input, name="conv5_3_pool2_interp")

    # conv5_3_pool3 = Avg_Pooling(conv5_3_relu, filter_height=30, filter_width=30, stride=30, name="conv5_3_pool3")
    conv5_3_pool3 = Avg_Pooling(conv5_3_relu, filter_height=7, filter_width=24, stride_height=7, stride_width=24, name="conv5_3_pool3")
    conv5_3_pool3_conv = Conv2D_Block(conv5_3_pool3, 512, filter_height=1, filter_width=1, stride=1, batch_normalization=True,
                                      relu=True, name="conv5_3_pool3_conv")
    # conv5_3_pool3_interp = Resize_Bilinear(conv5_3_pool3_conv, shape, name="conv5_3_pool3_interp")
    conv5_3_pool3_interp = Resize_Bilinear(conv5_3_pool3_conv, shape_input, name="conv5_3_pool3_interp")

    # conv5_3_pool6 = Avg_Pooling(conv5_3_relu, filter_height=15, filter_width=15, stride=15, name="conv5_3_pool6")
    conv5_3_pool6 = Avg_Pooling(conv5_3_relu, filter_height=4, filter_width=12, stride_height=4, stride_width=12, name="conv5_3_pool6")
    conv5_3_pool6_conv = Conv2D_Block(conv5_3_pool6, 512, filter_height=1, filter_width=1, stride=1, batch_normalization=True,
                                      relu=True, name="conv5_3_pool6_conv")
    # conv5_3_pool6_interp = Resize_Bilinear(conv5_3_pool6_conv, shape, name="conv5_3_pool6_interp")
    conv5_3_pool6_interp = Resize_Bilinear(conv5_3_pool6_conv, shape_input, name="conv5_3_pool6_interp")

    conv5_3_concat = Concat([conv5_3_relu_interp, conv5_3_pool6_interp, conv5_3_pool3_interp, conv5_3_pool2_interp,
                             conv5_3_pool1_interp], axis=-1, name="conv5_3_concat")

    conv5_4 = Conv2D_Block(conv5_3_concat, 512, batch_normalization=True, relu=True, name="conv5_4")
    conv6 = Conv2D_Block(conv5_4, num_classes, filter_height=1, filter_width=1, stride=1, name="conv6")


    outputs = tf.argmax(conv6, dimension=3, name="prediction")
    print("Build Network done...")

    return tf.expand_dims(outputs, dim=3), conv6

def run():
    # GPU
    tf.debugging.set_log_device_placement(True)
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if gpu:
        try:
            tf.config.experimental.set_memory_growth(gpu[0], True)
        except RuntimeError as e:
            print(e)\

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Training 데이터 셋을 불러옴
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 4], name="input_image")
    prediction = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 2], name="prediction")

    # PSPNet 선언
    # pred_annotation, logits = PSPNet101(image, NUM_OF_CLASSESS)
    pred_annotation, logits = PSPNet50(image, NUM_OF_CLASSESS)

    # Tensorboard를 위한 summary들을 지정
    tf.summary.image('input_image', image, max_outputs=2)
    # 손실 함수를 선언하고 손실 함수에 대한 summary들을 지정
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=prediction))
    tf.summary.scalar('entropy', loss)

    # 옵티마이저를 선언하고 파라미터를 한 스텝 업데이트하는 train_step 연산을 정의
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    # train_step = optimizer.minimize(loss)

    # Constant to scale sum of gradient
    const = tf.constant(1/BATCH_SIZE*3)

    # Get all trainable variables
    t_vars = tf.trainable_variables()

    # Create a copy of all trainable variables with '0' as initial values
    accum_tvars = [tf.Variable(tf.zeros_like(t_var.initialized_value()), trainable=False) for t_var in t_vars]

    # Create a op to initialize all accums vars
    zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvars]

    # Compute gradients for a batch
    batch_grads_vars = optimizer.compute_gradients(loss, t_vars)

    # Collect the (scaled by const) batch gradient into accumulated vars
    accum_ops = [accum_tvars[i].assign_add(tf.scalar_mul(const, batch_grad_var[0])) for i, batch_grad_var in enumerate(batch_grads_vars)]

    # Apply accums gradients
    train_step = optimizer.apply_gradients([(accum_tvars[i], batch_grad_var[1]) for i, batch_grad_var in enumerate(batch_grads_vars)])

    # Tensorboard를 위한 summary를 하나로 merge
    print("Setting up summary up")
    summary_op = tf.summary.merge_all()

    # training 데이터와 validation 데이터 개수를 불러옴.
    training_labels_count = len(glob(os.path.join(DATA_DIR, 'training/gt_image_2/*_road_*.png')))
    training_images_count = len(glob(os.path.join(DATA_DIR, 'training/merge/*.png')))
    testing_images_count = len(glob(os.path.join(DATA_DIR, 'testing/merge/*.png')))

    assert not (training_images_count == training_labels_count == testing_images_count == 0), \
        'Kitti dataset not found. Extract Kitti dataset in {}'.format(DATA_DIR)
    assert training_images_count == 259, 'Expected 289 training images, found {} images.'.format(
        training_images_count)  # 289
    assert training_labels_count == 259, 'Expected 289 training labels, found {} labels.'.format(
        training_labels_count)  # 289
    assert testing_images_count == 290, 'Expected 290 testing images, found {} images.'.format(testing_images_count)

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

    start = time.time()  # 시작 시간 저장
    for epoch in range(EPOCHS):
        s_time = time.time()
        # 학습 데이터를 불러오고 feed_dict에 데이터를 지정
        for images, labels in get_batches_fn(batch_size=BATCH_SIZE):
            feed_dict = {image: images, prediction: labels, keep_probability: KEEP_PROB}

            # Initialize the accumulated grads
            sess.run(zero_ops)
            for i in range(len(images)):
                sess.run(accum_ops, feed_dict=feed_dict)

            # train_step을 실행해서 파라미터를 한 스텝 업데이트 함
            sess.run(train_step, feed_dict=feed_dict)

            # Tensorboard 를 위한 sess.run()
            summary = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step=sess.run(global_step))

        print("[Epoch: {0}/{1} Time: {2}]".format(epoch + 1, EPOCHS, str(timedelta(seconds=(time.time() - s_time)))))

    print("Time: ", time.time() - start)  # 현재 시각 - 시작 시간 = 실행 시간
    print("Training Successfully")

    # 훈련이 끝나고 학습된 파라미터 저장
    saver.save(sess, './model/PSPNet.ckpt', global_step=global_step)

    # 훈련이 끝나고 테스트 데이터 셋으로 테스트
    output_dir = os.path.join(DATA_DIR, 'output')
    mask_dir = os.path.join(DATA_DIR, 'mask')
    print("Training Finished. Saving test images to: {}".format(output_dir))
    image_output = gen_test_output(sess, logits, keep_probability, image, os.path.join(DATA_DIR, 'validating'),
                                   IMAGE_SHAPE_KITTI)

    total_processing_time = 0
    for name, mask, image, processing_time in image_output:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        scipy.misc.imsave(os.path.join(mask_dir, name), mask)
        total_processing_time += processing_time
        # cv.imwrite(os.path.join(output_dir, name), image)
        # cv.imwrite(os.path.join(mask_dir, name), mask)
    print("Average processing time is : ", total_processing_time / 30)

if __name__ == '__main__':
    run()
