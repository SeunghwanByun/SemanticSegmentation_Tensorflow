import cv2 as cv
import tensorflow as tf
import numpy as np
import scipy.misc
from datetime import timedelta
import os
import time
from glob import glob

from Utils_TransNet import *

# 학습에 필요한 설정값들을 지정
KEEP_PROB = 0.8
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

def TransNet(x_img, x_lid, out_ch, lamda=0.1, name=None):
    concat = Concat([x_img, x_lid], axis=-1, name=name + "_concat")

    alpha = Conv2D_Block(concat, out_ch, filter_height=1, filter_width=1, stride=1, padding='VALID', name=name+"_alpha") + 1
    beta = Conv2D_Block(concat, out_ch, filter_height=1, filter_width=1, stride=1, padding="VALID", name=name+"_beta")

    lid_feat = Conv2D_Block(x_lid, out_ch, filter_height=1, filter_width=1, stride=1, padding="VALID", name=name+"_lid_feat")

    lid_feat = alpha * lid_feat + beta
    lid_feat = ReLU(lid_feat)

    fuse = Conv2D_Block(lid_feat, out_ch, name=name+"_fuse")

    fuse = x_img + lamda * fuse
    fuse = ReLU(fuse)

    return fuse

def PSPNet50(x_img, x_lid, keep_prob, num_classes):
    shape = tf.shape(x_img)[1:3]

    # Network 구현
    print("Network Build Start...")
    # Stage 1
    # rgb feature map
    conv1_img = Zero_Padding(x_img, paddings=3, name="conv1_img")
    conv1_img = Conv2D_Block(conv1_img, 64, 7, 7, 2, 'VALID', True, True, name="conv1_img")
    conv1_img = Zero_Padding(conv1_img, paddings=1, name="conv1_img")

    pool1_img = Max_Pooling(conv1_img, filter_height=3, filter_width=3, stride=2, name="pool1_img")

    # lidar feature map
    conv1_lid = Zero_Padding(x_lid, paddings=3, name="conv1_lid")
    conv1_lid = Conv2D_Block(conv1_lid, 64, 7, 7, 2, 'VALID', True, True, name="conv1_lid")
    conv1_lid = Zero_Padding(conv1_lid, paddings=1, name="conv1_lid")

    pool1_lid = Max_Pooling(conv1_lid, filter_height=3, filter_width=3, stride=2, name="pool1_lid")

    # Stage 2
    # image feature map
    conv2_1_1_img = Conv2D_Block(pool1_img, 64, 1, 1, 1, 'VALID', True, True, name="conv2_1_1_img")
    conv2_1_2_img = Conv2D_Block(conv2_1_1_img, 64, 3, 3, 1, 'SAME', True, True, name="conv2_1_2_img")
    conv2_1_3_img = Conv2D_Block(conv2_1_2_img, 256, 1, 1, 1, 'VALID', True, name="conv2_1_3_img")
    shortcut1_img = Conv2D_Block(pool1_img, 256, 1, 1, 1, 'VALID', True, name="shortcut1_img")
    conv2_1_img = tf.add(conv2_1_3_img, shortcut1_img, name="conv2_1_img")
    conv2_1_img = tf.nn.relu(conv2_1_img)

    conv2_2_1_img = Conv2D_Block(conv2_1_img, 64, 1, 1, 1, 'VALID', True, True, name="conv2_2_1_img")
    conv2_2_2_img = Conv2D_Block(conv2_2_1_img, 64, 3, 3, 1, 'SAME', True, True, name="conv2_2_2_img")
    conv2_2_3_img = Conv2D_Block(conv2_2_2_img, 256, 1, 1, 1, 'VALID', True, name="conv2_2_3_img")
    conv2_2_img = tf.add(conv2_2_3_img, conv2_1_img, name="conv2_2_img")
    conv2_2_img = tf.nn.relu(conv2_2_img)

    conv2_3_1_img = Conv2D_Block(conv2_2_img, 64, 1, 1, 1, 'VALID', True, True, name="conv2_3_1_img")
    conv2_3_2_img = Conv2D_Block(conv2_3_1_img, 64, 3, 3, 1, 'SAME', True, True, name="conv2_3_2_img")
    conv2_3_3_img = Conv2D_Block(conv2_3_2_img, 256, 1, 1, 1, 'SAME', True, name="conv2_3_3_img")
    conv2_3_img = tf.add(conv2_3_3_img, conv2_2_img, name="conv2_3_img")
    conv2_3_img = tf.nn.relu(conv2_3_img)

    # lidar feature map
    conv2_1_1_lid = Conv2D_Block(pool1_lid, 64, 1, 1, 1, 'VALID', True, True, name="conv2_1_1_lid")
    conv2_1_2_lid = Conv2D_Block(conv2_1_1_lid, 64, 3, 3, 1, 'SAME', True, True, name="conv2_1_2_lid")
    conv2_1_3_lid = Conv2D_Block(conv2_1_2_lid, 256, 1, 1, 1, 'VALID', True, name="conv2_1_3_lid")
    shortcut1_lid = Conv2D_Block(pool1_lid, 256, 1, 1, 1, 'VALID', True, name="shortcut1_lid")
    conv2_1_lid = tf.add(conv2_1_3_lid, shortcut1_lid, name="conv2_1_lid")
    conv2_1_lid = tf.nn.relu(conv2_1_lid)

    conv2_2_1_lid = Conv2D_Block(conv2_1_lid, 64, 1, 1, 1, 'VALID', True, True, name="conv2_2_1_lid")
    conv2_2_2_lid = Conv2D_Block(conv2_2_1_lid, 64, 3, 3, 1, 'SAME', True, True, name="conv2_2_2_lid")
    conv2_2_3_lid = Conv2D_Block(conv2_2_2_lid, 256, 1, 1, 1, 'VALID', True, name="conv2_2_3_lid")
    conv2_2_lid = tf.add(conv2_2_3_lid, conv2_1_lid, name="conv2_2_lid")
    conv2_2_lid = tf.nn.relu(conv2_2_lid)

    conv2_3_1_lid = Conv2D_Block(conv2_2_lid, 64, 1, 1, 1, 'VALID', True, True, name="conv2_3_1_lid")
    conv2_3_2_lid = Conv2D_Block(conv2_3_1_lid, 64, 3, 3, 1, 'SAME', True, True, name="conv2_3_2_lid")
    conv2_3_3_lid = Conv2D_Block(conv2_3_2_lid, 256, 1, 1, 1, 'SAME', True, name="conv2_3_3_lid")
    conv2_3_lid = tf.add(conv2_3_3_lid, conv2_2_lid, name="conv2_3_lid")
    conv2_3_lid = tf.nn.relu(conv2_3_lid)

    # TransNet
    out_ch = conv2_3_lid.get_shape()[-1]
    trans1 = TransNet(conv2_3_img, conv2_3_lid, out_ch, name="trans1")

    # Stage 3
    # rgb feature map
    conv3_1_1_img = Conv2D_Block(trans1, 128, 1, 1, 2, 'VALID', True, True, name="conv3_1_img")
    conv3_1_2_img = Conv2D_Block(conv3_1_1_img, 128, 3, 3, 1, 'SAME', True, True, name="conv3_2_img")
    conv3_1_3_img = Conv2D_Block(conv3_1_2_img, 256, 1, 1, 1, 'VALID', True, name="conv3_3_img")
    shortcut2_img = Conv2D_Block(conv2_3_img, 256, 1, 1, 2, 'VALID', True, name="shortcut2_img")
    conv3_1_img = tf.add(conv3_1_3_img, shortcut2_img, name="conv3_1_img")
    conv3_1_img = tf.nn.relu(conv3_1_img)

    conv3_2_1_img = Conv2D_Block(conv3_1_img, 128, 1, 1, 1, 'VALID', True, True, name="conv3_2_1_img")
    conv3_2_2_img = Conv2D_Block(conv3_2_1_img, 128, 3, 3, 1, 'SAME', True, True, name="conv3_2_2_img")
    conv3_2_3_img = Conv2D_Block(conv3_2_2_img, 256, 1, 1, 1, 'VALID', True, name="conv3_2_3_img")
    conv3_2_img = tf.add(conv3_2_3_img, conv3_1_img, name="conv3_2_img")
    conv3_2_img = tf.nn.relu(conv3_2_img)

    conv3_3_1_img = Conv2D_Block(conv3_2_img, 128, 1, 1, 1, 'VALID', True, True, name="conv3_3_1_img")
    conv3_3_2_img = Conv2D_Block(conv3_3_1_img, 128, 3, 3, 1, 'SAME', True, True, name="conv3_3_2_img")
    conv3_3_3_img = Conv2D_Block(conv3_3_2_img, 256, 1, 1, 1, 'VALID', True, name="conv3_3_3_img")
    conv3_3_img = tf.add(conv3_3_3_img, conv3_1_img, name="conv3_3_img")
    conv3_3_img = tf.nn.relu(conv3_3_img)

    conv3_4_1_img = Conv2D_Block(conv3_3_img, 128, 1, 1, 1, 'VALID', True, True, name="conv3_4_1_img")
    conv3_4_2_img = Conv2D_Block(conv3_4_1_img, 128, 3, 3, 1, 'SAME', True, True, name="conv3_4_2_img")
    conv3_4_3_img = Conv2D_Block(conv3_4_2_img, 256, 1, 1, 1, 'VALID', True, name="conv3_4_3_img")
    conv3_4_img = tf.add(conv3_4_3_img, conv3_1_img, name="conv3_4_img")
    conv3_4_img = tf.nn.relu(conv3_4_img)

    # lidar feature map
    conv3_1_1_lid = Conv2D_Block(conv2_3_lid, 128, 1, 1, 2, 'VALID', True, True, name="conv3_1_lid")
    conv3_1_2_lid = Conv2D_Block(conv3_1_1_lid, 128, 3, 3, 1, 'SAME', True, True, name="conv3_2_lid")
    conv3_1_3_lid = Conv2D_Block(conv3_1_2_lid, 256, 1, 1, 1, 'VALID', True, name="conv3_3_lid")
    shortcut2_lid = Conv2D_Block(conv2_3_lid, 256, 1, 1, 2, 'VALID', True, name="shortcut2_lid")
    conv3_1_lid = tf.add(conv3_1_3_lid, shortcut2_lid, name="conv3_1_lid")
    conv3_1_lid = tf.nn.relu(conv3_1_lid)

    conv3_2_1_lid = Conv2D_Block(conv3_1_lid, 128, 1, 1, 1, 'VALID', True, True, name="conv3_2_1_lid")
    conv3_2_2_lid = Conv2D_Block(conv3_2_1_lid, 128, 3, 3, 1, 'SAME', True, True, name="conv3_2_2_lid")
    conv3_2_3_lid = Conv2D_Block(conv3_2_2_lid, 256, 1, 1, 1, 'VALID', True, name="conv3_2_3_lid")
    conv3_2_lid = tf.add(conv3_2_3_lid, conv3_1_lid, name="conv3_2_lid")
    conv3_2_lid = tf.nn.relu(conv3_2_lid)

    conv3_3_1_lid = Conv2D_Block(conv3_2_lid, 128, 1, 1, 1, 'VALID', True, True, name="conv3_3_1_lid")
    conv3_3_2_lid = Conv2D_Block(conv3_3_1_lid, 128, 3, 3, 1, 'SAME', True, True, name="conv3_3_2_lid")
    conv3_3_3_lid = Conv2D_Block(conv3_3_2_lid, 256, 1, 1, 1, 'VALID', True, name="conv3_3_3_lid")
    conv3_3_lid = tf.add(conv3_3_3_lid, conv3_1_lid, name="conv3_3_lid")
    conv3_3_lid = tf.nn.relu(conv3_3_lid)

    conv3_4_1_lid = Conv2D_Block(conv3_3_lid, 128, 1, 1, 1, 'VALID', True, True, name="conv3_4_1_lid")
    conv3_4_2_lid = Conv2D_Block(conv3_4_1_lid, 128, 3, 3, 1, 'SAME', True, True, name="conv3_4_2_lid")
    conv3_4_3_lid = Conv2D_Block(conv3_4_2_lid, 256, 1, 1, 1, 'VALID', True, name="conv3_4_3_lid")
    conv3_4_lid = tf.add(conv3_4_3_lid, conv3_1_lid, name="conv3_4_lid")
    conv3_4_lid = tf.nn.relu(conv3_4_lid)

    out_ch = conv3_4_lid.get_shape()[-1]
    trans2 = TransNet(conv3_4_img, conv3_4_lid, out_ch, name="trans2")

    # Stage 4
    # rgb feature map
    conv4_1_1_img = Conv2D_Block(trans2, 256, 1, 1, 2, 'VALID', True, True, name="conv4_1_1_img")
    conv4_1_2_img = Conv2D_Block(conv4_1_1_img, 256, 3, 3, 1, 'SAME', True, True, name="conv4_1_2_img")
    conv4_1_3_img = Conv2D_Block(conv4_1_2_img, 512, 1, 1, 1, 'VALID', True, name="conv4_1_3_img")
    shortcut3_img = Conv2D_Block(conv3_4_img, 512, 1, 1, 2, 'VALID', True, name="shortcut3_img")
    conv4_1_img = tf.add(conv4_1_3_img, shortcut3_img, name="conv4_1_img")
    conv4_1_img = tf.nn.relu(conv4_1_img)

    conv4_2_1_img = Conv2D_Block(conv4_1_img, 256, 1, 1, 1, 'VALID', True, True, name="conv4_2_1_img")
    conv4_2_2_img = Conv2D_Block(conv4_2_1_img, 256, 3, 3, 1, 'SAME', True, True, name="conv4_2_2_img")
    conv4_2_3_img = Conv2D_Block(conv4_2_2_img, 512, 1, 1, 1, 'VALID', True, name="conv4_2_3_img")
    conv4_2_img = tf.add(conv4_2_3_img, conv4_1_img, name="conv4_2_img")
    conv4_2_img = tf.nn.relu(conv4_2_img)

    conv4_3_1_img = Conv2D_Block(conv4_2_img, 256, 1, 1, 1, 'VALID', True, True, name="conv4_3_1_img")
    conv4_3_2_img = Conv2D_Block(conv4_3_1_img, 256, 3, 3, 1, 'SAME', True, True, name="conv4_3_2_img")
    conv4_3_3_img = Conv2D_Block(conv4_3_2_img, 512, 1, 1, 1, 'VALID', True, name="conv4_3_3_img")
    conv4_3_img = tf.add(conv4_3_3_img, conv4_2_img, name="conv4_3_img")
    conv4_3_img = tf.nn.relu(conv4_3_img)

    conv4_4_1_img = Conv2D_Block(conv4_3_img, 256, 1, 1, 1, 'VALID', True, True, name="conv4_4_1_img")
    conv4_4_2_img = Conv2D_Block(conv4_4_1_img, 256, 3, 3, 1, 'SAME', True, True, name="conv4_4_2_img")
    conv4_4_3_img = Conv2D_Block(conv4_4_2_img, 512, 1, 1, 1, 'VALID', True, name="conv4_4_3_img")
    conv4_4_img = tf.add(conv4_4_3_img, conv4_3_img, name="conv4_4_img")
    conv4_4_img = tf.nn.relu(conv4_4_img)

    conv4_5_1_img = Conv2D_Block(conv4_4_img, 256, 1, 1, 1, 'VALID', True, True, name="conv4_5_1_img")
    conv4_5_2_img = Conv2D_Block(conv4_5_1_img, 256, 3, 3, 1, 'SAME', True, True, name="conv4_5_2_img")
    conv4_5_3_img = Conv2D_Block(conv4_5_2_img, 512, 1, 1, 1, 'VALID', True, name="conv4_5_3_img")
    conv4_5_img = tf.add(conv4_5_3_img, conv4_4_img, name="conv4_5_img")
    conv4_5_img = tf.nn.relu(conv4_5_img)

    conv4_6_1_img = Conv2D_Block(conv4_5_img, 256, 1, 1, 1, 'VALID', True, True, name="conv4_6_1_img")
    conv4_6_2_img = Conv2D_Block(conv4_6_1_img, 256, 3, 3, 1, 'SAME', True, True, name="conv4_6_2_img")
    conv4_6_3_img = Conv2D_Block(conv4_6_2_img, 512, 1, 1, 1, 'VALID', True, name="conv4_6_3_img")
    conv4_6_img = tf.add(conv4_6_3_img, conv4_5_img, name="conv4_6_img")
    conv4_6_img = tf.nn.relu(conv4_6_img)

    # lidar feature map
    conv4_1_1_lid = Conv2D_Block(conv3_4_lid, 256, 1, 1, 2, 'VALID', True, True, name="conv4_1_1_lid")
    conv4_1_2_lid = Conv2D_Block(conv4_1_1_lid, 256, 3, 3, 1, 'SAME', True, True, name="conv4_1_2_lid")
    conv4_1_3_lid = Conv2D_Block(conv4_1_2_lid, 512, 1, 1, 1, 'VALID', True, name="conv4_1_3_lid")
    shortcut3_lid = Conv2D_Block(conv3_4_lid, 512, 1, 1, 2, 'VALID', True, name="shortcut3_lid")
    conv4_1_lid = tf.add(conv4_1_3_lid, shortcut3_lid, name="conv4_1_lid")
    conv4_1_lid = tf.nn.relu(conv4_1_lid)

    conv4_2_1_lid = Conv2D_Block(conv4_1_lid, 256, 1, 1, 1, 'VALID', True, True, name="conv4_2_1_lid")
    conv4_2_2_lid = Conv2D_Block(conv4_2_1_lid, 256, 3, 3, 1, 'SAME', True, True, name="conv4_2_2_lid")
    conv4_2_3_lid = Conv2D_Block(conv4_2_2_lid, 512, 1, 1, 1, 'VALID', True, name="conv4_2_3_lid")
    conv4_2_lid = tf.add(conv4_2_3_lid, conv4_1_lid, name="conv4_2_lid")
    conv4_2_lid = tf.nn.relu(conv4_2_lid)

    conv4_3_1_lid = Conv2D_Block(conv4_2_lid, 256, 1, 1, 1, 'VALID', True, True, name="conv4_3_1_lid")
    conv4_3_2_lid = Conv2D_Block(conv4_3_1_lid, 256, 3, 3, 1, 'SAME', True, True, name="conv4_3_2_lid")
    conv4_3_3_lid = Conv2D_Block(conv4_3_2_lid, 512, 1, 1, 1, 'VALID', True, name="conv4_3_3_lid")
    conv4_3_lid = tf.add(conv4_3_3_lid, conv4_2_lid, name="conv4_3_lid")
    conv4_3_lid = tf.nn.relu(conv4_3_lid)

    conv4_4_1_lid = Conv2D_Block(conv4_3_lid, 256, 1, 1, 1, 'VALID', True, True, name="conv4_4_1_lid")
    conv4_4_2_lid = Conv2D_Block(conv4_4_1_lid, 256, 3, 3, 1, 'SAME', True, True, name="conv4_4_2_lid")
    conv4_4_3_lid = Conv2D_Block(conv4_4_2_lid, 512, 1, 1, 1, 'VALID', True, name="conv4_4_3_lid")
    conv4_4_lid = tf.add(conv4_4_3_lid, conv4_3_lid, name="conv4_4_lid")
    conv4_4_lid = tf.nn.relu(conv4_4_lid)

    conv4_5_1_lid = Conv2D_Block(conv4_4_lid, 256, 1, 1, 1, 'VALID', True, True, name="conv4_5_1_lid")
    conv4_5_2_lid = Conv2D_Block(conv4_5_1_lid, 256, 3, 3, 1, 'SAME', True, True, name="conv4_5_2_lid")
    conv4_5_3_lid = Conv2D_Block(conv4_5_2_lid, 512, 1, 1, 1, 'VALID', True, name="conv4_5_3_lid")
    conv4_5_lid = tf.add(conv4_5_3_lid, conv4_4_lid, name="conv4_5_lid")
    conv4_5_lid = tf.nn.relu(conv4_5_lid)

    conv4_6_1_lid = Conv2D_Block(conv4_5_lid, 256, 1, 1, 1, 'VALID', True, True, name="conv4_6_1_lid")
    conv4_6_2_lid = Conv2D_Block(conv4_6_1_lid, 256, 3, 3, 1, 'SAME', True, True, name="conv4_6_2_lid")
    conv4_6_3_lid = Conv2D_Block(conv4_6_2_lid, 512, 1, 1, 1, 'VALID', True, name="conv4_6_3_lid")
    conv4_6_lid = tf.add(conv4_6_3_lid, conv4_5_lid, name="conv4_6_lid")
    conv4_6_lid = tf.nn.relu(conv4_6_lid)

    out_ch = conv4_6_lid.get_shape()[-1]
    trans3 = TransNet(conv4_6_img, conv4_6_lid, out_ch, name="trans3")

    aux_loss = Conv2D_Block(trans3, 256, 3, 3, 1, 'SAME', batch_normalization=True, relu=True, name="aux_loss")
    aux_loss = Dropout(aux_loss, keep_prob=keep_prob)
    aux_loss_interp = Resize_Bilinear(aux_loss, shape, name="aux_loss_interp")
    aux_classifier = Conv2D_Block(aux_loss_interp, NUM_OF_CLASSESS, filter_height=1, filter_width=1, stride=1, name="aux_classifier")

    # Stage 5
    # rgb feature map
    conv5_1_1_img = Conv2D_Block(trans3, 512, 1, 1, 2, 'VALID', True, True, name="conv5_1_1_img")
    print("conv5_1_1.get_shape()", conv5_1_1_img.get_shape())
    conv5_1_2_img = Conv2D_Block(conv5_1_1_img, 512, 3, 3, 1, 'SAME', True, True, name="conv5_1_2_img")
    conv5_1_3_img = Conv2D_Block(conv5_1_2_img, 1024, 1, 1, 1, 'VALID', True, name="conv5_1_3_img")
    shortcut4_img = Conv2D_Block(conv4_6_img, 1024, 1, 1, 2, 'VALID', True, name="shortcut4_img")
    conv5_1_img = tf.add(conv5_1_3_img, shortcut4_img, name="conv5_1_img")
    conv5_1_img = tf.nn.relu(conv5_1_img)

    conv5_2_1_img = Conv2D_Block(conv5_1_img, 512, 1, 1, 1, 'VALID', True, True, name="conv5_2_1_img")
    conv5_2_2_img = Conv2D_Block(conv5_2_1_img, 512, 3, 3, 1, 'SAME', True, True, name="conv5_2_2_img")
    conv5_2_3_img = Conv2D_Block(conv5_2_2_img, 1024, 1, 1, 1, 'VALID', True, name="conv5_2_3_img")
    conv5_2_img = tf.add(conv5_2_3_img, conv5_1_img, name="conv5_2_img")
    conv5_2_img = tf.nn.relu(conv5_2_img)

    conv5_3_1_img = Conv2D_Block(conv5_2_img, 512, 1, 1, 1, 'VALID', True, True, name="conv5_3_1_img")
    conv5_3_2_img = Conv2D_Block(conv5_3_1_img, 512, 3, 3, 1, 'SAME', True, True, name="conv5_3_2_img")
    conv5_3_3_img = Conv2D_Block(conv5_3_2_img, 1024, 1, 1, 1, 'VALID', True, name="conv5_3_3_img")
    conv5_3_img = tf.add(conv5_3_3_img, conv5_1_img, name="conv5_3_img")
    conv5_3_img = tf.nn.relu(conv5_3_img)

    # lidar feature map
    conv5_1_1_lid = Conv2D_Block(conv4_6_lid, 512, 1, 1, 2, 'VALID', True, True, name="conv5_1_1_lid")
    print("conv5_1_1.get_shape()", conv5_1_1_lid.get_shape())
    conv5_1_2_lid = Conv2D_Block(conv5_1_1_lid, 512, 3, 3, 1, 'SAME', True, True, name="conv5_1_2_lid")
    conv5_1_3_lid = Conv2D_Block(conv5_1_2_lid, 1024, 1, 1, 1, 'VALID', True, name="conv5_1_3_lid")
    shortcut4_lid = Conv2D_Block(conv4_6_lid, 1024, 1, 1, 2, 'VALID', True, name="shortcut4_lid")
    conv5_1_lid = tf.add(conv5_1_3_lid, shortcut4_lid, name="conv5_1_lid")
    conv5_1_lid = tf.nn.relu(conv5_1_lid)

    conv5_2_1_lid = Conv2D_Block(conv5_1_lid, 512, 1, 1, 1, 'VALID', True, True, name="conv5_2_1_lid")
    conv5_2_2_lid = Conv2D_Block(conv5_2_1_lid, 512, 3, 3, 1, 'SAME', True, True, name="conv5_2_2_lid")
    conv5_2_3_lid = Conv2D_Block(conv5_2_2_lid, 1024, 1, 1, 1, 'VALID', True, name="conv5_2_3_lid")
    conv5_2_lid = tf.add(conv5_2_3_lid, conv5_1_lid, name="conv5_2_lid")
    conv5_2_lid = tf.nn.relu(conv5_2_lid)

    conv5_3_1_lid = Conv2D_Block(conv5_2_lid, 512, 1, 1, 1, 'VALID', True, True, name="conv5_3_1_lid")
    conv5_3_2_lid = Conv2D_Block(conv5_3_1_lid, 512, 3, 3, 1, 'SAME', True, True, name="conv5_3_2_lid")
    conv5_3_3_lid = Conv2D_Block(conv5_3_2_lid, 1024, 1, 1, 1, 'VALID', True, name="conv5_3_3_lid")
    conv5_3_lid = tf.add(conv5_3_3_lid, conv5_1_lid, name="conv5_3_lid")
    conv5_3_lid = tf.nn.relu(conv5_3_lid)

    lid_aux_classifier = Dropout(conv5_3_lid, keep_prob=keep_prob)
    lid_aux_classifier = Resize_Bilinear(lid_aux_classifier, shape, name="lid_aux_classifier_interp")
    lid_aux_classifier = Conv2D_Block(lid_aux_classifier, NUM_OF_CLASSESS, filter_height=1, filter_width=1, stride=1, name="lid_aux_loss")

    out_ch = conv5_3_lid.get_shape()[-1]
    trans4 = TransNet(conv5_3_img, conv5_3_lid, out_ch, name="trans4")

    encoder_output = Conv2D_Block(trans4, 512, 1, 1, 1, 'SAME', name='encoder_output')

    print("Encoder Build Finished...")
    encoder_interp = Resize_Bilinear(encoder_output, shape, name="conv5_3_interp")

    pool1 = Avg_Pooling(encoder_output, filter_height=4, filter_width=10, stride_height=4, stride_width=10,
                               name="conv5_3_pool1")
    pool1_conv = Conv2D_Block(pool1, 128, filter_height=1, filter_width=1, stride=1,
                                      batch_normalization=True, relu=True, name="conv5_3_pool1_conv")
    pool1_interp = Resize_Bilinear(pool1_conv, shape, name="conv5_3_pool1_interp")

    pool2 = Avg_Pooling(encoder_output, filter_height=3, filter_width=9, stride_height=2, stride_width=8,
                                name="conv5_3_pool2")
    pool2_conv = Conv2D_Block(pool2, 128, filter_height=1, filter_width=1, stride=1,
                                      batch_normalization=True, relu=True, name="conv5_3_pool2_conv")
    pool2_interp = Resize_Bilinear(pool2_conv, shape, name="conv5_3_pool2_interp")

    pool3 = Avg_Pooling(encoder_output, filter_height=2, filter_width=5, stride_height=1, stride_width=3,
                                name="conv5_3_pool3")
    pool3_conv = Conv2D_Block(pool3, 128, filter_height=1, filter_width=1, stride=1,
                                      batch_normalization=True, relu=True, name="conv5_3_pool3_conv")
    pool3_interp = Resize_Bilinear(pool3_conv, shape, name="conv5_3_pool3_interp")

    pool4 = Avg_Pooling(encoder_output, filter_height=1, filter_width=3, stride_height=1, stride_width=3,
                                name="conv5_3_pool6")

    pool4_conv = Conv2D_Block(pool4, 128, filter_height=1, filter_width=1, stride=1,
                                      batch_normalization=True, relu=True, name="conv5_3_pool6_conv")
    pool4_interp = Resize_Bilinear(pool4_conv, shape, name="conv5_3_pool6_interp")

    concatenation = Concat([encoder_interp, pool1_interp, pool2_interp, pool3_interp, pool4_interp], axis=-1,
                           name="concatenation")

    conv5_4 = Conv2D_Block(concatenation, 128, batch_normalization=True, relu=True, name="conv5_4")
    conv6 = Conv2D_Block(conv5_4, num_classes, filter_height=1, filter_width=1, stride=1, name="conv6")

    prediction = tf.argmax(conv6, dimension=3, name="prediction")
    print("Builde Network done...")


    return tf.expand_dims(prediction, dim=3), conv6, lid_aux_classifier, aux_classifier


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
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 3], name="input_image")
    adi = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 1], name="input_adi")
    prediction = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 2],
                                name="prediction")

    # Network 선언
    pred_annotation, logits, lid_logits, aux_logits = PSPNet50(image, adi, keep_probability, NUM_OF_CLASSESS)

    # Tensorboard를 위한 summary들을 지정
    tf.summary.image('input_image', image, max_outputs=2)
    # 손실 함수를 선언하고 손실 함수에 대한 summary들을 지정
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=prediction))
    training_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=prediction)
                                   + 0.4 * tf.nn.softmax_cross_entropy_with_logits(logits=lid_logits, labels=prediction)
                                   + 1.6 * tf.nn.softmax_cross_entropy_with_logits(logits=aux_logits, labels=prediction))
    tf.summary.scalar('entropy', loss)

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
    batch_grads_vars = optimizer.compute_gradients(training_loss, t_vars)

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
    testing_images_count = len(glob(os.path.join(DATA_DIR, 'testing/image_2/*.png')))
    training_ADI_count = len(glob(os.path.join(DATA_DIR, 'training/ADI/*.png')))
    testing_ADI_count = len(glob(os.path.join(DATA_DIR, 'testing/ADI/*.png')))

    assert not (training_images_count == training_labels_count == testing_images_count == 0), \
        'Kitti dataset not found. Extract Kitti dataset in {}'.format(DATA_DIR)
    assert training_images_count == 259, 'Expected 259 training images, found {} images.'.format(
        training_images_count)  # 289
    assert training_labels_count == 259, 'Expected 259 training labels, found {} labels.'.format(
        training_labels_count)  # 289
    assert testing_images_count == 290, 'Expected 290 testing images, found {} images.'.format(
        testing_images_count)
    assert training_ADI_count == 259, 'Expected 259 training ADI, found {} images.'.format(
        training_ADI_count)
    assert testing_ADI_count == 290, 'Expected 290 testing ADI, found {} images.'.format(
        testing_images_count)

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

    # f = open('./loss.csv', 'w', newline='')
    # makewrite = csv.writer(f)

    start = time.time()  # 시작 시간 저장
    for epoch in range(EPOCHS):
        s_time = time.time()
        # 학습 데이터를 불러오고 feed_dict에 데이터를 지정
        for images, adis, labels in get_batches_fn(batch_size=BATCH_SIZE):
            feed_dict = {image: images, adi: adis, prediction: labels, keep_probability: KEEP_PROB}

            # Initialize the accumulated grads
            sess.run(zero_ops)
            for i in range(len(images)):
                sess.run(accum_ops, feed_dict=feed_dict)

            # train_step을 실행해서 파라미터를 한 스텝 업데이트 함
            _, cost = sess.run([train_step, loss], feed_dict=feed_dict)

            # makewrite.writerow(cost)

            # Tensorboard 를 위한 sess.run()
            summary = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step=sess.run(global_step))

        print("[Epoch: {0}/{1} Time: {2}]".format(epoch + 1, EPOCHS, str(timedelta(seconds=(time.time() - s_time)))))

    print("Time: ", time.time() - start)  # 현재 시각 - 시작 시간 = 실행 시간
    print("Training Successfully")

    # 훈련이 끝나고 학습된 파라미터 저장
    saver.save(sess, './model/PSPNet50_TransNet.ckpt', global_step=global_step)

    # 훈련이 끝나고 테스트 데이터 셋으로 테스트
    output_dir = os.path.join(DATA_DIR, 'output')
    mask_dir = os.path.join(DATA_DIR, 'mask')
    print("Training Finished. Saving test images to: {}".format(output_dir))
    image_output = gen_test_output(sess, logits, keep_probability, image, adi, os.path.join(DATA_DIR, 'validating'),
                                   IMAGE_SHAPE_KITTI)

    total_processing_time = 0
    for name, mask, image, processing_time in image_output:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        scipy.misc.imsave(os.path.join(mask_dir, name), mask)
        total_processing_time += processing_time

    print("Average processing time is : ", total_processing_time / 30)


if __name__ == '__main__':
    run()