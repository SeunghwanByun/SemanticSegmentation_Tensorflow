import os
import re
import time
import numpy as np
import random
import scipy.misc
import tensorflow as tf
from glob import glob
from tflearn.layers.conv import global_avg_pool, global_max_pool
import cv2 as cv

def img_size(image):
    return image.shape[0], image.shape[1]

def crop_image(image, adi, gt_image):
    h, w = img_size(image)
    nw = random.randint(1150, w-5) # Random crop size
    nh = int(nw / 3.3) # Keep original aspect ration
    x1 = random.randint(0, w - nw) # Random position of crop
    y1 = random.randint(0, h - nh)

    return image[y1:(y1 + nh), x1:(x1 + nw), :], adi[y1:(y1 + nh), x1:(x1 + nw)], gt_image[y1:(y1 + nh), x1:(x1 + nw), :]

def flip_image(image, adi, gt_image):
    return np.flip(image, axis=1), np.flip(adi, axis=1), np.flip(gt_image, axis=1)

def bc_img(img, s=1, m=0.0):
    img = img.astype(np.int)
    img = img * s + m
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img

def process_gt_image(gt_image):
    background_color = np.array([255, 0, 0])
    # background_color = np.array([0, 0, 255])
    gt_bg = np.all(gt_image == background_color, axis=2)
    gt_bg = gt_bg.reshape(gt_bg.shape[0], gt_bg.shape[1], 1)

    gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
    return gt_image

def paste_mask(street_im, im_soft_max, image_shape, color, obj_color_schema):
    """
        Generate labeled images to test images
    :param street_im: test raw image
    :param im_soft_max: image after softmax function
    :param image_shape: Tuple - Shape of image
    :param color:
    :param obj_color_schema:
    :return:
    """
    # zero images
    im_soft_max_r = np.squeeze(im_soft_max[0])[:, :, color].reshape(image_shape[0], image_shape[1])
    segmentation_r = (im_soft_max_r > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation_r, np.array(obj_color_schema))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im.paste(mask, box=None, mask=mask)


    # return street_im
    return mask, street_im

def gen_test_output(sess, logits, keep_prob, image_pl, adi_pl, data_folder, image_shape):
    """
        Generate test output using the test images
        :param sess: TF session
        :param logits: TF Tensor for the logits
        :param keep_prob: TF Placeholder for the dropout keep probability
        :param image_pl: TF Placeholder for the image placeholder
        :param adi_pl: TF Placeholder for the image placeholder
        :param data_folder: Path to the folder that contains the datasets
        :param image_shape: Tuple - Shape of image
        :return: Output for for each test image
        """
    for i in range(len(glob(os.path.join(data_folder, 'image_2', '*.png')))):
        start = time.time()
        image_file_by_train = glob(os.path.join(data_folder, 'image_2', '*.png'))[i]
        adi_file_by_train = glob(os.path.join(data_folder, 'ADI', '*.png'))[i]
        image_file_for_visualize = glob(os.path.join(data_folder, 'image_2', '*.png'))[i]

        by_train = scipy.misc.imread(image_file_by_train)  # temp.shape = (375, 1242, 4)
        add_adi = scipy.misc.imread(adi_file_by_train)
        for_visualize = scipy.misc.imread(image_file_for_visualize)

        image_by_train = scipy.misc.imresize(by_train, image_shape)  # image_shape = 160, 576
        image_add_adi = scipy.misc.imresize(add_adi, image_shape)
        image_add_adi = image_add_adi.reshape(image_add_adi.shape[0], image_add_adi.shape[1], 1)
        image_for_visualize = scipy.misc.imresize(for_visualize, image_shape)

        street_im = scipy.misc.toimage(image_for_visualize)

        im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_pl: [image_by_train], adi_pl: [image_add_adi]})
        processing_time = time.time() - start
        mask, street_im = paste_mask(street_im, im_softmax, image_shape, 1, [[0, 255, 0, 127]])

        yield os.path.basename(image_file_by_train), np.array(mask), np.array(street_im), processing_time

def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        # lidar_paths = glob(os.path.join(data_folder, 'ADI', '*.png'))

        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)):path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

        lidar_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)):path
            for path in glob(os.path.join(data_folder, 'ADI', '*.png'))}


        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            adis = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:

                gt_image_file = label_paths[os.path.basename(image_file)]
                adi_file = lidar_paths[os.path.basename(image_file)]

                image = scipy.misc.imread(image_file)
                # adi_file = (data_folder + '\\ADI\\') + image_file.split('\\')[-1]
                adi = scipy.misc.imread(adi_file)

                # image = cv.imread(image_file)
                # print("temp.shape", image.shape)
                gt_image = scipy.misc.imread(gt_image_file)
                # gt_image = cv.imread(gt_image_file)

                image2, adi2, gt_image2 = crop_image(image, adi, gt_image)
                image3, adi3, gt_image3 = flip_image(image, adi, gt_image)

                image = scipy.misc.imresize(image, image_shape)
                # image = cv.resize(image, image_shape)
                adi = scipy.misc.imresize(adi, image_shape)
                adi = adi.reshape(adi.shape[0], adi.shape[1], 1)
                gt_image = scipy.misc.imresize(gt_image, image_shape)
                # gt_image = cv.resize(gt_image, image_shape)

                image2 = scipy.misc.imresize(image2, image_shape)
                # image2 = cv.resize(image2, image_shape)
                adi2 = scipy.misc.imresize(adi2, image_shape)
                adi2 = adi2.reshape(adi.shape[0], adi.shape[1], 1)
                gt_image2 = scipy.misc.imresize(gt_image2, image_shape)
                # gt_image2 = cv.resize(gt_image2, image_shape)

                image3 = scipy.misc.imresize(image3, image_shape)
                # image3 = cv.resize(image3, image_shape)
                adi3 = scipy.misc.imresize(adi3, image_shape)
                adi3 = adi3.reshape(adi.shape[0], adi.shape[1], 1)
                gt_image3 = scipy.misc.imresize(gt_image3, image_shape)
                # gt_image3 = cv.resize(gt_image3, image_shape)

                contrast = random.uniform(0.85, 1.15) # Contrast augmentation
                bright = random.randint(-45, 30) # Brightness augmentation
                image = bc_img(image, contrast, bright)

                gt_image = process_gt_image(gt_image)
                gt_image2 = process_gt_image(gt_image2)
                gt_image3 = process_gt_image(gt_image3)

                images.append(image)
                adis.append(adi)
                gt_images.append(gt_image)

                images.append(image2)
                adis.append(adi2)
                gt_images.append(gt_image2)

                images.append(image3)
                adis.append(adi3)
                gt_images.append(gt_image3)
            yield np.array(images), np.array(adis), np.array(gt_images)

    return get_batches_fn

def Conv2D_Layer(x,
                 num_filters,
                 filter_height=3,
                 filter_width=3,
                 stride=1,
                 padding='SAME',
                 dilation=1,
                 name=None):
    """ Create a convolution layer. """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        # Create tf variables for the weights and biases of the conv layer
        W = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters],
                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        # b = tf.get_variable('biases', shape=[num_filters], initializer=tf.constant_initializer(0.0))
        # Perform convolution.
        conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], dilations=dilation, padding=padding)

        return conv

def Conv2D_Block(x,
                 num_filters,
                 filter_height=3,
                 filter_width=3,
                 stride=1,
                 padding='SAME',
                 dilation=1,
                 batch_normalization=False,
                 relu=False,
                 name=None):

    conv = Conv2D_Layer(x, num_filters, filter_height=filter_height, filter_width=filter_width, stride=stride,
                 padding=padding, dilation=dilation, name=name)

    # Apply Batch normalization
    if batch_normalization is True:
        conv = Batch_Normalization(conv)

    # Apply ReLU non linearity
    if relu is True:
        conv = tf.nn.relu(conv)

    return conv

def Atrous_Conv2D_Layer(x,
                        num_filters,
                        filter_height=3,
                        filter_width=3,
                        dilation=1,
                        padding='SAME',
                        name=None):
    """ Create a convolution layer. """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        # Create tf variables for the weights and biases of the conv layer
        W = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters],
                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        # b = tf.get_variable('biases', shape=[num_filters], initializer=tf.constant_initializer(0.0))
        # Perform convolution.
        conv = tf.nn.atrous_conv2d(x, W, dilation, padding=padding)

        return conv

def Atrous_Conv2D_Block(x,
                 num_filters,
                 filter_height=3,
                 filter_width=3,
                 dilation=1,
                 padding='SAME',
                 batch_normalization=False,
                 relu=False,
                 name=None):

    conv = Atrous_Conv2D_Layer(x, num_filters, filter_height=filter_height, filter_width=filter_width, dilation=dilation,
                 padding=padding, name=name)

    # Apply Batch normalization
    if batch_normalization is True:
        conv = Batch_Normalization(conv)

    # Apply ReLU non linearity
    if relu is True:
        conv = tf.nn.relu(conv)

    return conv


def Deconv2D_Layer(x, shape, num_filters, output_shape,
                   filter_height = 4, filter_width = 4,
                   stride=2, padding='SAME', name=None):
    """ Create a deconvolution layer. """

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        # Create tf variables for the weights and biases of the deconv layer
        W = tf.get_variable('weights', shape=[filter_height, filter_width, shape[3].value, num_filters],
                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

        # Perform deconvolution.
        if output_shape is None:
            output_shape = x.get_shape().as_list()
            output_shape[1] *= 2
            output_shape[2] *= 2
            output_shape[3] = W.get_shape().as_list()[2]

        deconv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding=padding, name=name)

        return deconv

def Deconv2D_Block(x,
                   shape,
                   num_filters,
                   output_shape,
                   filter_height=4,
                   filter_width=4,
                   stride=2,
                   padding='SAME',
                   batch_normalization=False,
                   relu=False,
                   name=None):
    deconv = Deconv2D_Layer(x, shape, num_filters, output_shape, filter_height=filter_height, filter_width=filter_width,
                            stride=stride, padding=padding, name=name)

    # Apply Batch normalization
    if batch_normalization is True:
        deconv = Batch_Normalization(deconv)

    # Apply ReLU non linearity
    if relu is True:
        deconv = tf.nn.relu(deconv)

    return deconv

def Batch_Normalization(x):
    return tf.layers.batch_normalization(x)

def ReLU(x):
    return tf.nn.relu(x)

def Max_Pooling(x, name, filter_height=2, filter_width=2, stride=2, padding="VALID"):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def Avg_Pooling(x, name, filter_height=2, filter_width=2, stride_height=2, stride_width=2, padding="VALID"):
    return tf.nn.avg_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_height, stride_width, 1], padding=padding, name=name)

def Global_Avg_Pool(x, stride=1):
    return global_avg_pool(x, name="global_avg_pooling")

def Global_Max_Pool(x, stride=1):
    return global_max_pool(x, name="global_max_pooling")

def Dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob=keep_prob)

def Softmax(x):
    return tf.nn.softmax(x)

# PSPNet
def Zero_Padding(x, paddings, name):
    pad_mat = np.array([[0,0], [paddings, paddings], [paddings, paddings], [0, 0]])
    return tf.pad(x, paddings=pad_mat, name=name)

def Resize_Bilinear(x, size, name):
    return tf.image.resize_bilinear(x, size=size, align_corners=True, name=name)

def Concat(x, axis, name):
    return tf.concat(x, axis=axis, name=name)
