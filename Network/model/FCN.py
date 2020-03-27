import os
from glob import glob
import re
import scipy.misc
import matplotlib.pyplot as plt
from scipy import ndimage
import random
import numpy as np
import tensorflow as tf
import time
from datetime import timedelta
import cv2 as cv

# 사용 안 함.
VGG_MEAN = [103.939, 116.779, 123.68]

# 학습에 필요한 설정값들을 지정
L2_REG = 1e-5
STDDEV = 1e-2
KEEP_PROB = 0.8
MAX_ITERATION = int(100000 + 1)
NUM_OF_CLASSESS = 2 # 레이블 개수
IMAGE_SIZE = 224 # (160, 576)
IMAGE_SHAPE_KITTI = (160, 576)
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4

DATA_DIR = "data_road"

class FCN(object):
    """ Implementation of VGG16 network """

    def __init__(self, x, keep_prob, num_classess):
        """ Create the graph of the Alex model.
        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classess: Number of classess in the dataset.
        """
        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSESS = num_classess
        self.KEEP_PROB = keep_prob

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """ Create the network graph. """

        conv1_1 = conv_layer(self.X, 64, 'conv1_1')
        conv1_2 = conv_layer(conv1_1, 64, 'conv1_2')
        pool1 = max_pool(conv1_2, 'pool1')

        conv2_1 = conv_layer(pool1, 128, 'conv2_1')
        conv2_2 = conv_layer(conv2_1, 128, 'conv2_2')
        pool2 = max_pool(conv2_2, 'pool2')

        conv3_1 = conv_layer(pool2, 256, 'conv3_1')
        conv3_2 = conv_layer(conv3_1, 256, 'conv3_2')
        conv3_3 = conv_layer(conv3_2, 256, 'conv3_3')
        pool3 = max_pool(conv3_3, 'pool3')

        conv4_1 = conv_layer(pool3, 512, 'conv4_1')
        conv4_2 = conv_layer(conv4_1, 512, 'conv4_2')
        conv4_3 = conv_layer(conv4_2, 512, 'conv4_3')
        conv4_4 = conv_layer(conv4_3, 512, 'conv4_4')
        pool4 = max_pool(conv4_4, 'pool4')

        conv5_1 = conv_layer(pool4, 512, 'conv5_1')
        conv5_2 = conv_layer(conv5_1, 512, 'conv5_2')
        conv5_3 = conv_layer(conv5_2, 512, 'conv5_3')
        # conv5_4 = conv_layer(conv5_3, 512, 'conv5_4')
        pool5 = max_pool(conv5_3, 'pool5')

        # conv6 정의
        conv6 = conv_layer(pool5, 4096, 'conv6', filter_height=7, filter_width=7)
        dropout6 = dropout(conv6, self.KEEP_PROB)

        # conv7 정의
        conv7 = conv_layer(dropout6, 4096, 'conv7', filter_height=1, filter_width=1)
        dropout7 = dropout(conv7, self.KEEP_PROB)

        # conv8 정의
        conv8 = conv_layer(dropout7, self.NUM_CLASSESS, 'conv8', filter_height=1, filter_width=1)

        # FCN-8s를 위한 Skip Layers Fusion을 설정
        # 이제 원본 이미지 크기로 Upsampling을 하기 위한 deconv 레이어를 정의
        deconv_shape1 = pool4.get_shape()
        conv_t1 = deconv_layer(conv8, deconv_shape1, self.NUM_CLASSESS, 'conv_t1', tf.shape(pool4))
        fuse_1 = fuse(conv_t1, pool4, 'fuse_1')

        deconv_shape2 = pool3.get_shape()
        conv_t2 = deconv_layer(fuse_1, deconv_shape2, deconv_shape1[3].value, 'conv_t2', tf.shape(pool3))
        fuse_2 = fuse(conv_t2, pool3, 'fuse_2')

        shape = tf.shape(self.X)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], self.NUM_CLASSESS])

        with tf.variable_scope('conv_t3', reuse=tf.AUTO_REUSE):
            W_t3 = tf.get_variable('weights', shape=[16, 16, self.NUM_CLASSESS, deconv_shape2[3].value], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            b_t3 = tf.get_variable('bias', shape=[self.NUM_CLASSESS], initializer=tf.constant_initializer(0.0))

        # Perform convolution.
        conv_t3 = tf.nn.conv2d_transpose(fuse_2, W_t3, deconv_shape3, strides=[1, 8, 8, 1], padding='SAME')
        conv_t3 = tf.nn.bias_add(conv_t3, b_t3)

        print("conv_t3:", conv_t3.shape)
        # 최종 prediction 결과를 결정하기 위해 마지막 activation들 중에서 argmax로 최대값을 가진 activation을 추출.
        annotation_pred = tf.argmax(conv_t3, dimension=3, name='prediction')

        print("annotation_pred:", annotation_pred.shape)
        return tf.expand_dims(annotation_pred, dim=3), conv_t3 # tf.expand_dims(~~) => (?,?,?), conv_t3 => (?,?,?,2)


def conv_layer(x, num_filters, name, filter_height = 3, filter_width = 3, stride = 1, padding = 'SAME'):
    """ Create a convolution layer. """

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        # Create tf variables for the weights and biases of the conv layer
        W = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

        b = tf.get_variable('biases', shape=[num_filters], initializer=tf.constant_initializer(0.0))

        # Perform convolution.
        conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        # Add the biases.
        z = tf.nn.bias_add(conv, b)
        # Apply ReLU non linearity.
        a = tf.nn.relu(z)

        return a

def deconv_layer(x, shape, num_filters, name, output_shape, filter_height = 4, filter_width = 4, stride = 2, padding = 'SAME'):
    """ Create a deconvolution layer. """

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        # Create tf variables for the weights and biases of the deconv layer
        W = tf.get_variable('weights', shape=[filter_height, filter_width, shape[3].value, num_filters], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

        b = tf.get_variable('biases', shape=[shape[3].value], initializer=tf.constant_initializer(0.0))

        # Perform convolution.
        if output_shape is None:
            output_shape = x.get_shape().as_list()
            output_shape[1] *= 2
            output_shape[2] *= 2
            output_shape[3] = W.get_shape().as_list()[2]

        # Perform deconvolution
        deconv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding=padding)
        # Add the biases
        z = tf.nn.bias_add(deconv, b)

        return z

def max_pool(x, name, filter_height = 2, filter_width = 2, stride = 2, padding = 'VALID'):
    """ Create a max pooling layer. """
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def dropout(x, keep_prob):
    """ Create a dropout layer. """
    return tf.nn.dropout(x, keep_prob=keep_prob)

def fuse(x1, x2, name):
    """ Create a fuse layer. """
    return tf.add(x1, x2, name=name)

def img_size(image):
    return image.shape[0], image.shape[1]

def crop_image(image, gt_image):
    h, w = img_size(image)
    nw = random.randint(1150, w-5) # Random crop size
    nh = int(nw / 3.3) # Keep original aspect ration
    x1 = random.randint(0, w - nw) # Random position of crop
    y1 = random.randint(0, h - nh)
    return image[y1:(y1+nh), x1:(x1+nw), :], gt_image[y1:(y1+nh), x1:(x1+nw), :]

def flip_image(image, gt_image):
    return np.flip(image, axis=1), np.flip(gt_image, axis=1)

def bc_img(img, s=1.0, m=0.0):
    img = img.astype(np.int)
    img = img * s + m
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img

def process_gt_image(gt_image):
    background_color = np.array([255, 0, 0])
    gt_bg = np.all(gt_image == background_color, axis=2)
    gt_bg = gt_bg.reshape(gt_bg.shape[0], gt_bg.shape[1], 1)

    gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
    return gt_image

def paste_mask(street_im, im_soft_max, image_shape, color, obj_color_schema):
    im_soft_max_r = np.squeeze(im_soft_max[0])[:, :, color].reshape(image_shape[0], image_shape[1])
    # im_soft_max_r = im_soft_max[0][:, color]
    segmentation_r = (im_soft_max_r > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation_r, np.array(obj_color_schema))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im.paste(mask, box=None, mask=mask)

    return street_im

def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep probability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'merge', '*.png')):
        temp = scipy.misc.imread(image_file) # temp.shape = (375, 1242, 4)
        image = scipy.misc.imresize(temp, image_shape) # image_shape = 160, 576
        street_im = scipy.misc.toimage(image)

        im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_pl: [image]})

        street_im = paste_mask(street_im, im_softmax, image_shape, 1, [[0, 255, 0, 127]])

        yield os.path.basename(image_file), np.array(street_im)

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
        image_paths = glob(os.path.join(data_folder, 'merge', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)):path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:

                gt_image_file = label_paths[os.path.basename(image_file)]

                # i = cv.imread(image_file[0])
                # g = cv.imread(gt_image_file[0])
                # cv.imshow("i", i)
                # cv.imshow("g", g)
                # cv.waitKey(0)

                image = scipy.misc.imread(image_file)
                gt_image = scipy.misc.imread(gt_image_file)

                image2, gt_image2 = crop_image(image, gt_image)
                image3, gt_image3 = flip_image(image, gt_image)

                image = scipy.misc.imresize(image, image_shape)
                gt_image = scipy.misc.imresize(gt_image, image_shape)

                image2 = scipy.misc.imresize(image2, image_shape)
                gt_image2 = scipy.misc.imresize(gt_image2, image_shape)

                image3 = scipy.misc.imresize(image3, image_shape)
                gt_image3 = scipy.misc.imresize(gt_image3, image_shape)

                contrast = random.uniform(0.85, 1.15) # Contrast augmentation
                bright = random.randint(-45, 30) # Brightness augmentation
                image = bc_img(image, contrast, bright)

                gt_image = process_gt_image(gt_image)
                # print("gt_image.shape #3")
                # print(gt_image.shape)
                gt_image2 = process_gt_image(gt_image2)
                gt_image3 = process_gt_image(gt_image3)

                # plt.imshow(image)
                # plt.imshow(image2)
                # plt.imshow(image3)
                # plt.show()

                images.append(image)
                gt_images.append(gt_image)

                images.append(image2)
                gt_images.append(gt_image2)

                images.append(image3)
                gt_images.append(gt_image3)
            yield np.array(images), np.array(gt_images)

    return get_batches_fn

def main():
    # 인풋 이미지와 타겟 이미지, 드롭아웃 확률을 받을 플레이스홀더를 정의
    keep_probability = tf.placeholder(tf.float32, name='keep_probability')
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 4], name='input_image')
    annotation = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 2], name='annotation')

    # FCN class 선언
    fcn = FCN(image, keep_probability, NUM_OF_CLASSESS)

    # FCN 그래프를 선언하고 Tensorboard를 위한 summary들을 지정
    pred_annotation, logits = fcn.create()

    print("pred_annotation:", pred_annotation.shape)
    print("logits :", logits.shape)

    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)

    # 손실 함수를 선언하고 손실 함수에 대한 summary를 지정
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(annotation, squeeze_dims=[3], name='entropy')))

    print("logits 2:", logits.shape)
    print("annotation: ", annotation.shape)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=annotation))
    tf.summary.scalar('entropy', loss)

    # 옵티마이저를 선언하고 파라미터를 한스텝 업데이트하는 train_step 연산을 정의
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    # optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE) # 후에 옵티마이저를 RMSProp 으로도 설정해보자.
    train_step = optimizer.minimize(loss)

    # Tensorboard를 위한 summary들을 하나로 merge
    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    # training 데이터와 validation 데이터의 개수를 불러옴
    print("Setting up image reader...")

    training_labels_count = len(glob(os.path.join(DATA_DIR, 'training/gt_image_2/*_road_*.png')))
    training_images_count = len(glob(os.path.join(DATA_DIR, 'training/merge/*.png')))
    testing_images_count = len(glob(os.path.join(DATA_DIR, 'testing/merge/*.png')))

    assert not (training_images_count == training_labels_count == testing_images_count == 0), \
        'Kitti dataset not found. Extract Kitti dataset in {}'.format(DATA_DIR)
    assert training_images_count == 289, 'Expected 289 training images, found {} images.'.format(
        training_images_count)  # 289
    assert training_labels_count == 289, 'Expected 289 training labels, found {} labels.'.format(
        training_labels_count)  # 289
    assert testing_images_count == 290, 'Expected 290 testing images, found {} images.'.format(testing_images_count)

    # training 데이터를 불러옴
    get_batches_fn = gen_batch_function(os.path.join(DATA_DIR, 'training'), IMAGE_SHAPE_KITTI)

    # 세션을 염
    sess = tf.Session()

    # 학습된 파라미터를 저장하기 위한 tf.train.Saver()
    # tensorboard summary들을 저장하기 위한 tf.summary.FileWriter를 선언
    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter("logs/", sess.graph)

    # 변수들을 초기화하고 저장된 ckpt 파일이 있으면 저장된 파라미터를 불러옴
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("logs/")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    start = time.time() # 시작 시간 저장
    for epoch in range(EPOCHS):
        loss = None
        s_time = time.time()
        # 학습 데이터를 불러오고 feed_dict에 데이터를 지정
        for images, labels in get_batches_fn(batch_size=BATCH_SIZE):
            # print(images[0])
            # print(images[0].shape)
            # print(labels[0].shape)
            # cv.imshow("images", images[0])
            # cv.waitKey(0)
            # cv.imshow("labels", labels[0])
            # cv.waitKey(0)
            # print(images[0].shape)
            # print(labels[0].shape)
            feed_dict = {image: images, annotation: labels, keep_probability: 0.8}

            # train_step을 실행해서 파라미터를 한 스텝 업데이트 함
            sess.run(train_step, feed_dict=feed_dict)

        print("[Epoch: {0}/{1} Time: {2}]" .format(epoch + 1, EPOCHS, str(timedelta(seconds=(time.time() - s_time)))))

    print("Time: ", time.time() - start) # 현재 시각 - 시작 시간 = 실행 시간
    print("Training Successfully")

    # 훈련이 끝나고 테스트 데이터 셋으로 테스트
    output_dir = os.path.join(DATA_DIR, 'output')
    print("Training Finished. Saving test images to: {}".format(output_dir))
    image_output = gen_test_output(sess, logits, keep_probability, image, os.path.join(DATA_DIR, 'testing'), IMAGE_SHAPE_KITTI)

    for name, image in image_output:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

    # for image_file in glob(os.path.join(DATA_DIR, 'testing', 'merge', '*.png')):
    #     test_image = scipy.misc.imresize(scipy.misc.imread(image_file), IMAGE_SHAPE_KITTI)
    #
    #     street_im = scipy.misc.toimage(test_image)
    #     im_softmax = sess.run([tf.nn.softmax(logits=logits)], {keep_probability: 1.0, image: test_image})
    #
    #     street_im = paste_mask(street_im, im_softmax, IMAGE_SHAPE_KITTI, 1, [[0, 255, 0, 127]])
    #
    #     for name, image in os.path.basename(image_file), np.array(street_im):
    #         scipy.misc.imsave(os.path.join(DATA_DIR, 'output', name), image)

main()
