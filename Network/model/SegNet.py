import cv2 as cv
import tensorflow as tf
import random
import re
from glob import glob
import numpy as np
import os
import time
import scipy.misc
from datetime import timedelta

from Utils import *

# 학습에 필요한 설정값들을 지정
L2_REG = 1e-5
STDDEV = 1e-2
KEEP_PROB = 0.8
MAX_ITERATION = int(100000 + 1)
NUM_OF_CLASSESS = 2 # 레이블 개수
IMAGE_SHAPE_KITTI = (160, 576)
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-4

DATA_DIR = "data_road"


def SegNet(x, num_classes):
    # encoder
    conv1 = Conv2D_Block(x, 64, batch_normalization=True, name="conv1")
    conv2 = Conv2D_Block(conv1, 64, batch_normalization=True, name="conv2")
    pool1 = Max_Pooling(conv2, "pool1")

    conv3 = Conv2D_Block(pool1, 128, batch_normalization=True, name="conv3")
    conv4 = Conv2D_Block(conv3, 128, batch_normalization=True, name="conv4")
    pool2 = Max_Pooling(conv4, "pool2")

    conv5 = Conv2D_Block(pool2, 256, batch_normalization=True, name="conv5")
    conv6 = Conv2D_Block(conv5, 256, batch_normalization=True, name="conv6")
    conv7 = Conv2D_Block(conv6, 256, batch_normalization=True, name="conv7")
    pool3 = Max_Pooling(conv7, "pool3")

    conv8 = Conv2D_Block(pool3, 512, batch_normalization=True, name="conv8")
    conv9 = Conv2D_Block(conv8, 512, batch_normalization=True, name="conv9")
    conv10 = Conv2D_Block(conv9, 512, batch_normalization=True, name="conv10")
    pool4 = Max_Pooling(conv10, "pool4")

    conv11 = Conv2D_Block(pool4, 512, batch_normalization=True, name="conv11")
    conv12 = Conv2D_Block(conv11, 512, batch_normalization=True, name="conv12")
    conv13 = Conv2D_Block(conv12, 512, batch_normalization=True, name="conv13")
    pool5 = Max_Pooling(conv13, "pool5")
    print("Build encoder done...")

    # decoder
    unpool1 = Deconv2D_Block(pool5, conv13.get_shape(), 512, tf.shape(conv13), name="unpool1")

    conv14 = Conv2D_Block(unpool1, 512, batch_normalization=True, name="conv14")
    conv15 = Conv2D_Block(conv14, 512, batch_normalization=True, name="conv15")
    conv16 = Conv2D_Block(conv15, 512, batch_normalization=True, name="conv16")

    unpool2 = Deconv2D_Block(conv16, conv10.get_shape(), 512, tf.shape(conv10), name="unpool2")

    conv17 = Conv2D_Block(unpool2, 512, batch_normalization=True, name="conv17")
    conv18 = Conv2D_Block(conv17, 512, batch_normalization=True, name="conv18")
    conv19 = Conv2D_Block(conv18, 256, batch_normalization=True, name="conv19")

    unpool3 = Deconv2D_Block(conv19, conv7.get_shape(), 256, tf.shape(conv7), name="unpool3")

    conv20 = Conv2D_Block(unpool3, 256, batch_normalization=True, name="conv20")
    conv21 = Conv2D_Block(conv20, 256, batch_normalization=True, name="conv21")
    conv22 = Conv2D_Block(conv21, 128, batch_normalization=True, name="conv22")

    unpool4 = Deconv2D_Block(conv22, conv4.get_shape(), 128, tf.shape(conv4), name="unpool4")

    conv23 = Conv2D_Block(unpool4, 128, batch_normalization=True, name="conv23")
    conv24 = Conv2D_Block(conv23, 64, batch_normalization=True, name="conv24")

    unpool5 = Deconv2D_Block(conv24, conv2.get_shape(), 64, tf.shape(conv2), name="unpool5")

    conv25 = Conv2D_Block(unpool5, 64, batch_normalization=True, name="conv25")
    conv26 = Conv2D_Layer(conv25, num_classes, name="conv26")
    conv26 = Batch_Normalization(conv26)

    outputs = tf.argmax(conv26, dimension=3, name="prediction")
    print("Build decoder done...")

    return tf.expand_dims(outputs, dim=3), conv26

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

    # SegNet 선언
    # pred_annotation, logits = PSPNet101(image, NUM_OF_CLASSESS)
    pred_annotation, logits = SegNet(image, NUM_OF_CLASSESS)

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
    saver.save(sess, './model/SegNet.ckpt', global_step=global_step)

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
