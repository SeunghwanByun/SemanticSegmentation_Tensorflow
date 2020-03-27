import cv2 as cv
import tensorflow as tf
import numpy as np
import scipy.misc
from datetime import timedelta
import os
import csv
import time
from glob import glob


# 학습에 필요한 설정값들을 지정
KEEP_PROB = 0.1
MAX_ITERATION = 1e-2
NUM_OF_CLASSESS_IMG = 2
NUM_OF_CLASSESS_LID = 1
IMAGE_SHAPE_KITTI = (128, 480)
# IMAGE_SHAPE_KITTI = (160, 576)
# IMAGE_SHAPE_KITTI = (192, 704)
# IMAGE_SHAPE_KITTI = (384, 1280)
# IMAGE_SHAPE_KITTI = (713, 1280)
BATCH_SIZE = 1
EPOCHS = 30
LEARNING_RATE = 1e-4

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
                               name="input_rgb")
    LiDAR_IMAGE = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 3],
                                 name="input_lidar")
    RGB_LABEL = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 2],
                                name="prediction_img")
    LiDAR_LABEL = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 1],
                                    name="prediction_lid")

    # Network 선언
    logits_img, logits_lid = GSNet(RGB_IMAGE, LiDAR_IMAGE, keep_probability, NUM_OF_CLASSESS_IMG, NUM_OF_CLASSESS_LID)

    # Tensorboard를 위한 summary들을 지정
    tf.summary.image('input_image', RGB_IMAGE, max_outputs=2)
    tf.summary.image('input_lidar', LiDAR_IMAGE, max_outputs=1)
    # 손실 함수를 선언하고 손실 함수에 대한 summary들을 지정
    training_loss_lid = tf.losses.mean_squared_error(labels=LiDAR_LABEL, predictions=logits_lid)
    training_loss_img = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_img,
                                                                           labels=RGB_LABEL) + 0.5 * training_loss_lid)
    inference_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_img, labels=RGB_LABEL))

    tf.summary.scalar('training_loss_lid', training_loss_lid)
    tf.summary.scalar('training_loss_img', training_loss_img)
    tf.summary.scalar('inference_loss', inference_loss)

    # 옵티마이저를 선언하고 파라미터를 한 스텝 업데이트하는 train_step 연산을 정의
    optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE)
    # train_step = optimizer.minimize(loss)

    # Constant to scale sum of gradient
    const = tf.compat.v1.constant(1 / BATCH_SIZE * 3)

    # Get all trainable variables
    t_vars_lid = tf.compat.v1.trainable_variables()
    t_vars_img = tf.compat.v1.trainable_variables()

    # Create a copy of all trainable variables with '0' as initial values
    accum_tvars_img = [tf.compat.v1.Variable(tf.compat.v1.zeros_like(t_var.initialized_value()), trainable=False) for t_var
                   in t_vars_lid]
    accum_tvars_lid = [tf.compat.v1.Variable(tf.compat.v1.zeros_like(t_var.initialized_value()), trainable=False) for t_var
                   in t_vars_img]

    # Create a op to initialize all accums vars
    zero_ops_lid = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvars_lid]
    zero_ops_img = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvars_img]

    # Compute gradients for a batch
    batch_grads_vars_lid = optimizer.compute_gradients(training_loss_lid, t_vars_lid)
    batch_grads_vars_img = optimizer.compute_gradients(training_loss_lid, t_vars_img)

    # Collect the (scaled by const) batch gradient into accumulated vars
    accum_ops_ild = [accum_tvars_lid[i].assign_add(tf.scalar_mul(const, batch_grad_var[0])) for i, batch_grad_var in
                 enumerate(batch_grads_vars_lid)]
    accum_ops_img = [accum_tvars_img[i].assign_add(tf.scalar_mul(const, batch_grad_var[0])) for i, batch_grad_var in
                 enumerate(batch_grads_vars_img)]

    # Apply accums gradients
    train_step_lid = optimizer.apply_gradients(
        [(accum_tvars_lid[i], batch_grad_var[1]) for i, batch_grad_var in enumerate(batch_grads_vars_lid)])
    train_step_img = optimizer.apply_gradients(
        [(accum_tvars_img[i], batch_grad_var[1]) for i, batch_grad_var in enumerate(batch_grads_vars_img)])

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
        # 학습 데이터를 불러오고 feed_dict에 데이터를 지정
        for images, img_labels, lidars, lid_labels in get_batches_fn(batch_size=BATCH_SIZE):
            feed_dict = {RGB_IMAGE: images, LiDAR_IMAGE: lidars, prediction_img: img_labels, prediction_lid: lid_labels,
                         keep_probability: KEEP_PROB}

            # Initialize the accumulated grads
            sess.run(zero_ops_lid)
            sess.run(zero_ops_img)
            for i in range(len(images)):
                sess.run(accum_ops_ild, feed_dict=feed_dict)
                sess.run(accum_ops_img, feed_dict=feed_dict)


            # train_step을 실행해서 파라미터를 한 스텝 업데이트 함
                _, cost = sess.run([train_step_lid, training_loss_lid, train_step_img, training_loss_img],
                                   feed_dict=feed_dict)

            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(cost)

            # Tensorboard 를 위한 sess.run()
            summary = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step=sess.run(global_step))

        print("[Epoch: {0}/{1} Time: {2}]".format(epoch + 1, EPOCHS, str(timedelta(seconds=(time.time() - s_time)))))

    print("Time: ", time.time() - start)  # 현재 시각 - 시작 시간 = 실행 시간
    print("Training Successfully")

    # 훈련이 끝나고 학습된 파라미터 저장
    saver.save(sess, './model/GSNet.ckpt', global_step=global_step)

    # 훈련이 끝나고 테스트 데이터 셋으로 테스트
    output_dir = os.path.join(DATA_DIR, 'output')
    mask_dir = os.path.join(DATA_DIR, 'mask')
    print("Training Finished. Saving test images to: {}".format(output_dir))
    image_output = gen_test_output(sess, logits_img, keep_probability, RGB_IMAGE, LiDAR_IMAGE,
                                   os.path.join(DATA_DIR, 'validating'),
                                   IMAGE_SHAPE_KITTI)

    total_processing_time = 0
    for name, mask, image, processing_time in image_output:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        scipy.misc.imsave(os.path.join(mask_dir, name), mask)
        total_processing_time += processing_time

    print("Average processing time is : ", total_processing_time / 30)


if __name__ == '__main__':
    run()
