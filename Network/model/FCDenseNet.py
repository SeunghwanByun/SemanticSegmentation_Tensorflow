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
KEEP_PROB = 0.2
MAX_ITERATION = 1e-2
NUM_OF_CLASSESS = 2
IMAGE_SHAPE_KITTI = (160, 576)
BATCH_SIZE = 1
EPOCHS = 20
LEARNING_RATE = 1e-4

DATA_DIR = "data_road"

def bottleneck_layer(x, growth_rate, keep_prob, name):
    with tf.name_scope(name):
        x = Batch_Normalization(x)
        x = ReLU(x)
        x = Conv2D_Block(x, 4 * growth_rate, filter_height=1, filter_width=1, stride=1, name=name+"_conv1")
        x = Dropout(x, keep_prob=keep_prob)

        x = Batch_Normalization(x)
        x = ReLU(x)
        x = Conv2D_Block(x, growth_rate, stride=1, name=name+"_conv2")
        x = Dropout(x, keep_prob=keep_prob)

        return x

def Transition_Layer(x, theta, name):
    with tf.name_scope(name):
        x = Batch_Normalization(x)
        x = ReLU(x)

        in_channel = int(x.shape[-1])
        x = Conv2D_Block(x, int(in_channel * theta), filter_height=1, filter_width=1, stride=1, name=name+"_conv")
        x = Avg_Pooling(x, name=name+"avg_pool")

        return x

def DenseBlock(x, num_bottleneck_layers, growth_rate, keep_prob, name):
    layers_concat = list()
    layers_concat.append(x)

    x = bottleneck_layer(x, growth_rate=growth_rate, keep_prob=keep_prob, name=name+"bottleneck_layer_0")
    layers_concat.append(x)
    for i in range(num_bottleneck_layers):
        x = Concat(layers_concat, axis=-1, name=name+"bottleneck_layer_concatenate_"+str(i+1))
        x = bottleneck_layer(x, growth_rate=growth_rate, keep_prob=keep_prob, name=name+"bottleneck_layer_"+ str(i+1))
        layers_concat.append(x)

    x = Concat(layers_concat, axis=-1, name=name+"bottleneck_layer_concatenate_final")

    return x


# Architecture of FC-DenseNet
# => Layer : BN + ReLU + 3x3 conv + droput(0.2)
# => Transition Down(TD) : BN + ReLU + 1x1 conv + dropout(0.2) + 2x2 max_pool
# => Transition Up(TU) : 3x3 transposed conv(stride=2)
# input m = 3
# 3x3 conv m = 48
# DB (4 layers) + TD, m = 112
# DB (5 layers) + TD, m = 192
# DB (7 layers) + TD, m = 304
# DB (10 layers) + TD, m = 464
# DB (12 layers) + TD, m = 656
# DB (15 layers) + TD, m = 896
# TU + DB(12 layers), m = 1088
# TU + DB(10 layers), m = 816
# TU + DB(7 layers), m = 578
# TU + DB(5 layers), m = 384
# TU + DB(4 layers), m = 256
# 1x1 conv, m = c
# softmax
def FCDenseNet(x, keep_prob, num_classes):
    n_filters_first_conv = 48
    GROWTH_RATE = 16
    THETA = 0.5
    n_layers_per_blocks = [4, 5, 7, 10, 12, 15]

    print("Build Network Start...")
    print("Build encoder start...")
    # input convolution
    dense_init = Conv2D_Block(x, n_filters_first_conv, name="dense_init")
    print("dense init", dense_init.get_shape())

    # Encoder
    dense_block1 = DenseBlock(dense_init, n_layers_per_blocks[0], growth_rate=GROWTH_RATE, keep_prob=keep_prob, name="denseblock1")
    transition_down1 = Transition_Layer(dense_block1, theta=THETA, name="transition_layer1")
    print("db1", dense_block1.get_shape())
    print("td1", transition_down1.get_shape())


    dense_block2 = DenseBlock(transition_down1, n_layers_per_blocks[1], growth_rate=GROWTH_RATE, keep_prob=keep_prob, name="denseblock2")
    transition_down2 = Transition_Layer(dense_block2, theta=THETA, name="transition_layer2")
    print("db2", dense_block2.get_shape())
    print("td2", transition_down2.get_shape())


    dense_block3 = DenseBlock(transition_down2, n_layers_per_blocks[2], growth_rate=GROWTH_RATE, keep_prob=keep_prob, name="denseblock3")
    transition_down3 = Transition_Layer(dense_block3, theta=THETA, name="transition_layer3")
    print("db3", dense_block3.get_shape())
    print("td3", transition_down3.get_shape())



    dense_block4 = DenseBlock(transition_down3, n_layers_per_blocks[3], growth_rate=GROWTH_RATE, keep_prob=keep_prob, name="denseblock4")
    transition_down4 = Transition_Layer(dense_block4, theta=THETA, name="transition_layer4")
    print("db4", dense_block4.get_shape())
    print("td4", transition_down4.get_shape())



    dense_block5 = DenseBlock(transition_down4, n_layers_per_blocks[4], growth_rate=GROWTH_RATE, keep_prob=keep_prob, name="denseblock5")
    transition_down5 = Transition_Layer(dense_block5, theta=THETA, name="transition_layer5")
    print("db5", dense_block5.get_shape())
    print("td5", transition_down5.get_shape())



    dense_block6 = DenseBlock(transition_down5, n_layers_per_blocks[5], growth_rate=GROWTH_RATE, keep_prob=keep_prob, name="denseblock6")
    # transition_down6 = Transition_Layer(dense_block6, theta=THETA, name="transition_layer6")


    print("db6", dense_block6.get_shape())
    # print("td6", transition_down6.get_shape())
    print("Encoder build done...")
    print("Build decoder start...")

    # Decoder\
    # print("transition down 6", transition_down6.get_shape())
    print("dense block 5", dense_block5.get_shape())
    transition_up1 = Deconv2D_Block(dense_block6, dense_block5.get_shape(), 430, tf.shape(dense_block5), name="transition_up1")
    tu_db_concat1 = Concat([transition_up1, dense_block5], axis=-1, name="tu_db_concat1")

    transition_up2 = Deconv2D_Block(tu_db_concat1, dense_block4.get_shape(), 696, tf.shape(dense_block4), name="transition_up2")
    tu_db_concat2 = Concat([transition_up2, dense_block4], axis=-1, name="tu_db_concat2")

    transition_up3 = Deconv2D_Block(tu_db_concat2, dense_block3.get_shape(), 560, tf.shape(dense_block3), name="transition_up3")
    tu_db_concat3 = Concat([transition_up3, dense_block3], axis=-1, name="tu_db_concat3")

    transition_up4 = Deconv2D_Block(tu_db_concat3, dense_block2.get_shape(), 416, tf.shape(dense_block2), name="transition_up4")
    tu_db_concat4 = Concat([transition_up4, dense_block2], axis=-1, name="tu_db_concat4")

    transition_up5 = Deconv2D_Block(tu_db_concat4, dense_block1.get_shape(), 320, tf.shape(dense_block1), name="transition_up5")
    tu_db_concat5 = Concat([transition_up5, dense_block1], axis=-1, name="tu_db_concat5")

    print("Decoder build done...")

    final_conv = Conv2D_Block(tu_db_concat5, num_classes, filter_height=1, filter_width=1, name="final_conv")

    prediction = tf.argmax(final_conv, dimension=3, name="prediction")
    print("Build Network done...")

    return tf.expand_dims(prediction, dim=3), final_conv


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
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 4], name="input_image")
    prediction = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 2], name="prediction")

    # Network 선언
    pred_annotation, logits = FCDenseNet(image, keep_probability, NUM_OF_CLASSESS)

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
    saver.save(sess, './model/FCHarDNet.ckpt', global_step=global_step)

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
