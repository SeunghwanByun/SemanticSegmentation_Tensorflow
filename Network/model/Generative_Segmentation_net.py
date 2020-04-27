import cv2 as cv
import tensorflow as tf
import numpy as np
import scipy.misc
from datetime import timedelta
import os
import csv
import time
from glob import glob

from utils_for_LiDAR2RGBImage import *

# 학습에 필요한 설정값들을 지정
KEEP_PROB = 0.1
MAX_ITERATION = 1e-2
NUM_OF_CLASSESS_IMG = 2
NUM_OF_CLASSESS_LID = 10)
IMAGE_SHAPE_KITTI = (160, 576)
BATCH_SIZE = 1
EPOCHS = 30
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
    outputs = x
    skip = x
    for i in range(3):
        residual = SepConv_BN(residual, depth_list[i], prefix+'_separable_conv{}'.format(i+1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = Conv2D_Block(x, depth_list[-1], filter_height=1, filter_width=1, stride=stride, padding='SAME',
                                batch_normalization=True, name=prefix+"_shortcut")
        outputs = tf.add(shortcut, residual, name=prefix+'_add_conv')
    elif skip_connection_type == 'sum':
        outputs = tf.add(residual, x, name=prefix+'_add_sum')
    elif skip_connection_type == 'none':
        outputs = residual

    if return_skip:
        return outputs, skip
    else:
        return outputs


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it.
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed.
        tf.import_graph_def(graph_def, name="")

    return graph


def GSNet(x_img, keep_prob, num_classes, after_entry_conv1_lid,
    after_entry_conv1_2_lid, after_xblock_entry_1_lid,
    after_xblock_entry_2_lid, after_skip1_lid, after_xblock_entry_3_lid,
    after_xblock_middle, after_xblock_exit_1_lid, after_extracted_feature_lid):
    """
        Instantiates the DeepLabV3+ architecture.
     :param x_img: Tensor for RGB Image
    :param keep_prob: Tensor for keep probability
    :param num_classes: int, number of classes
    :param after_entry_conv1_lid: Tensor from pretrained model
    :param after_entry_conv1_2_lid: Tensor from pretrained model
    :param after_xblock_entry_1_lid: Tensor from pretrained model
    :param after_xblock_entry_2_lid: Tensor from pretrained model
    :param after_skip1_lid: Tensor from pretrained model
    :param after_xblock_entry_3_lid: Tensor from pretrained model
    :param after_xblock_middle: Tensor from pretrained model
    :param after_xblock_exit_1_lid: Tensor from pretrained model
    :param after_extracted_feature_lid: Tensor from pretrained model
    :return: output
    """
    entry_block3_stride = 2
    middle_block_rate = 1
    exit_block_rates = (1, 2)
    atrous_rate = (6, 12, 18)
    
    print("Training Start...")
    # Entry flow
    entry_conv1_img = Conv2D_Block(x_img, 32, filter_height=3, filter_width=3, stride=2, batch_normalization=True,
                               relu=True, name='entry_flow_conv1_1_BN_img')
    print("entry_conv1_img.shape", entry_conv1_img.shape)
    fuse1 = Concat([entry_conv1_img, after_entry_conv1_lid_sg], axis=-1, name='fusion1')
    entry_conv1_2_img = Conv2D_Block(fuse1, 64, filter_height=3, filter_width=3, stride=1,
                                 batch_normalization=True, relu=True, name='entry_flow_conv1_2_BN_img')

    fuse2 = Concat([entry_conv1_2_img, after_entry_conv1_2_lid_sg], axis=-1, name='fusion2')

    xblock_entry_1_img = Xception_block(fuse2, [128, 128, 128], 'entry_flow_block1_img',
                                    skip_connection_type='conv', stride=2, depth_activation=False)

    fuse3 = Concat([xblock_entry_1_img, after_xblock_entry_1_lid_sg], axis=-1, name='fusion3')

    xblock_entry_2_img, skip1_img = Xception_block(fuse3, [256, 256, 256], 'entry_flow_block2_img',
                                           skip_connection_type='conv', stride=2, depth_activation=False,
                                           return_skip=True)

    fuse4 = Concat([xblock_entry_2_img, after_xblock_entry_2_lid_sg], axis=-1, name='fusion4')

    xblock_entry_3_img = Xception_block(fuse4, [728, 728, 728], 'entry_flow_block3_img',
                                    skip_connection_type='conv', stride=entry_block3_stride,
                                    depth_activation=False)

    fuse5 = Concat([xblock_entry_3_img, after_xblock_entry_3_lid_sg], axis=-1, name='fusion5')

    fuse5 = Conv2D_Block(fuse5, 728, filter_height=1, filter_width=1, stride=1, padding='SAME', name='fusion4')
    for i in range(16):
        fuse5 = Xception_block(fuse5, [728, 728, 728], 'middle_flow_unit_{}_img'.format(i + 1),
                                        skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                        depth_activation=False)

    fuse6 = Concat([fuse5, after_xblock_middle_sg], axis=-1, name='fusion6')

    # Exit flow for RGB image
    xblock_exit_1_img = Xception_block(fuse6, [728, 728, 1024], 'exit_flow_block1_img',
                                   skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                                   depth_activation=False)

    fuse7 = Concat([xblock_exit_1_img, after_xblock_exit_1_lid_sg], axis=-1, name='fusion7')
    extracted_feature_img = Xception_block(fuse7, [1536, 1536, 2048], 'exit_flow_block2_img',
                                       skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                                       depth_activation=True)

    fuse8 = Concat([extracted_feature_img, after_extracted_feature_lid_sg], axis=-1, name='fusion8')

    # end of feature extractor

    # branching for Atrous Spatia Pyramid Pooling
    # Image Featrue branch
    b4 = Global_Avg_Pool(fuse8, name='b4')

    print("b4", b4.shape)

    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = tf.expand_dims(b4, dim=1)
    b4 = tf.expand_dims(b4, dim=1)

    print("after b4.shape", b4.shape)

    b4 = Conv2D_Block(b4, 256, filter_height=1, filter_width=1, stride=1, padding='SAME',
                          batch_normalization=True, relu=True, name='image_pooling_lid')

    print("1x1 conv b4.shape", b4.shape)

    # Upsampling. have to use compat because of the option align_corners
    size_before = tf.shape(extracted_feature_img)

    b4 = Resize_Bilinear(b4, size_before[1:3], name='upsampling_after_global_avg_pooling_lid')

    # simple 1x1
    b0 = Conv2D_Block(fuse8, 256, filter_height=1, filter_width=1, stride=1, padding='SAME',
                          batch_normalization=True, relu=True, name='aspp0_img')

    # rate = 6 (12)
    b1 = SepConv_BN(fuse8, 256, 'aspp1_lid', rate=atrous_rate[0], depth_activation=True,
                        epsilon=1e-5)

    # rate = 12 (24)
    b2 = SepConv_BN(fuse8, 256, 'aspp2_lid', rate=atrous_rate[1], depth_activation=True,
                        epsilon=1e-5)

    # rate = 18 (36)
    b3 = SepConv_BN(fuse8, 256, 'aspp3_lid', rate=atrous_rate[2], depth_activation=True,
                        epsilon=1e-5)

    # concatenate ASPP branches & project
    concatenated = Concat([b4, b0, b1, b2, b3], axis=-1, name='concatenation')

    concat_conv = Conv2D_Block(concatenated, 256, filter_height=1, filter_width=1, stride=1, padding='SAME',
                               batch_normalization=True, relu=True, name='concatenation_conv')
    concat_conv = Dropout(concat_conv, keep_prob=keep_prob)

    # Feature projection
    # x4 (x2) block
    size_before2 = tf.shape(xblock_entry_1_img)

    dec_up1 = Resize_Bilinear(concat_conv, size_before2[1:3], name="Upsampling2")

    fuse9 = Concat([skip1_img, after_skip1_lid_sg], axis=-1, name='concatenate_skips')
    dec_skip1 = Conv2D_Block(fuse9, 48, filter_height=1, filter_width=1, stride=1, padding='SAME',
                             batch_normalization=True, relu=True, name='feature_projection0')

    dec_concatenated = Concat([dec_up1, dec_skip1], axis=-1, name="concatenation_decoder0")

    dec_block1 = SepConv_BN(dec_concatenated, 256, 'decoder_conv0', depth_activation=True, epsilon=1e-5)
    dec_block2 = SepConv_BN(dec_block1, 256, 'decoer_conv1', depth_activation=True, epsilon=1e-5)

    last_layer = Conv2D_Block(dec_block2, num_classess_img, filter_height=1, filter_width=1, stride=1, padding='SAME', name='Last_layer')

    size_before3 = tf.shape(x_img)

    outputs = Resize_Bilinear(last_layer, size_before3[1:3], name="Last_Upsampling")

    return outputs

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
    # keep_probability = tf.placeholder(tf.float32, name="keep_probability")
    RGB_IMAGE = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 3],
                               name="input_rgb")
    RGB_LABEL = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], 2],
                               name="prediction_img")
    TENSOR_entry_conv1_lid = tf.placeholder(tf.float32,
                                            shape=[None, IMAGE_SHAPE_KITTI[0] / 2, IMAGE_SHAPE_KITTI[1] / 2, 32],
                                            name='tensor_entry_conv1')
    TENSOR_entry_conv1_2_lid = tf.placeholder(tf.float32,
                                              shape=[None, IMAGE_SHAPE_KITTI[0] / 2, IMAGE_SHAPE_KITTI[1] / 2, 64],
                                              name='tensor_entry_conv1_2')
    TENSOR_xblock_entry_1_lid = tf.placeholder(tf.float32,
                                               shape=[None, IMAGE_SHAPE_KITTI[0] / 4, IMAGE_SHAPE_KITTI[1] / 4, 128],
                                               name='tensor_xblock_entry_1')
    TENSOR_xblock_entry_2_lid = tf.placeholder(tf.float32,
                                               shape=[None, IMAGE_SHAPE_KITTI[0] / 8, IMAGE_SHAPE_KITTI[1] / 8, 256],
                                               name='tensor_xblock_entry_2')
    TENSOR_skip1_lid = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE_KITTI[0] / 4, IMAGE_SHAPE_KITTI[1] / 4, 256],
                                      name='tensor_skip_1')
    TENSOR_xblock_entry_3_lid = tf.placeholder(tf.float32,
                                               shape=[None, IMAGE_SHAPE_KITTI[0] / 16, IMAGE_SHAPE_KITTI[1] / 16, 728],
                                               name='tensor_xblock_entry_3')
    TENSOR_xblock_middle = tf.placeholder(tf.float32,
                                          shape=[None, IMAGE_SHAPE_KITTI[0] / 16, IMAGE_SHAPE_KITTI[1] / 16, 728],
                                          name='tensor_xblock_middle')
    TENSOR_xblock_exit_1_lid = tf.placeholder(tf.float32,
                                              shape=[None, IMAGE_SHAPE_KITTI[0] / 16, IMAGE_SHAPE_KITTI[1] / 16, 1024],
                                              name='tensor_xblock_exit_1')
    TENSOR_extracted_feature_lid = tf.placeholder(tf.float32,
                                                  shape=[None, IMAGE_SHAPE_KITTI[0] / 16, IMAGE_SHAPE_KITTI[1] / 16,
                                                         2048], name='tensor_extracted_feature')
    TENSOR_output_lid = tf.placeholder(tf.float32,
                                       shape=[None, IMAGE_SHAPE_KITTI[0], IMAGE_SHAPE_KITTI[1], NUM_OF_CLASSESS_LID],
                                       name='tensor_output_lid')
    # Load Graph
    graph = load_graph("./frozen_model.pb")
    
    # List of operations
    for op in graph.get_operations():
        print(op.name)

    # Access the input and output nodes
    # Encoder output nodes in generator models
    input_lidar = graph.get_tensor_by_name('input_lidar:0')
    output_lid = graph.get_tensor_by_name('output:0')
    keep_probability = graph.get_tensor_by_name('keep_probability:0')
    after_entry_conv1_lid = graph.get_tensor_by_name('Relu:0')
    after_entry_conv1_2_lid = graph.get_tensor_by_name('Relu_2:0')
    after_xblock_entry_1_lid = graph.get_tensor_by_name('entry_flow_block1_lid_add_conv:0')
    after_xblock_entry_2_lid = graph.get_tensor_by_name('entry_flow_block2_lid_add_conv:0')
    after_skip1_lid = graph.get_tensor_by_name('Relu_11:0')
    after_xblock_entry_3_lid = graph.get_tensor_by_name('entry_flow_block3_add_conv:0')
    after_xblock_middle = graph.get_tensor_by_name('middle_flow_unit_16_lid_add_sum:0')
    after_xblock_exit_1_lid = graph.get_tensor_by_name('exit_flow_block1_lid_add_conv:0')
    after_extracted_feature_lid = graph.get_tensor_by_name('Relu_127:0')

    # Network 선언
    logits = GSNet(RGB_IMAGE, KEEP_PROBABILITY, NUM_OF_CLASSESS_IMG, TENSOR_entry_conv1_lid,
                   TENSOR_entry_conv1_2_lid, TENSOR_xblock_entry_1_lid, TENSOR_xblock_entry_2_lid, TENSOR_skip1_lid,
                   TENSOR_xblock_entry_3_lid, TENSOR_xblock_middle, TENSOR_xblock_exit_1_lid,
                   TENSOR_extracted_feature_lid)

    # Tensorboard를 위한 summary들을 지정
    tf.summary.image('input_image', RGB_IMAGE, max_outputs=2)

    # 손실 함수를 선언하고 손실 함수에 대한 summary들을 지정
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=RGB_LABEL))

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
    sess = tf.Session(config=config, graph=graph)
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
        costs = 0
        count = 0
        # 학습 데이터를 불러오고 feed_dict에 데이터를 지정
        for images, labels, lidar in get_batches_fn(batch_size=BATCH_SIZE):
            with tf.Session(graph=graph) as sess_generator:
                feed_dict = {keep_probability: 1.0, input_lidar: lidars}
                tensor_entry_conv1_lid, tensor_entry_conv1_2_lid, tensor_xblock_entry_1_lid, tensor_xblock_entry_2_lid, \
                tensor_skip1_lid, tensor_xblock_entry_3_lid, tensor_xblock_middle, tensor_xblock_exit_1_lid, \
                tensor_extracted_feature_lid, tensor_output_lid = sess_generator.run(
                    [after_entry_conv1_lid, after_entry_conv1_2_lid, after_xblock_entry_1_lid, after_xblock_entry_2_lid,
                     after_skip1_lid, after_xblock_entry_3_lid, after_xblock_middle, after_xblock_exit_1_lid,
                     after_extracted_feature_lid, output_lid], feed_dict=feed_dict)
                
            feed_dict = {RGB_IMAGE: images, RGB_LABEL: labels, KEEP_PROBABILITY: KEEP_PROB,
                         TENSOR_entry_conv1_lid: tensor_entry_conv1_lid,
                         TENSOR_entry_conv1_2_lid: tensor_entry_conv1_2_lid,
                         TENSOR_xblock_entry_1_lid: tensor_xblock_entry_1_lid,
                         TENSOR_xblock_entry_2_lid: tensor_xblock_entry_2_lid,
                         TENSOR_skip1_lid: tensor_skip1_lid,
                         TENSOR_xblock_entry_3_lid: tensor_xblock_entry_3_lid,
                         TENSOR_xblock_middle: tensor_xblock_middle,
                         TENSOR_xblock_exit_1_lid: tensor_xblock_exit_1_lid,
                         TENSOR_extracted_feature_lid: tensor_extracted_feature_lid,
                         TENSOR_output_lid: tensor_output_lid}

            # Initialize the accumulated grads
            sess.run(zero_ops)
            for i in range(len(images)):
                sess.run(accum_ops, feed_dict=feed_dict)

            # train_step을 실행해서 파라미터를 한 스텝 업데이트 함
            _, cost = sess.run([train_step, loss], feed_dict=feed_dict)
            
            costs += cost
            count += 1
            
            print("loss:", loss)
            
            # Tensorboard 를 위한 sess.run()
            summary = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step=sess.run(global_step))

        print("[Epoch: {0}/{1} Time: {2}, loss: {3}]".format(epoch + 1, EPOCHS, str(timedelta(seconds=(time.time() - s_time))), costs / count))

    print("Time: ", time.time() - start)  # 현재 시각 - 시작 시간 = 실행 시간
    print("Training Successfully")

    # 훈련이 끝나고 학습된 파라미터 저장
    saver.save(sess, './model/GSNet.ckpt', global_step=global_step)

    # 훈련이 끝나고 테스트 데이터 셋으로 테스트
    output_dir = os.path.join(DATA_DIR, 'output')
    mask_dir = os.path.join(DATA_DIR, 'mask')
    print("Training Finished. Saving test images to: {}".format(output_dir))
    with tf.Session(graph=graph) as sess_generator:
        image_output = gen_test_output(sess, sess_generator, logits, KEEP_PROBABILITY, RGB_IMAGE, input_lidar,
                                       keep_probability, TENSOR_entry_conv1_lid, TENSOR_entry_conv1_2_lid,
                                       TENSOR_xblock_entry_1_lid, TENSOR_xblock_entry_2_lid, TENSOR_skip1_lid,
                                       TENSOR_xblock_entry_3_lid, TENSOR_xblock_middle, TENSOR_xblock_exit_1_lid,
                                       TENSOR_extracted_feature_lid, TENSOR_output_lid, after_entry_conv1_lid,
                                       after_entry_conv1_2_lid, after_xblock_entry_1_lid, after_xblock_entry_2_lid,
                                       after_skip1_lid, after_xblock_entry_3_lid, after_xblock_middle,
                                       after_xblock_exit_1_lid, after_extracted_feature_lid, output_lid,
                                       os.path.join(DATA_DIR, 'validating'), IMAGE_SHAPE_KITTI)

        total_processing_time = 0
        for name, mask, image, processing_time in image_output:
            scipy.misc.imsave(os.path.join(output_dir, name), image)
            scipy.misc.imsave(os.path.join(mask_dir, name), mask)
            total_processing_time += processing_time
    
        print("Average processing time is : ", total_processing_time / 30)


if __name__ == '__main__':
    run()
