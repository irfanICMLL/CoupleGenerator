import numpy as np
import tensorflow as tf

from tensorflow.contrib.slim import layers


# def get_vgg(input):
#     data_dict = np.load('vgg19.npy').item()
#     tf_constant = {}
#     for k in ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4']:
#         tf_constant['%s_filter'%k] = tf.constant(data_dict['%s_weight'%k], name='%s_filter'%k)
#         tf_constant['%s_filter'%k] = tf.transpose(tf_constant['%s_filter'%k], perm=[2,3,1,0])
#         tf_constant['%s_bias'%k] = tf.constant(data_dict['%s_bias'%k], name='%s_bias'%k)
#
#     def conv_layer(bottom, name):
#         with tf.variable_scope(name):
#             conv = tf.nn.conv2d(bottom, tf_constant['%s_filter'%name], [1, 1, 1, 1], padding='SAME')
#             bias = tf.nn.bias_add(conv, tf_constant['%s_bias'%name])
#             relu = tf.nn.relu(bias)
#             return relu
#     def max_pool(bottom, name):
#         return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
#                               padding='SAME', name=name)
#     def avg_pool(bottom, name):
#         return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
#                               padding='SAME', name=name)
#     conv1_1 = conv_layer(input, 'conv1_1')
#     conv1_2 = conv_layer(conv1_1, 'conv1_2')
#     pool1 = avg_pool(conv1_2, 'pool1')
#     conv2_1 = conv_layer(pool1, 'conv2_1')
#     conv2_2 = conv_layer(conv2_1, 'conv2_2')
#     pool2 = avg_pool(conv2_2, 'pool2')
#     conv3_1 = conv_layer(pool2, 'conv3_1')
#     return conv3_1


def convs(data, num_filter, kernel, stride, pad, name):
    with tf.variable_scope(name):
        filter = tf.get_variable('conv0_filter', [kernel, kernel, int(data.shape[3]), 1], tf.float32)
        data = tf.nn.depthwise_conv2d(data, filter, [1, stride, stride, 1], 'SAME', name='conv0')
        data = layers.layer_norm(data, scale=False)
        data = tf.nn.relu(data)
        data = tf.layers.conv2d(data, num_filter, (1, 1), (1, 1), 'valid')
        data = layers.layer_norm(data, scale=False)
        data = tf.nn.relu(data)
        return data


def conv(data, num_filter, kernel, stride, pad, name):
    with tf.variable_scope(name):
        data = tf.layers.conv2d(data, num_filter, (kernel, kernel), (stride, stride), 'same')
        data = layers.layer_norm(data, scale=False)
        data = tf.nn.relu(data)
        return data


def deconv(data, num_filter, kernel, stride, pad, name):
    with tf.variable_scope(name):
        data = tf.image.resize_bilinear(data, [int(x) for x in [data.shape[1] * 2, data.shape[2] * 2]])
        #     data = conv(data, num_filter, kernel, 1, pad, name)
        return data


def generator(data):
    data1 = conv(data, 32, 7, 2, 3, 'conv1')
    data2 = conv(data1, 32, 3, 1, 1, 'conv2')
    data3 = conv(data2, 64, 3, 2, 1, 'conv3')
    data4 = conv(data3, 64, 3, 1, 1, 'conv4')
    data5 = convs(data4, 96, 3, 2, 1, 'conv5')
    data6 = convs(data5, 96, 3, 1, 1, 'conv6')
    data7 = convs(data6, 144, 3, 2, 1, 'conv7')
    data8 = convs(data7, 144, 3, 1, 1, 'conv8')
    data9 = convs(data8, 192, 3, 2, 1, 'conv9')
    data10 = convs(data9, 192, 3, 1, 1, 'conv10')
    udata9 = convs(data10, 192, 3, 1, 1, 'uconv9')
    udata9 = tf.concat([udata9, data9], axis=3)
    udata8 = deconv(udata9, 144, 3, 2, 1, 'uconv8')
    udata8 = tf.concat([udata8, data8], axis=3)
    udata7 = convs(udata8, 144, 3, 1, 1, 'uconv7')
    udata7 = tf.concat([udata7, data7], axis=3)
    udata6 = deconv(udata7, 96, 3, 2, 1, 'uconv6')
    udata6 = tf.concat([udata6, data6], axis=3)
    udata5 = convs(udata6, 96, 3, 1, 1, 'uconv5')
    udata5 = tf.concat([udata5, data5], axis=3)
    udata4 = deconv(udata5, 64, 3, 2, 1, 'uconv4')
    udata4 = tf.concat([udata4, data4], axis=3)
    udata3 = conv(udata4, 64, 3, 1, 1, 'uconv3')
    udata3 = tf.concat([udata3, data3], axis=3)
    udata2 = deconv(udata3, 32, 3, 2, 1, 'uconv2')
    udata1 = conv(udata2, 32, 3, 1, 1, 'uconv1')
    output = deconv(udata1, 32, 3, 2, 1, 'outconv1')
    output = conv(output, 32, 3, 1, 1, 'outconv2')
    output = tf.layers.conv2d(output, 3, (3, 3), (1, 1), 'SAME')
    output = tf.nn.tanh(output, 'final_out')
    return output * 128


def discriminator(data):
    data1 = conv(data, 32, 7, 2, 3, 'dconv1')
    data2 = conv(data1, 32, 3, 1, 1, 'dconv2')
    data3 = conv(data2, 64, 3, 2, 1, 'dconv3')
    data4 = conv(data3, 64, 3, 1, 1, 'dconv4')
    data5 = conv(data4, 96, 3, 2, 1, 'dconv5')
    data6 = conv(data5, 96, 3, 1, 1, 'dconv6')
    data7 = conv(data6, 144, 3, 2, 1, 'dconv7')
    #     data8 = conv(data7, 2, 3, 1, 1, 'dconv8')
    #     data8 = conv(data7, 1, 3, 1, 1, 'dconv8')
    with tf.variable_scope('dconv8'):
        data8 = tf.layers.conv2d(data7, 1, (3, 3), (1, 1), 'same')
    return data8
