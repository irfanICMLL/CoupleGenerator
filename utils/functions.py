import tensorflow as tf


def expand(image):
    # [0, 1] => [-1, 1]
    return image * 2 - 1


def shrink(image):
    # [-1, 1] => [0, 1]
    return (image + 1) / 2


def lrelu(x, a):
    # adding these together creates the leak part and linear part
    # then cancels them out by subtracting/adding an absolute value term
    # leak: a*x/2 - a*abs(x)/2
    # linear: x/2 + abs(x)/2
    # this block looks like it has 2 inputs on the graph unless we do this
    x = tf.identity(x)
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def sum_tv_loss(image):
    loss_y = tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :-1, :, :])
    loss_x = tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :-1, :])
    return tf.cast(2 * (loss_y + loss_x), tf.float32)


def feature_loss(image, vgg):
    vgg.build(image)
    return vgg.conv3_3 + vgg.conv3_2 + vgg.conv3_1


def batchnorm(data, n):
    with tf.variable_scope("batchnorm_%d" % n):
        # this block looks like it has 3 inputs on the graph unless we do this
        data = tf.identity(data)

        channels = data.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("expand", [channels], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(data, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(data, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels, n):
    with tf.variable_scope("deconv_%d" % n):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        fil = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        output = tf.nn.conv2d_transpose(batch_input, fil, [batch, in_height * 2, in_width * 2, out_channels],
                                        [1, 2, 2, 1], padding="SAME")
        return output


def conv(batch_input, out_channels, stride, n):
    with tf.variable_scope("conv_%d" % n):
        in_channels = batch_input.get_shape()[3]
        fil = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        output = tf.nn.conv2d(padded_input, fil, [1, stride, stride, 1], padding="VALID")
        return output


def upsampling(batch_input, out_channels, n):
    with tf.variable_scope("upsample_%d" % n):
        resized_input = tf.image.resize_bilinear(batch_input, [int(x) for x in [batch_input.shape[1] * 2,
                                                                                batch_input.shape[2] * 2]])
        in_channels = batch_input.get_shape()[3]
        fil = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.02))
        output = tf.nn.conv2d(resized_input, fil, [1, 1, 1, 1], padding="SAME")
        return output
