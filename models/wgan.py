import collections
from utils.functions import *


def generator_deconv(generator_inputs, generator_outputs_channels, a):
    encoder1 = conv(generator_inputs, a.ngf, stride=2, n=1)
    act1 = lrelu(encoder1, a.leak)
    encoder2 = conv(act1, a.ngf * 2, stride=2, n=2)
    encoder2 = batchnorm(encoder2, n=2)
    act2 = lrelu(encoder2, a.leak)
    encoder3 = conv(act2, a.ngf * 4, stride=2, n=3)
    encoder3 = batchnorm(encoder3, n=3)
    act3 = lrelu(encoder3, a.leak)
    encoder4 = conv(act3, a.ngf * 8, stride=2, n=4)
    encoder4 = batchnorm(encoder4, n=4)
    act4 = lrelu(encoder4, a.leak)
    encoder5 = conv(act4, a.ngf * 8, stride=2, n=5)
    encoder5 = batchnorm(encoder5, n=5)
    act5 = lrelu(encoder5, a.leak)
    encoder6 = conv(act5, a.ngf * 8, stride=2, n=6)
    encoder6 = batchnorm(encoder6, n=6)
    act6 = lrelu(encoder6, a.leak)
    encoder7 = conv(act6, a.ngf * 8, stride=2, n=7)
    encoder7 = batchnorm(encoder7, n=7)
    act7 = lrelu(encoder7, a.leak)
    encoder8 = conv(act7, a.ngf * 8, stride=2, n=8)
    encoder8 = batchnorm(encoder8, n=8)
    act8 = lrelu(encoder8, a.leak)
    decoder8 = deconv(act8, a.ngf * 8, n=8)
    decoder8 = batchnorm(decoder8, n=9)
    decoder8 = tf.nn.dropout(decoder8, keep_prob=0.5)
    act9 = lrelu(tf.concat([decoder8, encoder7], axis=3), a.leak)
    decoder7 = deconv(act9, a.ngf * 8, n=7)
    decoder7 = batchnorm(decoder7, n=10)
    decoder7 = tf.nn.dropout(decoder7, keep_prob=0.5)
    act10 = lrelu(tf.concat([decoder7, encoder6], axis=3), a.leak)
    decoder6 = deconv(act10, a.ngf * 8, n=6)
    decoder6 = batchnorm(decoder6, n=11)
    decoder6 = tf.nn.dropout(decoder6, keep_prob=0.5)
    act11 = lrelu(tf.concat([decoder6, encoder5], axis=3), a.leak)
    decoder5 = deconv(act11, a.ngf * 8, n=5)
    decoder5 = batchnorm(decoder5, n=12)
    act12 = lrelu(tf.concat([decoder5, encoder4], axis=3), a.leak)
    decoder4 = deconv(act12, a.ngf * 4, n=4)
    decoder4 = batchnorm(decoder4, n=13)
    act13 = lrelu(tf.concat([decoder4, encoder3], axis=3), a.leak)
    decoder3 = deconv(act13, a.ngf * 2, n=3)
    decoder3 = batchnorm(decoder3, n=14)
    act14 = lrelu(tf.concat([decoder3, encoder2], axis=3), a.leak)
    decoder2 = deconv(act14, a.ngf, n=2)
    decoder2 = batchnorm(decoder2, n=15)
    act15 = tf.nn.relu(tf.concat([decoder2, encoder1], axis=3))
    decoder1 = deconv(act15, generator_outputs_channels, n=1)
    act16 = tf.tanh(decoder1)
    return act16


def generator_upsampling(generator_inputs, generator_outputs_channels, a):
    encoder1 = conv(generator_inputs, a.ngf, stride=2, n=1)
    act1 = lrelu(encoder1, a.leak)
    encoder2 = conv(act1, a.ngf * 2, stride=2, n=2)
    encoder2 = batchnorm(encoder2, n=2)
    act2 = lrelu(encoder2, a.leak)
    encoder3 = conv(act2, a.ngf * 4, stride=2, n=3)
    encoder3 = batchnorm(encoder3, n=3)
    act3 = lrelu(encoder3, a.leak)
    encoder4 = conv(act3, a.ngf * 8, stride=2, n=4)
    encoder4 = batchnorm(encoder4, n=4)
    act4 = lrelu(encoder4, a.leak)
    encoder5 = conv(act4, a.ngf * 8, stride=2, n=5)
    encoder5 = batchnorm(encoder5, n=5)
    act5 = lrelu(encoder5, a.leak)
    encoder6 = conv(act5, a.ngf * 8, stride=2, n=6)
    encoder6 = batchnorm(encoder6, n=6)
    act6 = lrelu(encoder6, a.leak)
    encoder7 = conv(act6, a.ngf * 8, stride=2, n=7)
    encoder7 = batchnorm(encoder7, n=7)
    act7 = lrelu(encoder7, a.leak)
    encoder8 = conv(act7, a.ngf * 8, stride=2, n=8)
    encoder8 = batchnorm(encoder8, n=8)
    act8 = lrelu(encoder8, a.leak)
    decoder8 = upsampling(act8, a.ngf * 8, n=8)
    decoder8 = batchnorm(decoder8, n=9)
    decoder8 = tf.nn.dropout(decoder8, keep_prob=0.5)
    act9 = lrelu(tf.concat([decoder8, encoder7], axis=3), a.leak)
    decoder7 = upsampling(act9, a.ngf * 8, n=7)
    decoder7 = batchnorm(decoder7, n=10)
    decoder7 = tf.nn.dropout(decoder7, keep_prob=0.5)
    act10 = lrelu(tf.concat([decoder7, encoder6], axis=3), a.leak)
    decoder6 = upsampling(act10, a.ngf * 8, n=6)
    decoder6 = batchnorm(decoder6, n=11)
    decoder6 = tf.nn.dropout(decoder6, keep_prob=0.5)
    act11 = lrelu(tf.concat([decoder6, encoder5], axis=3), a.leak)
    decoder5 = upsampling(act11, a.ngf * 8, n=5)
    decoder5 = batchnorm(decoder5, n=12)
    act12 = lrelu(tf.concat([decoder5, encoder4], axis=3), a.leak)
    decoder4 = upsampling(act12, a.ngf * 4, n=4)
    decoder4 = batchnorm(decoder4, n=13)
    act13 = lrelu(tf.concat([decoder4, encoder3], axis=3), a.leak)
    decoder3 = upsampling(act13, a.ngf * 2, n=3)
    act14 = lrelu(tf.concat([decoder3, encoder2], axis=3), a.leak)
    decoder2 = upsampling(act14, a.ngf, n=2)
    decoder2 = batchnorm(decoder2, n=15)
    act15 = tf.nn.relu(tf.concat([decoder2, encoder1], axis=3))
    decoder1 = upsampling(act15, generator_outputs_channels, n=1)
    act16 = tf.tanh(decoder1)
    return act16


def generator(generator_inputs, generator_outputs_channels, a):
    if a.upsampling:
        return generator_upsampling(generator_inputs, generator_outputs_channels, a)
    else:
        return generator_deconv(generator_inputs, generator_outputs_channels, a)


def discriminator(disc_inputs, disc_targets, a):
    pair = tf.concat([disc_inputs, disc_targets], axis=3)
    encoder1 = conv(pair, a.ndf, stride=2, n=1)
    act1 = lrelu(encoder1, a.leak)
    encoder2 = conv(act1, a.ndf * 2, stride=2, n=2)
    encoder2 = batchnorm(encoder2, n=2)
    act2 = lrelu(encoder2, a.leak)
    encoder3 = conv(act2, a.ndf * 4, stride=2, n=3)
    encoder3 = batchnorm(encoder3, n=3)
    act3 = lrelu(encoder3, a.leak)
    encoder4 = conv(act3, a.ndf * 8, stride=1, n=4)
    encoder4 = batchnorm(encoder4, n=4)
    act4 = lrelu(encoder4, a.leak)
    encoder5 = conv(act4, 1, stride=1, n=5)
    act5 = tf.sigmoid(encoder5)
    return act5


def create_model(inputs, targets, net1, net2, a):

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = generator(inputs, out_channels, a)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = discriminator(inputs, targets, a)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = discriminator(inputs, outputs, a)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        disc_loss = tf.reduce_mean(predict_fake - predict_real)

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = -tf.reduce_mean(predict_fake)
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss_tv = tf.reduce_mean(tf.sqrt(tf.nn.l2_loss(sum_tv_loss(outputs))))
        gen_loss_f = tf.reduce_mean(tf.sqrt(tf.nn.l2_loss(feature_loss(targets, net1) - feature_loss(outputs, net2))))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight + gen_loss_tv * a.tv_weight + gen_loss_f * a.f_weight

    with tf.name_scope("discriminator_train"):
        disc_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        disc_optimizer = tf.train.RMSPropOptimizer(a.lr, a.beta1)
        disc_grads_and_vars = disc_optimizer.compute_gradients(disc_loss, var_list=disc_tvars)
        disc_train = disc_optimizer.apply_gradients(disc_grads_and_vars)
        clip_vars = [tf.assign(var, tf.clip_by_value(var, -0.02, 0.02)) for var in disc_tvars]
        tuple_vars = tf.tuple(clip_vars, control_inputs=[disc_train])

    with tf.name_scope("generator_train"):
        with tf.control_dependencies(tuple_vars):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.RMSPropOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([disc_loss, gen_loss_GAN, gen_loss_L1, gen_loss_tv, gen_loss_f])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return collections.namedtuple("Model", "predict_real, predict_fake, "
                                           "disc_loss, disc_grads_and_vars, "
                                           "gen_loss_GAN, gen_loss_tv, gen_loss_f, gen_loss_L1, gen_grads_and_vars, "
                                           "outputs, train")(
        predict_real=predict_real,
        predict_fake=predict_fake,
        disc_loss=ema.average(disc_loss),
        disc_grads_and_vars=disc_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_loss_tv=ema.average(gen_loss_tv),
        gen_loss_f=ema.average(gen_loss_f),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )
