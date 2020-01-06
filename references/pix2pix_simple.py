import os

import numpy as np
import tensorflow as tf
from skimage import io

from references import pix2pix_symbol as symbol

print(tf.__version__)
np.set_printoptions(precision=2)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main(task):
    tf.reset_default_graph()
    try:
        os.mkdir('wgan_mobile_%s' % task)
    except:
        pass
    batch = 1
    n_iter = 50000
    n_crit = 5
    lr = 1e-3
    cw = 1e-2
    tv_weight = 1e-9
    gan_weight = 1e0

    def load_example():
        img_dir = './sketches_input'
        img_dir2 = './sketches_target'
        imgs = os.listdir(img_dir2)[:]
        with tf.name_scope('load_image'):
            image_queue = tf.train.slice_input_producer([imgs])[0]
            im1 = img_dir + image_queue
            im2 = img_dir2 + image_queue
            im1 = tf.read_file(im1)
            im1 = tf.image.decode_image(im1)
            im1 = tf.cast(im1, tf.float32)
            im1 -= 128
            im1.set_shape([256, 256, 3])
            im2 = tf.read_file(im2)
            im2 = tf.image.decode_image(im2)
            im2 = tf.cast(im2, tf.float32)
            im2 -= 128
            im2.set_shape([256, 256, 3])
            im1, im2 = tf.train.batch([im1, im2], batch, 4)
        return im1, im2

    im1, im2 = load_example()
    with tf.variable_scope('generator', initializer=tf.truncated_normal_initializer(0, 0.02)):
        g_out = symbol.generator(im1)
    fake_and_real = tf.concat([g_out, im2], 0)
    # vgg_out = symbol.get_vgg(fake_and_real)
    # vgg_fake, vgg_real = tf.split(vgg_out, [batch, batch], 0)
    with tf.variable_scope('discriminator', initializer=tf.truncated_normal_initializer(0, 0.02)):
        dis_out = symbol.discriminator(fake_and_real)
    dis_fake, dis_real = tf.split(dis_out, [batch, batch], 0)
    # feature_loss = tf.reduce_mean(tf.square(vgg_fake-vgg_real))
    gan_real = tf.reduce_mean(dis_real)
    gan_fake = tf.reduce_mean(dis_fake)
    gan_loss = (gan_fake - gan_real) * gan_weight
    tv_loss = tf.reduce_mean(tf.image.total_variation(g_out)) * tv_weight
    #  loss = feature_loss
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train = tf.train.AdamOptimizer(lr).minimize(-gan_loss,
                                                      var_list=[x for x in tf.global_variables() if
                                                                x.name.startswith('discriminator')])
        g_train = tf.train.AdamOptimizer(lr).minimize(gan_loss + tv_loss,
                                                      var_list=[x for x in tf.global_variables() if
                                                                x.name.startswith('generator')])
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(n_iter + 1):
            #             print(i)
            sess.run([g_train, d_train])
            if i % 5 == 0:
                x = sess.run([g_train, d_train, g_out, gan_loss, tv_loss, gan_fake, gan_real])
                print('Step %6d, Loss %8.4f, GanL %8.4f, DisF %8.4f, DisR %8.4f, TVL %8.4f' % (
                    i, x[3], x[4], x[5], x[7], x[6]))
                if i % 500 == 0:
                    io.imsave('wgan_mobile_%s/tmp_%d.png' % (task, i), (x[2][0] + 128).astype(np.uint8))
            else:
                sess.run([g_train, d_train])
            if i % 5000 == 0:
                checkpoint_path = saver.save(sess, 'wgan_mobile_%s/model' % task, global_step=0)


for task in ['debug']:
    main(task)
