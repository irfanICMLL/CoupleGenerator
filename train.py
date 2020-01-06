import numpy as np
import argparse
import random
from utils.files import *
from models.wgan import *
from models.my_vgg import vgg16


def set_gpu(a):
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="path to folder containing images",
                        default="./dataset/")
    parser.add_argument("--output_dir", help="where to put output files",
                        default="./output/")
    parser.add_argument("--checkpoint", help="directory with checkpoint to resume training from or use for testing")
    parser.add_argument("--random_seed", type=int, help="global random seed")
    parser.add_argument("--mode", default='train', choices=["train", "test", "export"])
    parser.add_argument("--max_epochs", default=50, type=int, help="number of training epochs")
    parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
    parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
    parser.add_argument("--leak", type=float, default=0.2, help="weight of leak part in LReLU function")
    parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
    parser.add_argument("--ngf", type=int, default=64, help="number of filters in the first layer of generator")
    parser.add_argument("--ndf", type=int, default=64, help="number of filters in the first layer of discriminator ")
    parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
    parser.add_argument("--l1_weight", type=float, default=10.0, help="weight on L1 term for generator gradient")
    parser.add_argument("--gan_weight", type=float, default=1, help="weight on GAN term for generator gradient")
    parser.add_argument("--tv_weight", type=float, default=1e-5, help="weight on tv term for generator gradient")
    parser.add_argument("--f_weight", type=float, default=1e-4, help="weight on f term for generator gradient")
    parser.add_argument("--EPS", type=float, default=1e-12)
    parser.add_argument("--CROP_SIZE", type=int, default=512)
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--gpu", help="on which gpu the model running", default="0")
    parser.add_argument("--upsampling", type=bool, default=False,
                        help="whether to use upsampling instead of deconvolution")

    arg = parser.parse_args()

    if arg.random_seed is None:
        arg.random_seed = random.randint(0, 2 ** 31 - 1)
    tf.set_random_seed(arg.random_seed)
    np.random.seed(arg.random_seed)
    random.seed(arg.random_seed)

    if not os.path.exists(arg.output_dir):
        os.makedirs(arg.output_dir)

    return arg


def set_summary(m):
    tf.summary.scalar("discriminator_loss", m.disc_loss)
    tf.summary.scalar("generator_loss_GAN", m.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", m.gen_loss_L1)
    tf.summary.scalar("generator_loss_tv", m.gen_loss_tv)
    tf.summary.scalar("generator_loss_f", m.gen_loss_f)


if __name__ == "__main__":
    a = set_parser()
    conf = set_gpu(a)

    if a.mode == "test" or a.mode == "export":
        a = load_json(a)
    for k, v in a._get_kwargs():
        print(k, "=", v)
    write_json(a)

    if a.mode == "export":
        input = tf.placeholder(tf.string, shape=1)
        input_data = tf.decode_base64(input[0])
        input_data = tf.image.decode_png(input_data)[:, :, :3]
        input_data = tf.image.convert_image_dtype(input_data, dtype=tf.float32)
        input_data.set_shape([a.CROP_SIZE, a.CROP_SIZE, 3])
        input_data = tf.expand_dims(input_data, axis=0)

        with tf.variable_scope("generator"):
            output_data = shrink(generator(expand(input_data), 3, a))
        output_data = tf.image.convert_image_dtype(output_data, dtype=tf.uint8)[0]
        output_data = tf.image.encode_png(output_data)
        output = tf.convert_to_tensor([tf.encode_base64(output_data)])

        key = tf.placeholder(tf.string, shape=[1])
        inputs = {"key": key.name, "input": input.name}
        tf.add_to_collection("inputs", json.dumps(inputs))
        outputs = {"key": tf.identity(key).name, "input": input.name}
        tf.add_to_collection("outputs", json.dumps(outputs))

        with tf.Session(config=conf) as sess:
            restore_saver = tf.train.Saver()
            export_saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            print("Loading models from checkpoint...")
            restore_saver.save(sess, tf.train.latest_checkpoint(a.checkpoint))
            print("Exporting model...")
            export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))
            export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)

    else:
        examples = load_examples(a)
        net1 = vgg16.Vgg16()
        net2 = vgg16.Vgg16()
        model = create_model(examples.inputs, examples.targets, net1, net2, a)
        inputs = tf.image.convert_image_dtype(shrink(examples.inputs), dtype=tf.uint8, saturate=True)
        targets = tf.image.convert_image_dtype(shrink(examples.targets), dtype=tf.uint8, saturate=True)
        outputs = tf.image.convert_image_dtype(shrink(model.outputs), dtype=tf.uint8, saturate=True)
        output_images = tf.map_fn(tf.image.encode_png, outputs, dtype=tf.string)

        set_summary(model)

        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        train_saver = tf.train.Saver(max_to_keep=1)
        supervisor = tf.train.Supervisor(logdir=a.output_dir, save_model_secs=0, saver=None)
        with supervisor.managed_session(config=conf) as sess:
            print("Numbers of parameters: ", sess.run(parameter_count))
            if a.checkpoint is not None:
                print("Loading model from checkpoint...")
                train_saver.restore(sess, tf.train.latest_checkpoint(a.checkpoint))
            if a.mode == "test":
                for step in range(examples.steps_per_epoch):
                    paths, results = sess.run([examples.paths, output_images])
                    image_dir = os.path.join(a.output_dir, "images")
                    if not os.path.exists(image_dir):
                        os.makedirs(image_dir)
                    for i, path in enumerate(paths):
                        filename = get_name(path.decode("utf8"))
                        with open(os.path.join(image_dir, filename + ".png"), "wb") as f:
                            f.write(results[i])
                        print(filename, "evaluated.")
            else:
                def display_loss():
                    print("Epoch %d  Step %d" % (epoch, step))
                    print("Loss on Discriminator:", sess.run(model.disc_loss))
                    print("GAN Loss on Generator:", sess.run(model.gen_loss_GAN))
                    print("L1 Loss on Generator:", sess.run(model.gen_loss_L1))
                    print("TV Loss on Generator:", sess.run(model.gen_loss_tv))
                    print("F Loss on Generator:", sess.run(model.gen_loss_f))
                global_step = sess.run(supervisor.global_step)
                epoch = global_step // examples.steps_per_epoch
                step = global_step % examples.steps_per_epoch
                while epoch < a.max_epochs:
                    while step < examples.steps_per_epoch:
                        if supervisor.should_stop():
                            break
                        sess.run(model.train)
                        global_step = sess.run(supervisor.global_step)
                        if (global_step + 1) % a.progress_freq == 0:
                            display_loss()
                        if (global_step + 1) % a.save_freq == 0:
                            print("Saving...")
                            train_saver.save(sess, os.path.join(a.output_dir, "model"), global_step=global_step)
                        step = step + 1
                    step = 0
                    epoch = epoch + 1
                display_loss()
                train_saver.save(sess, os.path.join(a.output_dir, "model"), global_step=global_step)
