import os
import argparse


def setParse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='/data/irfan/auto_painter/nb_girls/all_sketch/',
                        help="path to folder containing images")
    parser.add_argument("--mode", default='train', choices=["train", "test", "export"])
    parser.add_argument("--output_dir", default='/data/irfan/auto_painter/nb_girls/result/wgan_auto_2_v2',
                        help="where to put output files")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", help="directory with checkpoint to resume training from or use for testing")
    parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
    parser.add_argument("--max_epochs", default=50, type=int, help="number of training epochs")
    parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
    parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
    parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
    parser.add_argument("--display_freq", type=int, default=0,
                        help="write current training images every display_freq steps")
    parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
    parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
    parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
    parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
    parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
    parser.add_argument("--scale_size", type=int, default=530,
                        help="expand images to this size before cropping to 256x256")
    parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
    parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
    parser.set_defaults(flip=True)
    parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
    parser.add_argument("--l1_weight", type=float, default=10.0, help="weight on L1 term for generator gradient")
    parser.add_argument("--gan_weight", type=float, default=1, help="weight on GAN term for generator gradient")
    parser.add_argument("--tv_weight", type=float, default=1e-5, help="weight on tv term for generator gradient")
    parser.add_argument("--f_weight", type=float, default=1e-4, help="weight on f term for generator gradient")
    # export options
    parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
    return parser.parse_args()


def setHP(a):
    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)
    #tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)






def train(HP):
    EPS = 1e-12
    CROP_SIZE = 512
    Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
    Model = collections.namedtuple("Model",
                                   "outputs, predict_real, predict_fake, disc_loss, disc_grads_and_vars, gen_loss_GAN,gen_loss_tv,gen_loss_f gen_loss_L1, gen_grads_and_vars, train")


if __name__ == '__main__':
    train(setHP(setParse()))
