import os
import json
import glob
import collections
import math
from utils.functions import *


def load_json(a):
    if a.checkpoint is None:
        raise Exception("Checkpoint required for test mode")
    options = {"which_direction", "ngf", "ndf"}
    with open(os.path.join(a.checkpoint, "options.json")) as f:
        for key, val in json.loads(f.read()).items():
            if key in options:
                print("loaded", key, "=", val)
                setattr(a, key, val)

    return a


def write_json(a):
    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))


def get_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def load_examples(a):
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
    decode = tf.image.decode_png
    if len(input_paths) == 0:
        raise Exception("input_dir contains no png files")
    input_paths = sorted(input_paths)

    paths, contents = tf.WholeFileReader().read(tf.train.string_input_producer(input_paths, shuffle=True))
    raw_input = tf.image.convert_image_dtype(decode(contents), dtype=tf.float32)
    raw_input.set_shape([None, None, 3])

    # break apart image pair and transfer to range [-1, 1]
    width = tf.shape(raw_input)[1]
    if a.input_channels == 3:
        target_images = expand(raw_input[:, :width // 2, :])
        input_images = expand(raw_input[:, width // 2:, :])
    else:
        target_images = expand(raw_input[:, :width // 3, :])
        input_images = expand(tf.concat([raw_input[:, width // 3:2 * width // 3, :],
                                         raw_input[:, 2 * width // 3:, 0:1]], 2))
        input_images.set_shape([None, None, 4])

    # area produces a nice downscaling, but does nearest neighbor for upscaling
    # assume we're going to be doing downscaling here
    input_images = tf.image.resize_images(input_images, [a.CROP_SIZE, a.CROP_SIZE], method=tf.image.ResizeMethod.AREA)
    target_images = tf.image.resize_images(target_images, [a.CROP_SIZE, a.CROP_SIZE], method=tf.image.ResizeMethod.AREA)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images],
                                                              batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    print("examples count = %d" % len(input_paths))

    return collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )
