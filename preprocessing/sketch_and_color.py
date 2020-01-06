import os
import glob
import numpy as np
from scipy.ndimage import filters
from PIL import Image, ImageEnhance

in_dir = '/data/ruidong/ruidong/auto_painter/nb_girls/train/test/'
#in_dir = '/data/irfan/auto_painter/nb_girls/all_sketch/'
out_dirs = ['/data/ruidong/ruidong/auto_painter/nb_girls/datasets/test/new/dataset_test/',
            '/data/ruidong/ruidong/auto_painter/nb_girls/datasets/test/new/dataset_test_cover/']
'''
in_dir = 'G:/ICMLL/AutoPainter/some outputs'
out_dirs = ['G:/ICMLL/AutoPainter/testpreprocessing/1',
            'G:/ICMLL/AutoPainter/testpreprocessing/2']
'''

size = 512
k = 2
sigma = 1.5
gamma = 0.97
epsilon = 0.1
phi = 198
sampling_fre = 0.0005
sampling_size = 256


def resize_pic(pic):
    if pic.size[0] > pic.size[1]:
        pic = pic.resize([size, size * pic.size[1] // pic.size[0]])
        pic = np.atleast_2d(pic)[:, :, :3]
        pad_up = np.multiply(np.ones([(size - pic.shape[0]) // 2, size, 3]), 255).astype(np.uint8)
        pad_down = np.multiply(np.ones([(size - pic.shape[0] - pad_up.shape[0]), size, 3]), 255).astype(np.uint8)
        return np.append(np.append(pad_up, pic, axis=0), pad_down, axis=0)
    else:
        pic = pic.resize([size * pic.size[0] // pic.size[1], size])
        pic = np.atleast_2d(pic)[:, :, :3]
        pad_left = np.multiply(np.ones([size, (size - pic.shape[1]) // 2, 3]), 255).astype(np.uint8)
        pad_right = np.multiply(np.ones([size, (size - pic.shape[1] - pad_left.shape[1]), 3]), 255).astype(np.uint8)
        return np.append(np.append(pad_left, pic, axis=1), pad_right, axis=1)


def generate_sketch(pic):
    pic = np.atleast_2d(ImageEnhance.Sharpness(Image.fromarray(pic).convert("L")).enhance(3.0))
    pic = filters.gaussian_filter(pic, sigma) - gamma * filters.gaussian_filter(pic, sigma * k)
    pic = np.multiply(np.tanh(np.multiply(pic, phi)) + 250, pic > epsilon).astype(np.uint8)
    if pic.ndim == 2:
        pic = np.tile(np.expand_dims(pic, 2), [1, 1, 3])
    sketched = pic < 127
    return pic, sketched


def draw_color_blocks(color, line):
    def bfs(x_start, y_start):
        def valid(vx, vy):
            if vx < 0 or vx >= size or vy < 0 or vy >= size or line[vx][vy][0] or colored[vx][vy]:
                return False
            return True
        directs = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        array = [[x_start, y_start]]
        head = 0
        tail = 0
        num = 1
        while head <= tail and num < sampling_size:
            for x_new, y_new in np.add(directs, array[head]).tolist():
                if valid(x_new, y_new):
                    array.append([x_new, y_new])
                    colored[x_new][y_new] = True
                    tail += 1
                    num += 1
            head += 1
        color_average = []
        for x, y in array:
            color_average.append(color[x][y])
        color_average = np.average(color_average, axis=0).astype(np.uint8)
        for x, y in array:
            color_blocks[x][y] = color_average
        return

    color_blocks = np.multiply(np.ones([size, size, 3]), 255).astype(np.uint8)
    luminance = np.average(color, axis=2)
    colored = np.logical_and(luminance > 5, luminance < 250)
    start = np.logical_and(colored, np.random.rand(size, size) < sampling_fre)
    print(np.sum(start), "blocks")
    colored = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            if start[i][j]:
                bfs(i, j)
    return color_blocks


if __name__ == "__main__":
    for out_dir in out_dirs:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    input_paths = glob.glob(in_dir + "/*.jpg") + glob.glob(in_dir + "/th*.png")
    for path in input_paths:
        filePath, fileName = os.path.split(path)
        print("Coloring", fileName, end=' with ')
        fileName, _ = os.path.splitext(fileName)
        origin = Image.open(path)
        origin = resize_pic(origin.crop([0, 0, origin.size[0] // 2, origin.size[1]]))
        sketch, dark = generate_sketch(origin)
        blocks = draw_color_blocks(origin, dark)
        concatenations = [np.append(np.append(origin, sketch, axis=1), blocks, axis=1),
                          np.append(origin, np.logical_not(dark) * blocks + dark * sketch, axis=1)]
        for i, concatenation in enumerate(concatenations):
            Image.fromarray(concatenation.astype(np.uint8)).save(os.path.join(out_dirs[i], os.path.basename(fileName) + ".png"))
