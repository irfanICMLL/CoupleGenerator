from PIL import Image, ImageEnhance, ImageFilter
from pylab import *
from scipy.ndimage import filters
import glob, os

in_dir = '/data/ruidong/ruidong/auto_painter/nb_girls/train/test/'
out_dir = '/data/ruidong/ruidong/auto_painter/nb_girls/datasets/dataset_test/'
out_dir_cover = '/data/ruidong/ruidong/auto_painter/nb_girls/datasets/dataset_test_cover/'
out_dir_covered = '/data/ruidong/ruidong/auto_painter/nb_girls/datasets/dataset_test_covered/'
if not os.path.exists(out_dir): os.mkdir(out_dir)


class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)


def start(mat):
    w, h, c = mat.shape
    ws = int(w * random(1))
    hs = int(h * random(1))
    while (mat[ws, hs, 0] > 254) and (mat[ws, hs, 1] > 254) and (mat[ws, hs, 2] > 254):
        ws = int(w * random(1))
        hs = int(h * random(1))
    return ws, hs


def gen_color_line(mat, dir,max_length,max_dif):
    w, h, c = mat.shape
    ws = int(w * random(1))
    hs = int(h * random(1))
    if np.min(mat) > 254:
      size = min(max_length, w - ws, h - hs)
      return ws, hs, ws + size, hs + size, size * 2
    while (mat[ws, hs, 0] > 254) and (mat[ws, hs, 1] > 254) and (mat[ws, hs, 2] > 254):
        ws = int(w * random(1))
        hs = int(h * random(1))
    if dir == 1:
        wt = ws
        ht = hs
        while (wt < w - 1) and (abs(int(mat[wt, ht, 1]) + int(mat[wt, ht, 2]) + int(mat[wt, ht, 0]) - (
                        int(mat[wt + 1, ht, 1]) + int(mat[wt + 1, ht, 2]) + int(mat[wt + 1, ht, 0]))) < 80):
            wt = wt + 1
    if dir == 2:
        wt = ws
        ht = hs
        while (ht < h - 1) and (abs(int(mat[wt, ht, 1]) + int(mat[wt, ht, 2]) + int(mat[wt, ht, 0]) - (
                        int(mat[wt, ht + 1, 1]) + int(mat[wt, ht + 1, 2]) + int(mat[wt, ht + 1, 0]))) < 3):
            ht = ht + 1
    if dir == 3:
        wt = ws
        ht = hs
        length = 0
        while (length < max_length) and (wt < w-1) and (ht < h-1) and (
            abs(int(mat[wt, ht, 1]) + int(mat[wt, ht, 2]) + int(mat[wt, ht, 0]) - (
                            int(mat[wt + 1, ht + 1, 1]) + int(mat[wt + 1, ht + 1, 2]) + int(
                        mat[wt + 1, ht + 1, 0]))) < max_dif):
            ht += 1
            wt += 1
            length = abs(wt - ws) + abs(ht - hs)
    return ws, hs, wt, ht, length


def main():
    
    count = 0
    #parameter
    wsize = 1024  # double the resolution
    Gamma = 0.97
    Phi = 200
    Epsilon = 0.1
    k = 2
    Sigma = 1.5
    max_length=20
    min_length=10
    max_dif=30
    n_point=50
    dir = 3

    input_paths = glob.glob(in_dir+ '/*.jpg')	
    input_paths+=(glob.glob(in_dir+ '/*.jpeg'))
    input_paths+=(glob.glob(in_dir+ '/*.png'))
    #input_paths = [in_dir+'/da (1007).png']
    
    for files1 in input_paths:
        filepath, filename = os.path.split(files1)
        print(filename)

        im = Image.open(files1)
        w, h = im.size
        w = w // 2
        color_pi = im.crop((0, 0, w, h))
        gray_pic = im.crop((w, 0, w * 2, h))
        color_pic = np.atleast_2d(color_pi)
           
        if color_pic.ndim == 3 and color_pic.shape[-1] == 3:
            image = color_pi.filter(MyGaussianBlur(radius=5))
            mat = np.atleast_2d(image)
            mix_pic = np.atleast_2d(gray_pic)
            mix_pic.flags.writeable = True
            mix_pic_covered = np.atleast_2d(gray_pic)
            mix_pic_covered.flags.writeable = True
            real = np.atleast_2d(im)
            white_pic = np.uint8(np.multiply(np.ones((w, h, 3)), 255))

            for i in range(n_point):
                length = 0
                while length < min_length:
                    ws, hs, wt, ht, length = gen_color_line(mat, dir,max_length,max_dif)
                mix_pic[ws:wt, hs:ht, :] = np.min(np.array([mix_pic[ws:wt, hs:ht, :], mat[ws:wt, hs:ht, :]]), axis = 0)
                mix_pic_covered[ws:wt, hs:ht, :] = mat[ws:wt, hs:ht, :]
                white_pic[ws:wt, hs:ht, :] = mat[ws:wt, hs:ht, :]

            Image.fromarray(np.append(color_pic, mix_pic, axis=1)).save(os.path.join(out_dir_cover, 'c' + filename))
            Image.fromarray(np.append(real, white_pic, axis=1)).save(os.path.join(out_dir, 's' + filename))
            Image.fromarray(np.append(color_pic, mix_pic_covered, axis=1)).save(os.path.join(out_dir_covered, 'e' + filename))
            count += 1
            print('done!' + str(count))
            

if __name__ == '__main__':
    main()
