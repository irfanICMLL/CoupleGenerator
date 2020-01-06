import os
import glob
import numpy as np
from PIL import Image

in_dirs = ['/data/ruidong/ruidong/auto_painter/nb_girls/datasets/test/old/dataset_test/',
           '/data/ruidong/ruidong/auto_painter/nb_girls/datasets/test/old/dataset_test_cover/']
out_dirs = ['/data/ruidong/ruidong/auto_painter/nb_girls/datasets/test/custom/custom_test/',
            '/data/ruidong/ruidong/auto_painter/nb_girls/datasets/test/custom/custom_test_cover/']
            
size = 512
sieve_size = 16
sieve_inteval = 2
sieve_num = size // sieve_size // sieve_inteval

if __name__ == "__main__":
    for num_dir, in_dir in enumerate(in_dirs):
        input_paths = glob.glob(in_dir + '/*_color.png')
        for path in input_paths:
            filePath, fileName = os.path.split(path)
            print('Rectifing', fileName)
            colorFileName, _ = os.path.splitext(fileName)
            originFileName = colorFileName[:4]
            colorPic = np.atleast_2d(Image.open(path).crop([0, 0, size, size]))
            originPic = np.atleast_2d(Image.open(os.path.join(filePath, originFileName + '.png')).crop([0, 0, size, size]))
            whitePic = 255 * np.ones([size, size, 3])
            chromatic = np.max(colorPic, axis=2) - np.min(colorPic, axis=2) > 20
            sieve = np.zeros([sieve_inteval, sieve_inteval])
            sieve[0][0] = 1
            sieve = np.tile(np.repeat(np.repeat(sieve, sieve_size, axis=0), sieve_size, axis=1), [sieve_num, sieve_num])
            chromatic = np.logical_and(chromatic, sieve)
            chromatic = np.tile(np.expand_dims(chromatic, 2), [1, 1, 3])
            if filePath[-1] == 't':
                originPic = np.append(originPic, originPic, axis=1)
                colorPic = colorPic * chromatic + whitePic * np.logical_not(chromatic)
            else:
                colorPic = colorPic * chromatic + originPic * np.logical_not(chromatic)            
            originOut = np.append(originPic, whitePic, axis=1).astype(np.uint8)
            colorOut = np.append(originPic, colorPic, axis=1).astype(np.uint8)
            Image.fromarray(originOut).save(os.path.join(out_dirs[num_dir], originFileName[2:] + ".png"))
            Image.fromarray(colorOut).save(os.path.join(out_dirs[num_dir], colorFileName[2:] + ".png"))
            