# CoupleGenerator
### Update on 20200108

I'm surprised so many people are interested in this toy project. 

I saw some friends on Weibo who said that the training results were not good.

Here are the results on the training set after 8800 steps.
![image](https://github.com/irfanICMLL/CoupleGenerator/blob/master/Screenshot%20from%202020-01-08%2009-35-46.png)

The model can fit the training images in a short time. 

This ckpt maybe useful, but I am not sure.
https://cloudstor.aarnet.edu.au/plus/s/YHDWgez1g3RFc6o


### Environments
I guess the environment should be tensorflow==1.1...

You also need to download the pretrain weight for vgg through following links:


https://github.com/machrisaa/tensorflow-vgg

### Quick start

Download the training iamges and unzip it: https://cloudstor.aarnet.edu.au/plus/s/VWZJaWfbla3kFch

Run bash autotest.sh

### Original version
Generate your lover with your photo

This project is an old project from two years ago. 

We collect pictures of marriage couples on the Internet, and pre-process the images.

The model was trained by using the code of pix2pix.

An example of training images:

![image](https://github.com/irfanICMLL/CoupleGenerator/blob/master/datasets/marriage_crop/120/1.jpg)

You can make photos of yourself with the person you loved as a training pair~

This repo is just for entertaments.

If you have any interests, please feel free to use the code and the dataset.

### PS
I am not very familiar with tensorflow...

I have abandoned tf and started using pytorch.

This project is totally a toy project, and I am working on semantic segmentation, instance segmentation on 2D images and videos for my PHD career. If you are interested in these topics, you can follow my github. I will release new code on video segmentation recently.

### Extension
I collect the marriage images on BaiDu image by using spyder. Here is an very old and simple code for collecting images:
https://blog.csdn.net/qq_27879381/article/details/65015280#comments

Just change the key word.

If you really want to train an effective model, you can collect more images.

### AD
CVPR2019： https://github.com/irfanICMLL/structure_knowledge_distillation
