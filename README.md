# CoupleGenerator

### Introduction

Generate your lover with your photo

This project is an old project from two years ago. I accidentally found it in my storage medium.

We collect pictures of married couples on the Internet and pre-process the images.

The model was trained by using the code of pix2pix.

An example of training images:

![image](https://github.com/irfanICMLL/CoupleGenerator/blob/master/datasets/marriage_crop/120/1.jpg)

You can make photos of yourself with the person you loved as a training pair~

This repo is just for entertainment.

If you have any interests (not for business), please feel free to use the code and the dataset.

### Fitting the training data

Here are the results of the training set after 8800 steps.
![image](https://github.com/irfanICMLL/CoupleGenerator/blob/master/Screenshot%20from%202020-01-08%2009-35-46.png)

The model can fit the training images in a short time.


### Environments
Tensorflow==1.1

You also need to download the pretrain weight for vgg through following links:

https://github.com/machrisaa/tensorflow-vgg


### Quick start

Download the training images and unzip it: https://cloudstor.aarnet.edu.au/plus/s/VWZJaWfbla3kFch

Run bash autotest.sh

Trained model (with limited data and training steps): https://cloudstor.aarnet.edu.au/plus/s/YHDWgez1g3RFc6o

We use all the data for training and testing, it is not scientific for research. The motivation of this project is a  surprise for my friend. We want to overfit her with the boy she loves secretly. Therefore we do not consider the
 generalization of the model.


### Extension
The marriage images were collected on BaiDu image by using a spider. Here is a very old and simple code for collecting images:
https://blog.csdn.net/qq_27879381/article/details/65015280#comments

Just change the keyword.

Thanks [GZHermit](https://github.com/GZHermit) for helping me to process the data. He detected the face from the collecting images and re-located the couple into training pairs. 

The images we provide is just for fun, not for business activities

If you really want to train an effective model for business, you have to collect more images in legal way.

### AD

This project is a toy project, and I am working on semantic segmentation, instance segmentation on 2D images and videos for my Ph.D. career. If you are interested in these topics, you can follow my github. I will release a new code on video segmentation recently.

CVPR2019： https://github.com/irfanICMLL/structure_knowledge_distillation
