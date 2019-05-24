# bestfitter-airbus-challenge-sp19

## Introduction
This repository contains work done on the [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection) as part of the Data Science Practicum class (CSCI-8360) at the University of Georgia. The competition is hosted by Kaggle and part of the task is addressed in this work. The task is originally one of semantic object segmentation but this work limits the task to segmentation at pixel level. Objects with the same label are part of the same mass and unsegmented from each other. Additionally, the competition also asks that the segmentation be done in the form of aligned bounding boxes about the ships but the methods implemented here freely label pixels without following that restriction. 

<p align="center">
 <img align="center" src="https://i.ibb.co/yQ090jT/image.png">
</p>


## Hardware Requirements
The parameter space for the nets is huge and it is essential that the code is run on machines with GPUs. The default configuration is for *4x GPUs*. It is set as a constant in the *Brain.py* module and can be easily adjusted to your machine type. Typical speed is about 3-5 epochs/hour on Nvidia P100 GPUs.

## Software Requirements
The following python libraries are required
 1. tensorflow-gpu
 2. numpy
 3. python-opencv (cv2)
 4. matplotlib
 5. keras
 
## The model
The Brain module currently implements a few neural network models for image segmentation. The major one - ResNetFCN - is a fully convolutional neural network built on top of the ResNet50 CNN. The end layers of ResNet50 are replaced by transpose convolutional layers that upsample the input volume. The strides are set in a way that the input dimensions are recreated at the output layer. It is the default model that is run using directions below.
 
## How to run
Enter the source directory
```
cd src
```
and run the brain module as
```
python Brain.py
```
pass the ```--help``` argument to look at the requried parameters to pass. A typical run would look like this
```
python Brain.py --epochs=200 --learningrate=0.01 --batchsize=20 --samplesize=40000
```
After the run is complete, the weights are saved in ```models``` directory in a file with model name and a timestamp as the title.

## Ongoing work
The networks need some work done. Their learning is extremely limited at the moment. The FCN is limited by its small parameter space but the ResNetFCN - with pretrained weights on ImageNet - seems promising (refer this [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html)) and it appears that all it needs is some more careful thought and tweaking.

## LICENSE
The project is licensed under the MIT license.
