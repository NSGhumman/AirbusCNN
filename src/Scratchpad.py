"""
				Module: Scratchpad.py  |  Author: Narinder Singh Ghumman  |  Date: 01 May 2019
				Course: CSCI 8360 - Data Science Practicum @ UGA  | Spring '19

Scratchpad to expriment quickly with code.
"""

from keras.applications import resnet50
from keras.layers import Conv2DTranspose, Activation
from keras.models import Model

from Helpers import *
import matplotlib.pyplot as plt
import numpy as np

import cv2

if __name__ == '__main__':
	resnet = resnet50.ResNet50(include_top = False)
	for layer in resnet.layers: layer.trainable = False
	
	im = readImage("9edaf3c10")
	data = np.asarray([im])
	
	print 'data.shape: ', data.shape

	output = Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=32)(resnet.layers[-1].output)
	output = Activation('relu')(output)
	output = Conv2DTranspose(filters=1, kernel_size=(1, 1))(output)
	output = Activation('sigmoid')(output)

	newresnet = Model(resnet.input, output)

	result = newresnet.predict(data)
	print 'result.shape: ', result.shape

	print resnet.summary()
	print newresnet.summary()

	for layer in newresnet.layers:
		print layer.trainable
