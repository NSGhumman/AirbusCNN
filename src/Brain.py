"""
				Module: Brain.py  |  Author: Narinder Singh Ghumman  |  Date: 14 April 2019
				Course: CSCI 8360 - Data Science Practicum @ UGA  | Spring '19

This module contains neural network architectures for training models.
"""

import sys
from argparse import ArgumentParser

import keras
import tensorflow as tf
from keras.models import Model, Sequential
from keras.applications import resnet50
from keras.layers import Dense, Conv2D, Conv2DTranspose, Activation
from keras.optimizers import SGD
from keras.utils import multi_gpu_model

from Helpers import *
from Losses import dice_coef, dice_coef_clipped

# Number of GPUs
GPUs = 4


def MLP():
	"""
	This is a multilayer fully connected perceptron model. This method is unimplemented as yet. CNNs have shown great success with image related tasks and we'll try them first.
	"""
	pass


def FCN(im_dims, lr=0.01):
	"""
	4/26/19: This implementation is not very well thought out as it stands right now. The basic idea is to have a representationally powerful fully convolutional neural network to learn the ship masks. It might be quite computationally expensive to train one though. Runs on 4 gpus.
	
	Update 01/05/19: It appreas that the number of arguments it too few to train anything and the dataset is not large enough to train something the size of successful CNNs like ResNet or VGG (or some other net succesful with the ImageNet challenge).
	
	A fully convolutional neural net for your modeling needs. The net is a succession of convolutional layers all the way to the output and retains the same width and height of volume throughout. @layers is the number of hidden convolutional layers. @filters is the number of filters per convolutional layer. @kernel_size is the kernel's (width, height).
	"""
	
	model = Sequential()
	# The first convolutional layer over the input image.
	model.add(Conv2D(256, (3, 3), padding='same', input_shape=im_dims))
	model.add(Activation('relu'))
	
	# Add more convolutional layers with same width and height.
	for x in xrange(3):
		model.add(Conv2D(128, (3, 3), padding='same'))
		model.add(Activation('relu'))

	# Add more layers, again retaining the width and height.
	for x in xrange(3):
		model.add(Conv2D(64, (3, 3), padding='same'))
		model.add(Activation('relu'))

	# Add a final layer with a single kernel to generate a 2D mask as output
	model.add(Conv2D(1, (3, 3), padding='same'))
	model.add(Activation('sigmoid'))
	
	optimizer = keras.optimizers.SGD(lr)

	# Use dice_coef for loss - it's easy to score well on binary crossentropy.
	parallel_model = multi_gpu_model(model, gpus=GPUs)
	parallel_model.compile(loss=dice_coef,
				  optimizer=optimizer,
              	  metrics=[dice_coef])

	# DEBUGGING
	print model.summary()
	return parallel_model


def ResFCN(lr=0.01):
	"""
		This implementation augments the end layers of the ResNet to make it a fully convolutional network for predicting image segmentattion masks.
	"""
	# Start with an implementation of ResNet50 but don't include the dense end layers.
	resnet = resnet50.ResNet50(include_top=False, weights='imagenet')
	# Freeze the network from training.
	for layer in resnet.layers: layer.trainable = False

	# Add transpose convolutional layers at the end for upsampling the output volume.
	output = Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=32)(resnet.layers[-1].output)
	
	# Convolution with 'unit' sized kernel to make the prediction volume 2 dimensional and sigmoid to bring the output to desired range.
	output = Conv2DTranspose(filters=1, kernel_size=(1, 1))(output)
	output = Activation('sigmoid')(output)

	# Make the new net.
	fcnresnet = Model(resnet.input, output)
	optimizer = keras.optimizers.SGD(lr)
	print fcnresnet.summary()
	
	# Parallelize the model.
	parallel_model = multi_gpu_model(fcnresnet, gpus=GPUs)
	parallel_model.compile(loss=dice_coef,
				  optimizer=optimizer,
              	  metrics=[dice_coef])

	return parallel_model


if __name__ == '__main__':
	# Parse command line arguments.
	parser = ArgumentParser("Brain")
	parser.add_argument('-e', '--epochs', help="Hyperparamter: Number of epochs to train the model for.", required=True)
	parser.add_argument('-b', '--batchsize', help="Hyperparameter: Batch size for training.", required=True)
	parser.add_argument('-s', '--samplesize', help="The size of the data smaple to train the network on.", required=True)
	parser.add_argument('-f', '--fraction', help="Fraction of the sample size to train on. Rest is used for testing.", required=True)
	parser.add_argument('-l', '--learningrate', help="Learning rate for the optimizer.", required=True)
	args = parser.parse_args()

	# Read data and make train and test sets.
	sample = getHashesOfInterest(int(args.samplesize))
	train, test = splitListFractionally(sample, frac=float(args.fraction))
	
	print 'Total sample size:', len(sample)
	print 'Training sample size: ', len(train)
	print 'Testing sample size: ', len(test)

	# Train and test data generators for keras batch based modeling and predicting functions.
	train_datagen = ShipGenerator(train, int(args.batchsize), im_processor=resnet50.preprocess_input)
	test_datagen = ShipGenerator(test, int(args.batchsize), im_processor=resnet50.preprocess_input)
	# Compile model
	im_dims = IM_DIMS + (3,)
	model = ResFCN(float(args.learningrate))

	# Fit and evaluate
	model.fit_generator(generator=train_datagen, epochs=int(args.epochs))
	
	result = model.evaluate_generator(test_datagen)
	print model.metrics_names
	print result
	
	# Save the model weights
	saveModelWeights(model, 'FCN')



