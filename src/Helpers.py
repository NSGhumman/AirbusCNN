"""
						Module: Helpers.py  |  Author: Narinder Singh Ghumman  |  Date: 14 April 2019
							Course: CSCI 8360 - Data Science Practicum @ UGA  | Spring '19

This module contians methods to make life easier.
"""

import os
import sys
import operator
import random
import datetime
from multiprocessing import Pool
from argparse import ArgumentParser

import cv2
import numpy as np

import matplotlib.pyplot as plt
from keras.utils import Sequence

class UnexpectedValueException(Exception): pass


# Path Constants.
DATA_PATH = "../data"
VISUALS_DIR = "../visuals"
MODELS_DIR = "../models"
GROUND_TRUTH_SEGEMENTS_FILE = os.path.join(DATA_PATH, "train_ship_segmentations_v2.csv")
TRAINING_SET_DIR = os.path.join(DATA_PATH, "train_v2")
TESTING_SET_DIR = os.path.join(DATA_PATH, "test_v2")

# Dataset image dimenions
IM_DIMS = (768, 768)


class ProgressBar:
	"""
		A handrolled implementation of a progress bar. The bar displays the progress as a ratio like this: (1/360).
	"""

	def __init__(self, max = 100, message = "Initiating ....."):
		"""
			Initialize the bar with the total number of units (scale).
		"""
		self.max = max
		self.current = 0
		print message + '\n'

	def update(self, add = 1):
		"""
			Record progress.
		"""
		self.current += add
		self._clear()
		self._display()

	def _display(self):
		"""
			Print the completion ratio on the screen.
		"""
		print "(" + str(self.current) + "/" + str(self.max) + ")"

	def _clear(self):
		"""
			Erase the old ratio from the console.
		"""
		sys.stdout.write("\033[F")
		sys.stdout.flush()


class ShipGenerator(Sequence):
	"""
	A generator object to batchwise generate data for training.
	"""
	def __init__(self, hashes, batch_size, im_processor=lambda x: x):
		self.hashes = hashes
		self.batch_size = batch_size
		self.im_processor = im_processor

	def __len__(self):
		"""
		Returns the no. of btaches.
		"""
		x = len(self.hashes) / self.batch_size
		return x

	def __getitem__(self, batch_no):
		"""
		Generates batch numbered @batch_no
		"""
		start = batch_no * self.batch_size
		end = start + self.batch_size
		bhashes = self.hashes[start: end]

		X, Y = readImages(bhashes, im_processor=self.im_processor), readRLEMasks(bhashes)
		return X, Y


def readRLEMasks(hashes, im_dims=IM_DIMS, file=GROUND_TRUTH_SEGEMENTS_FILE):
	"""
	Reads the run-length encoded ground truth mask(s) for image(s) (identified by their hashe(s)). @file is path to the csv file that contains RLE encoded masks for all hashes. The segments file is part of the challenge dataset. @im_dims specifies the dimensions of the images/masks - this information is needed for decoding.
	Returns a list of binary 2D numpy array(s) representing the mask(s).
	"""
	# Read the ground truth segments file and create a list to collect masks.
	csv = readKeyValuesCSV(file)
	masks = []
	
	for hash in hashes:
		# Initialize an empty mask and get rle encoded ship pixels.
		mask = np.zeros(reduce(operator.mul, im_dims))
		ships = csv[hash]
		# Mark bits in the mask for each ship.
		for ship in ships:
			ship = [int(x) for x in ship.split()]
			for start, length in splitConsecutive(ship, 2):
				mask[start: start + length] = 1
		masks.append(mask.reshape(im_dims).T)

	# Return a list only if it is non singular
	else: return np.array(masks)


def readImages(hashes, channel='RGB', im_processor=lambda x: x):
	"""
	Reads the training images into the main memory. @hashes must be an iterable collection of hashnames of images. Preserves I/O order.
	Returns a numpy array of the training instances.
	"""
	ims = []
	for hash in hashes:
		ims.append(readImage(hash, channel, im_processor=im_processor))

	return np.array(ims)


def readImage(hash, channel='RGB', scaling=True, im_processor=lambda x: x):
	"""
	Reads the image uniquely identified by the @hash. @channel specifies the color channel - must be one of ['R', 'G', 'B' 'RGB']. If @scaling is set, each pixel value is normalized to a [0, 1] floating-point scale.
	Returns a numpy array.
	"""
	# Check to see that @channel has a valid value.
	if channel not in ['RGB', 'R', 'G', 'B']:
		raise UnexpectedValueException("Unexpected value passed for channel: " + str(channel) + ". Must be one of ['R', 'G', 'B', 'RGB']")

	# Try reading the image from the training set.
	impath = os.path.join(TRAINING_SET_DIR, hash.strip() + ".jpg")
	im = cv2.imread(impath)
	
	# If unsuccessful, try the testing set.
	if im is None:
		impath = os.path.join(TESTING_SET_DIR, hash.strip() + ".jpg")
		im = cv2.imread(impath)
	
	if im is None: raise UnexpectedValueException("Hash: " + str(hash) + " could not be found in the dataset." )
	if scaling: im = im

	B, G, R = cv2.split(im)
	channels = {'R': R, 'G': G, 'B': B}

	return im_processor(channels.get(channel, im[..., ::-1]))


def makeTrainingSample(size, hashes_only=True):
	"""
	Randomly selects a @size many subset of unique images from the training dataset and makes a sample for training. If @hahes_only flag is set, the method does not read the images but only returns the hashes themselves.
	Returns an ordered pair of lists <images, masks>
	"""
	# Gather all hashes.
	filenames = os.listdir(TRAINING_SET_DIR)
	hashes = [os.path.splitext(filename)[0] for filename in filenames if not filename.startswith(".")]
	# Sample a random subset of given size
	sample = random.sample(hashes, size)
	
	# just return the hashes if flag is set
	if hashes_only == True: return sample
	
	# else read in the images and return them
	images = readImages(sample)
	masks = readRLEMasks(sample)
	# Make masks psuedo 3D for compatibility with keras. Reshape to (width, height, 1).
	masks = np.array([mask.reshape(mask.shape + (1, )) for mask in masks])
	return (images, masks)


def makeTestingSample(size):
	"""
	Rendomly selects a subsample of size @size from the testing dataset.
	Returns an iterable of hashes.
	"""
	filenames = os.listdir(TESTING_SET_DIR)
	hashes = [os.path.splitext(filename)[0] for filename in filenames if not filename.startswith(".")]
	sample = random.sample(hashes, size)
	return readImages(sample)


def splitIntoTrainTest(data, frac=0.1):
	"""
		Split the data into train/test sets. @frac is the fraction of data points that need to be taken to make the testing set. @data is tuple (X, Y).
		Returns ((X_train, Y_train), (X_test, Y_test))
	"""
	# Split X and Y and get the size of the dataset
	X, Y = data
	size = len(X)

	# Mark and split.
	marker = int(size * (1 - frac))
	return ((X[:marker], Y[:marker]), (X[marker:], Y[marker:]))


def readKeyValuesCSV(file, header=False):
	"""
	Reads a csv file with two columns as key-value pairs. @file specifies the path to file and @header specifies if the csv contains column headers(titles). Multiple key-value pairs with the same key are aggregated into a list.
	Returns a python dictionary of the form: {key: [values]}
	"""
	# Dictionary to collect results
	result = {}
	
	with open(file, 'r') as csv:
		for line in csv:
			# Ignore the header if there's one
			if header: header = False
			else:
				# Read the entries; Strip extension and whitespace; Add to results
				key, value = line.split(',')
				key, value = os.path.splitext(key.strip())[0], value.strip()
				if key not in result: result[key] = [value]
				else: result[key].append(value)

	return result


def splitConsecutive(collection, length):
	"""
	Split the elements of the list @collection into consecutive disjoint lists of length @length. If @length is greater than the no. of elements in the collection, the collection is returned as is.
	"""
	# Insufficient collection size for grouping
	if len(collection) < length: return collection

	# Iterate over the collection and collect groupings into results.
	groupings = []
	index = 0
	while index < len(collection):
		groupings.append(collection[index: index +length])
		index += length

	return groupings


def visualizeTrainingInstance(hash, channel='RGB', save=False, savedir=VISUALS_DIR):
	"""
	Displays training image - in all its color channels - and the ground-truth mask for the given hash. Using the @save flag, this method can also be used to save the figures.
	"""
	# Make room for two images on the plot.
	fig, axes = plt.subplots(nrows=1, ncols=2)
	(image, mask) = axes
	
	# Name the figure.
	fig.suptitle('Training instance: ' + str(hash))
	
	# Color channel option vs. matplotlib color-map.
	channel_map = { 'R': 'Reds', 'B': 'Blues', 'G': 'Greens', 'RGB': 'jet' }
	
	# Read and display the image.
	image.imshow(readImage(hash, channel), extent=[0, 7, 0, 7], cmap=channel_map[channel])
	image.set_title("Image (Channel: " + channel + ")")
	
	# Read and display mask.
	mask.imshow(readRLEMasks([hash])[0], cmap='Reds', extent=[0, 7, 0, 7])
	mask.set_title('Mask')
	
	if not save: plt.show()
	else:
		# Destination folders for different color maps
		savepath = os.path.join(savedir, "masks", hash + ".png")
		plt.savefig(savepath)
		plt.close()


def createTrainingSetVisuals():
	"""
	Using the visualizeTrainingInstance method, this method creates and saves visualizations for *all* training instances.
	"""
	# Get all training images (by hashnames).
	filenames = os.listdir(TRAINING_SET_DIR)
	hashes = [os.path.splitext(filename)[0] for filename in filenames if not filename.startswith(".")]
	
	# Create and save visualizations.
	for hash in hashes:
		visualizeTrainingInstance(hash, save=True)


def computeIoUs(preds, truths):
	"""
	Compute intersection over union for the predicted masks vs ground-truth masks. @preds and @truths must have the same length and both are iterables of numpy matrices of same dimensions.
	"""
	# List to collect IoU for each pair
	IoUs = []
	
	# Iterate over the collections and compute IoUs
	for predicted, truth in zip(preds, truths):
		intersection = predicted * truth
		union = predicted + truth
		
		# Re-adjust union back to [0, 1] scale and return the result.
		union[union == 2] = 1
		IoUs.append(float(sum(intersection.flat)) / (sum(union.flat) or 1))

	return IoUs


def mean(nums):
	"""
	Computes and returns mean of a collection.
	"""
	return sum(nums)/len(nums)


def saveModelWeights(model, name, timestamp=True):
	"""
	Save weights of a trained keras model. @name is the name of the model and the @timestamp flag indicates wether to add timestamp flag to the filename.
	"""
	filename = name + ('_' + str(datetime.datetime.now()) if timestamp else '')
	save_path = os.path.join(MODELS_DIR, filename)
	model.save_weights(save_path)


def splitListFractionally(x, frac=0.1):
	"""
	Split the list fractionally into two, returning a tuple with the larger one first.
	"""
	marker = int(len(x) * frac)
	return x[:marker], x[marker:]


def hasShip(hash):
	"""
	Checks wether or not the image (identified with the given @hash) contains a ship or not. This is done by reading the training mask for the image, so works only with training dataset hashes.
	"""
	mask = readRLEMasks([hash])[0]
	return True if np.sum(mask) else False


def divideList(x, k):
	"""
	Divide list @x into roughly @k parts.
	"""
	if k > len(x): return [[elem] for elem in x]
	part_len = int(len(x) / k)
	parts = []

	start = 0
	while start < len(x):
		parts.append(x[start: start + part_len])
		start += part_len

	return parts


def createTrainingSubsetFileWithShipsOnly(cpus):
	"""
	This method creates a training subset with just the hashes for those images that have at least one ship in their mask and then saves the hashes as a space separated string in a text file. This method creates 24 subprocesses to do the computation fast.
	"""
	# Get all training images (by hashnames).
	filenames = os.listdir(TRAINING_SET_DIR)
	hashes = [os.path.splitext(filename)[0] for filename in filenames if not filename.startswith(".")]
	
	hashes_breakdown = divideList(hashes, cpus)
	ppool = Pool(cpus)
	
	result =  flatten(ppool.map(extractHashesWithShipsOnly, hashes_breakdown))

	save_str = " ".join(result)
	save_file = os.path.join(DATA_PATH, "hashesofinterest.txt")
	with open(save_file, 'w') as outfile:
		outfile.write(save_str)


def flatten(lists):
	"""
		flattens a list of lists x
	"""
	result = []
	for x in lists:
		for elem in x: result.append(elem)

	return result


def extractHashesWithShipsOnly(hashes):
	"""
	From the given set of hashes, extract and return only those that have a ship in their mask.
	"""
	hashes_of_interest = []
	pbar = ProgressBar(max=len(hashes), message="Analyzing hashes for ships ...")
	for hash in hashes:
		if hasShip(hash): hashes_of_interest.append(hash)
		pbar.update()

	return hashes_of_interest


def getHashesOfInterest(size):
	"""
	Read the hashes of interest from the designated file.
	"""
	load_file = os.path.join(DATA_PATH, "hashesofinterest.txt")
	with open(load_file, 'r') as infile:
		hashes = infile.next().split()

	print 'Sampling: ' + str(size) + ' no. of hashes ' + 'from a total of: ' + str(len(hashes))
	return random.sample(hashes, size)


if __name__ == "__main__":
	"""
		This module is also runnable as a script. Look at the method being called below's description for more information.
	"""
	parser = ArgumentParser("Helpers")
	parser.add_argument('-c', '--cpus', help="The no. of CPUs available to run the operation on", required=True)
	args = parser.parse_args()
	
	createTrainingSubsetFileWithShipsOnly(int(args.cpus))


