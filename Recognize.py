import cv2
import numpy as np
import os


def segment_and_recognize(plate_images):
	"""
	In this file, you will define your own segment_and_recognize function.
	To do:
		1. Segment the plates character by character
		2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
		3. Recognize the character by comparing the distances
	Inputs:(One)
		1. plate_imgs: cropped plate images by Localization.plate_detection function
		type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
	Outputs:(One)
		1. recognized_plates: recognized plate characters
		type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
	Hints:
		You may need to define other functions.
	"""
	recognized_plates = []

	for image in plate_images:
		image = preprocess(image)
		char_images = crop(image)
		plate_string = recognise(char_images)

		recognized_plates.append(plate_string)

	# TODO: majority voting

	return recognized_plates


def preprocess(image):
	# Remove noise
	image = cv2.GaussianBlur(image, (5, 5), 0)

	# Resize to height of x pixels
	desired_height = 70
	aspect_ratio = image.shape[1] / image.shape[0]
	desired_width = int(aspect_ratio * desired_height)
	image = cv2.resize(image, (desired_width, desired_height))

	# Convert to grayscale
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Histogram equalisation
	image = cv2.equalizeHist(image)

	# TODO: extra 7% crop to get rid of noise?

	# Threshold to binarize the image
	mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)  # INV because letters are black

	# Apply erosion and dilation to remove noise
	kernel_size = 5
	erosion_kernel = np.ones((kernel_size, kernel_size), np.uint8)
	mask = cv2.erode(mask, erosion_kernel, iterations=1)
	dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)
	mask = cv2.dilate(mask, dilation_kernel, iterations=1)

	return mask


def crop(image):

	# Horizontal boundaries for each character
	char_index_ranges = []
	for j in range(image.shape[1]):
		pixel_sum = np.sum(image[:, j])

		if pixel_sum >= 4 and (len(char_index_ranges) == 0 or char_index_ranges[-1] == 2):
			char_index_ranges.append([])
			char_index_ranges[-1].append(j)
		elif pixel_sum < 4:
			if len(char_index_ranges) > 0 and len(char_index_ranges[-1]) == 1:
				char_index_ranges[-1].append(j - 1)

	# Crop characters
	chars = []
	for char_range in char_index_ranges:
		l, r = char_range
		chars.append(image[:, l: r + 1])

	# Vertical boundary for each character
	chars_cropped = []
	for char in chars:
		top_index, bottom_index = None, None
		for i in range(char.shape[0]):
			pixel_sum = np.sum(image[i, :])

			if pixel_sum >= 8 and top_index is None:
				top_index = i
			elif pixel_sum < 8 and top_index is not None:
				bottom_index = i - 1
				break

		chars_cropped.append(char[top_index: bottom_index + 1, :])

	return chars_cropped


def recognise(image):


