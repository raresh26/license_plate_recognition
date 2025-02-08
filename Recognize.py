import itertools

import cv2
import numpy as np
import os
from collections import deque


def segment_and_recognize(plate):
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
#TODO uncomment this if mutiple plates are given as input to the method

# for plate in plate_images:
	recognized_plates = []
	plate = preprocess_image(plate)
	plate_characters = character_segmentation(plate)

	characters_possibility_list = []
	for character in plate_characters:
		# try different accuracy_ration values
		characters_possibility_list.append(recognize_character(character, 0.1))

	combinations = plate_number_variants(characters_possibility_list)
	#TODO uncomment this if there are multiple combinations of chars
	# (each character has more recognized letters) ~ see recognize_character() method for this


	# for combination in combinations:
	# 	output_plate = plate_validity_check(combination)
	# 	if output_plate != "":
	#    recognized_plates.append(output_plate)
	if len(combinations[0]) != 6:
		return ''
	return combinations[0]


def binarize_image(img):
	"""
	Binarize the image using adaptive thresholding
	and (if it isn't already) convert to gray scale.
	Input:
		1. img
		type: numpy array
	Outputs:
		1. binarized_img: binarized image
		type: numpy array
	"""
	if len(img.shape) == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	binarized_img = cv2.GaussianBlur(img, (3,3), 0)
	binarized_img = cv2.adaptiveThreshold(binarized_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 7)
	return binarized_img

def preprocess_image(plate_img):
	"""
	Goes through multiple steps to make the plate image ready for the segmentation step
	Input:
		1. plate_img: cropped plate
		type: numpy array
	Outputs:
		1. preprocessed_img: preprocessed image resized to (235x55)
		type: numpy array
	"""

	# first resize image to standard size to make processing faster and easier
	# we choose (235x55) because, 235 is the mean width and 55 is the mean height of all the cropped images.
	preprocessed_img = cv2.resize(plate_img, (235, 55))

	# then binarize the image using adaptive thresholding
	preprocessed_img = binarize_image(preprocessed_img)

	# TODO (add this if necessary) finally, denoise the image
	#element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	#preprocessed_img = cv2.morphologyEx(preprocessed_img, cv2.MORPH_OPEN, element)

	return preprocessed_img

def character_segmentation(plate_img):
	"""
	Uses flood fill algorithm to check for potential characters:
	We start at x = 0 and y = height/2, then we go from left to right.
	Then, if we hit a white pixel, we use flood fill to mask out the entire character.
	Finally, check if the aspect ratio of the bounding box is plausible.
	Input:
		1. plate_img: preprocessed plate image
		type numpy array
	Outputs:
		1. characters: list of cropped character images
		type: list of numpy array
	"""

	characters = []
	char_mask = np.zeros_like(plate_img, dtype=bool)

	# scan across the middle line
	y = plate_img.shape[0]//2
	for x in range(plate_img.shape[1]):
		# if not a white pixel, skip it
		if plate_img[y, x] != 255:
			continue

		# if current pixel is part of a previous character's mask, skip it
		if char_mask[y, x]:
			continue

		char_mask = flood_fill(plate_img, (x, y))

		rx, ry, w, h = cv2.boundingRect(char_mask.astype('uint8'))
		aspect_ratio = w / h if w != 0 and h != 0 else -1

		# if aspect ratio is not valid, skip it
		#TODO adjust the lower bound to get better accuracy
		if not (0.45 < aspect_ratio < 0.85):
			continue

		# all checks have passed
		cropped_character = plate_img[ry:ry+h, rx:rx+w]
		characters.append(cropped_character)

	return characters


def flood_fill(img, start):
	"""
	Performs flood fill from given start location
	Input:
		1. img: img to perform flood fill on
		type: numpy array
		2. start: starting point of flood fill
		type: tuple containing (x, y)
	Outputs:
		1. mask: binary mask with filled area
		type: numpy array
	"""
	rows, cols = img.shape
	x, y = start
	target_color = 255
	mask = np.zeros_like(img, dtype=bool)

	directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

	# queue of tuples (y, x) to perform bfs
	queue = deque([(x, y)])

	while queue:
		cx, cy = queue.popleft()

		# skip if out of bounds, already marked or not white pixel
		if not (0 <= cx < cols and 0 <= cy < rows) or mask[cy, cx] or img[cy, cx] != target_color:
			continue

		# mark the current pixel
		mask[cy, cx] = True

		# add neighbors to the queue
		for dx, dy in directions:
			queue.append((cx + dx, cy + dy))

	return mask


def difference_score(input_char, reference_char):
	"""
	Perform a bitwise XOR between a reference character and an input character
	Parameters:
		1. input_char: input character
		2. reference_char: reference character
	Output:
		1. difference_score: difference score
	"""
	result = np.bitwise_xor(input_char, reference_char)
	return np.sum(result)


def remove_trailing_black_columns(image):
	"""
    Removes trailing black columns from an image.
    Parameters:
        1 image: Input image (grayscale or color doesn't matter).
    Returns:
        1: Image without trailing black columns.
    """
	if np.size(image.shape) == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# flipped because we want trailing black columns
	flipped_image = np.fliplr(image)

	# this is an array with truth values for each column telling whether its all black or not
	# true = column is not all black, false = column is all black
	is_column_black = np.any(flipped_image != 0, axis=0)

	# furthest column from idx = 0 that is not black anymore
	non_black_col = np.argmax(is_column_black)

	# ff no non-black column is found, return the original image
	if non_black_col == image.shape[1]:
		return image

	# return the image without trailing black columns
	return image[:, :image.shape[1] - non_black_col]

def createCharLists(pathLetters = "dataset/SameSizeLetters", pathNumbers = "dataset/SameSizeNumbers"):
	"""
	Create 2 lists of pair, each containing either (letter image, letter) or (digit image, digit)
	Parameters:
		1. pathLetters: path to letters images
		2. pathNumbers: path to numbers images
	Ouput:
		1. letters: list of (letter image, letter)
		2. numbers: list of (digit image, digit)
	"""
	letter = ["B", "D", "F", "G", "H", "J", "K", "L", "M", "N", "P", "R", "S", "T", "V", "X", "Z"]
	number = ["0", "1", "2", "3", "4", "5", "5", "6", "7", "8", "9"]

	# save letters
	if not os.path.exists(pathLetters) or not os.path.isdir(pathLetters):
		print("Invalid directory path")
		return []

	chars = []
	index = 0
	for file_name in sorted(os.listdir(pathLetters)):
		if file_name.endswith('.bmp'):
			file_path = os.path.join(pathLetters, file_name)
			character = cv2.imread(file_path)
			character = remove_trailing_black_columns(character)
			character = cv2.resize(character, (32, 32))
			chars.append((character, letter[index]))
			index += 1

	# save numbers
	if not os.path.exists(pathNumbers) or not os.path.isdir(pathNumbers):
		print("Invalid directory path")
		return []

	index = 0
	for file_name in sorted(os.listdir(pathNumbers)):
		if file_name.endswith('.bmp'):
			file_path = os.path.join(pathNumbers, file_name)
			digit = cv2.imread(file_path)
			digit = remove_trailing_black_columns(digit)
			digit = cv2.resize(digit, (32, 32))
			chars.append((digit, number[index]))
			index += 1

	return chars


def recognize_character(input_character, accuracy_rate):
	"""
	Recognize a character by comparing the XOR result of it and each character from the given set
	Parameters:
		1. input_character: input character
		2. accuracy_rate: accuracy rate (how close should the XORs be such that the character is ambigous)
	Output:
		1. output_letter: letter/letters which the character was associated with
	"""

	chars = createCharLists()

	xor_score = {}
	output_letter = []
	input_character = cv2.resize(input_character, (32, 32))

	for pair in chars:
		char_image = remove_trailing_black_columns(pair[0])
		char_image = cv2.resize(char_image, (32, 32))
		xor_score[pair[1]] = difference_score(input_character, char_image)

	first_letter = min(xor_score, key=xor_score.get)
	#TODO: uncomment this to get a list of most possible 3 characters instead of the first character only
	# after testing both option we concluded that, at least for the given video, the multiple character pick
	# would only falsely influence the majority voting process

	# first_score = xor_score.pop(first_letter)
	# if first_letter == "51":
	# 	first_letter = "5"
	#
	# second_letter = min(xor_score, key=xor_score.get)
	# second_score = xor_score.pop(second_letter)
	# if second_letter == "51":
	# 	second_letter = "5"
	#
	# if abs(first_score / second_score - 1) < accuracy_rate and first_letter != second_letter:
	# 	output_letter.append(first_letter)
	# 	output_letter.append(second_letter)
	#
	# 	third_letter = min(xor_score, key=xor_score.get)
	# 	third_score = xor_score.pop(third_letter)
	# 	if third_letter == "51":
	# 		third_letter = "5"
	#
	# 	if abs(second_score / third_score - 1) < accuracy_rate and third_letter != second_letter:
	# 		output_letter.append(third_letter)
	#
	# else:
	output_letter.append(first_letter)
	return output_letter

def plate_number_variants(output_letters):
	"""
	Create every possible combinations of plate numbers
	Parameters:
		1. output_letters: list of lists with every letter identified for each character
	Output:
		1. plate_combinations: every possible plate number
	"""
	plate_combinations = []
	plate_combinations = ["".join(combination) for combination in itertools.product(*output_letters)]

	return plate_combinations

#TODO: uncomment this in order to ouput the plate in the right format for recognition_evaluation testing
# For the actual process, we needed the plate without dashes in between and hence we commented this out

# def plate_validity_check(plate_combination):
# 	plate = ""
#
# 	if (len(plate_combination) != 6):
# 		return plate
#
# 	validity_array = []
# 	for letter in plate_combination:
# 		if letter.isdigit():
# 			validity_array.append(False)
# 		elif letter.isalpha():
# 			validity_array.append(True)
# 		else:
# 			return ""
#
# 	# 2 letter/digits - 2 digits/letters - 2 letters/digits
#
# 	if np.array_equal(validity_array, [True, True, False, False, True, True]) \
# 			or np.array_equal(validity_array, [False, False, True, True, False, False]) \
# 			or np.array_equal(validity_array, [False, False, False, False, True, True]) \
# 			or np.array_equal(validity_array, [True, True, False, False, False, False]) \
# 			or np.array_equal(validity_array, [True, True, True, True, False, False]) \
# 			or np.array_equal(validity_array, [False, False, True, True, True, True]):
# 		plate = plate_combination[0] + plate_combination[1] + "-" + plate_combination[2] + plate_combination[3] + "-" + \
# 				plate_combination[4] + plate_combination[5]
# 	# 2 letter/digits - 3 digits/letters - 1 letter/digit
# 	elif np.array_equal(validity_array, [False, False, True, True, True, False]) \
# 			or np.array_equal(validity_array, [True, True, False, False, False, True]):
# 		plate = plate_combination[0] + plate_combination[1] + "-" + plate_combination[2] + plate_combination[3] + \
# 				plate_combination[4] + "-" + plate_combination[5]
# 	# 1 letter/digit - 3 digits/letters - 2 letters/digits
# 	elif np.array_equal(validity_array, [False, True, True, True, False, False]) \
# 			or np.array_equal(validity_array, [True, False, False, False, True, True]):
# 		plate = plate_combination[0] + "-" + plate_combination[1] + plate_combination[2] + plate_combination[3] + "-" + \
# 				plate_combination[4] + plate_combination[5]
# 	# 3 letters/digits - 2 digits/letter - 1 letter/digit
# 	elif np.array_equal(validity_array, [False, False, False, True, True, False]) \
# 			or np.array_equal(validity_array, [True, True, True, False, False, True]):
# 		plate = plate_combination[0] + plate_combination[1] + plate_combination[2] + "-" + plate_combination[3] + \
# 				plate_combination[4] + "-" + plate_combination[5]
# 	else:
# 		return ''
# 	return plate

