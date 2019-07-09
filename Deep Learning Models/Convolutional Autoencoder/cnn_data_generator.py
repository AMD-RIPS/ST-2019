import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import random
from functools import partial
import numpy as np
import sys

def add_vertical_pattern(img):
	(height, width, channel) = img.shape
	pattern_dist = int(width * 0.01)
	pattern_length = int(height * 0.04)
	segment_length = pattern_length // 8

	row_count = 0
	start_row_index = 0
	horizontal_shift = random.randint(0, pattern_dist)
	for y in range(horizontal_shift, width, pattern_dist):
		if row_count % 4 == 0:
			vertical_shift = random.randint(0, pattern_length)
		for x in range(0, height, pattern_length):
			img[(vertical_shift + x) % height, y, :] = 255
			img[(vertical_shift + x + segment_length) % height, y, :] = 255
			img[((vertical_shift + x + 2 * segment_length)% height):((vertical_shift + x + 4 * segment_length)% height), y, :] = 255
			img[(vertical_shift + x + 5 * segment_length)% height, y, :] = 255
			img[(vertical_shift + x + 6 * segment_length)% height, y, :] = 255
		row_count += 1
	return img

def add_slant_pattern(img):
	(height, width, channel) = img.shape
	pattern_length = int(width * 0.008)
	
	row_count = 0
	for y in range(0, width, pattern_length):
		for x in range(row_count % 3, height, 3):
			try:
				img[x, y:(y+pattern_length), :] = 255
			except:
				pass
		row_count += 1

	return img

def add_checkerboard_pattern(is_pattern_intermittent, is_random, img):
	(height, width, channel) = img.shape
	pattern_width = int(width * 0.03)
	pattern_height = pattern_width
	
	row_count = 0
	for y in range(0, width, pattern_width):
		for x in range((row_count % 2) * pattern_height, height, 2 * pattern_height):
			if is_random and random.uniform(0, 1) < 0.45:
				continue
			if is_pattern_intermittent == 0:
				img[x:(x+pattern_height), y:(y+pattern_width), :] = 255
			else:
				img[x:(x+pattern_height):2, y:(y+pattern_width), :] = 255
		row_count += 1

	return img


def add_blurring(blur_type, img):
	p1 = 0
	p2 = 0
	p3 = 0

	if blur_type == "avg":
		p1 = random.randint(4, 8)
		p2 = random.randint(4, 8)
	elif blur_type == "bilateral":
		p1 = random.randint(5, 10)
		p2 = random.randint(80, 140)
		p3 = random.randint(80, 140)

	if blur_type == "avg":
		avging = cv2.blur(img,(p1, p2)) 
		return avging
	elif blur_type == "bilateral":
		bilFilter = cv2.bilateralFilter(img, p1, p2, p3) 
		return bilFilter

def add_lines(line_type, img):
	line_width = 3
	glitch_freq = 0.18

	if line_type == "horizontal":
		img = img.transpose(1,0,2)


	(height, width, channel) = img.shape

	for y in range(0, width - line_width, line_width):
		if random.uniform(0, 1) < glitch_freq:
			img[:, y:(y + line_width), 0] = random.randint(0, 255)
			img[:, y:(y + line_width), 1] = random.randint(0, 255)
			img[:, y:(y + line_width), 2] = random.randint(0, 255)

	if line_type == "horizontal":
		img = img.transpose(1,0,2)

	return img

def identity(img):
	return img


def read_images(imgs_path, target_height, target_width, num_limit):
	# imgs_path = "imgs/normal_imgs/dirt_rally"
	imgs_list = []

	# TODO: Add data augmentation functionality
	for image_path in os.listdir(imgs_path):
		if not image_path.endswith('.png') and not image_path.endswith('.jpg') and not image_path.endswith('.jpeg'):
			continue

		img = cv2.imread(os.path.join(imgs_path, image_path))
		img = cv2.resize(img, (target_width, target_height), interpolation = cv2.INTER_LINEAR)
		if len(imgs_list) < num_limit:
			imgs_list.append(img)
		else:
			break

	random.shuffle(imgs_list)
	return imgs_list

def transform_list(imgs_list):
	num_pics = len(imgs_list)

	checker_intermit_random = partial(add_checkerboard_pattern, 1, 1)
	checker_non_intermit_random = partial(add_checkerboard_pattern, 1, 0)
	checker_intermit_non_random = partial(add_checkerboard_pattern, 0, 1)
	checker_non_intermit_non_random = partial(add_checkerboard_pattern, 0, 0)

	avg_blurring = partial(add_blurring, "avg")
	bilateral_blurring = partial(add_blurring, "bilateral")

	vertical_line = partial(add_lines, "vertical")
	horizontal_line = partial(add_lines, "horizontal")

	# operation_list = [identity, add_vertical_pattern, add_slant_pattern, checker_intermit_random, \
	# checker_non_intermit_random, checker_intermit_non_random, checker_non_intermit_non_random, \
	# avg_blurring, bilateral_blurring, vertical_line, horizontal_line]

	# weight_list = [7, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2]
	operation_list = [identity]
	weight_list = [1]
	index_list = []

	total_index = 0
	total_weight = float(sum(weight_list))

	for i in range(len(weight_list)):
		total_index += int(weight_list[i] / total_weight * num_pics)
		index_list.append(min(total_index, num_pics - 1))

	X = []
	Y = []

	next_idx = 0
	label = 0
	for idx_max in index_list:
		for idx in range(next_idx, idx_max):
			X.append(operation_list[label](imgs_list[idx]))
			Y.append(label)
			# print(label)
		label += 1
		next_idx = idx_max

	X = np.asarray(X)
	Y = np.asarray(Y)
	return X, Y


def create_data(imgs_path, percent_training, target_height, target_width, num_limit):
	imgs_list = read_images(imgs_path, target_height, target_width, num_limit)
	num_pics = len(imgs_list)
	train_idx_max = int(num_pics * percent_training)

	X_train, Y_train = transform_list(imgs_list[:train_idx_max - 1])
	X_test, Y_test = transform_list(imgs_list[train_idx_max: ])
	return X_train, X_test, Y_train, Y_test


# imgs_path = sys.argv[1]
# percent_training = float(sys.argv[2])

# num_limit = 20
# target_width = 224
# target_height = 224

# X_train, X_test, Y_train, Y_test = create_data(imgs_path, percent_training, target_height, target_width, num_limit)
# print(X_train.shape)
# plt.imshow(X_train[-1,:,:,:])
# plt.show()

