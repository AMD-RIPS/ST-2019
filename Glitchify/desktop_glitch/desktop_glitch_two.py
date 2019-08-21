import cv2  
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def change_intensity(img, color, sub_color):
	height, width, channel = img.shape
	for i in range(height):
		for j in range(width):
			img[i,j,:] = color

	factor = np.random.uniform(0.95, 1)
	for i in range(height):
		img[height - i - 1,:,:] = (img[height - i - 1,:,:] * factor).astype(int)
		factor *= np.random.uniform(0.95, 1)
	
	img = add_pattern(img, sub_color)
	return img

def add_pattern(img, color):
	height, width, channel = img.shape

	# color = np.random.randint(0,255,size = 3)
	num_layers = np.random.randint(1, 5)

	num_pixels = np.random.randint(int(width/3), width)
	for i in range(num_layers):
		# print(color)
		img[height - 1 - i, (width - num_pixels):width, 0] = color[0]
		img[height - 1 - i, (width - num_pixels):width, 1] = color[1]
		img[height - 1 - i, (width - num_pixels):width, 2] = color[2]

		num_pixels = int(num_pixels * np.random.uniform(0.6, 1))

	return img

def random_patch(img):
	height, width, channel = img.shape
	for i in range(height):
		for j in range(width):
			for k in range(3):
				img[i,j,k] = np.random.randint(0,256)

	return img

def form_combined_pattern(img):
	height, width, channel = img.shape
	sub_height = int(height/4)
	sub_width = int(width/4)


	num_sqaures = np.random.randint(3, 5)
	color = np.random.randint(0,255,size = 3)
	sub_color = np.random.randint(0,255,size = 3)
	img[sub_height*3:height, sub_width:2*sub_width, :] = change_intensity(img[sub_height*3:height, sub_width:2*sub_width, :], color, sub_color)
	img[sub_height*3:height, 3*sub_width:4*sub_width, :] = change_intensity(img[sub_height*3:height, 3*sub_width:4*sub_width, :], color, sub_color)

	color = np.random.randint(0,255,size = 3)
	sub_color = np.random.randint(0,255,size = 3)
	img[sub_height*3:height, 2*sub_width:3*sub_width, :] = change_intensity(img[sub_height*3:height, 2*sub_width:3*sub_width, :], color, sub_color)
	if num_sqaures > 3:
		img[sub_height*3:height, :sub_width, :] = change_intensity(img[sub_height*3:height, :sub_width, :], color, sub_color)


	for i in range(4):
		if i < 2 and np.random.uniform() < 0.15:
			continue
		img[sub_height*2:sub_height*3, i * sub_width: (i+1) * sub_width, :] = random_patch(img[sub_height*2:sub_height*3, i * sub_width: (i+1) * sub_width, :])


	color = np.random.randint(0,255,size = 3)
	sub_color = np.random.randint(0,255,size = 3)
	if np.random.uniform() < 0.4:
		img[sub_height:sub_height*2, sub_width:2*sub_width, :] = change_intensity(img[sub_height:sub_height*2, sub_width:2*sub_width, :], color, sub_color)
	if np.random.uniform() < 0.4:
		img[sub_height:sub_height*2, 3*sub_width:4*sub_width, :] = change_intensity(img[sub_height:sub_height*2, 3*sub_width:4*sub_width, :], color, sub_color)

	color = np.random.randint(0,255,size = 3)
	sub_color = np.random.randint(0,255,size = 3)
	if np.random.uniform() < 0.4:
		img[sub_height:sub_height*2, 2*sub_width:3*sub_width, :] = change_intensity(img[sub_height:sub_height*2, 2*sub_width:3*sub_width, :], color, sub_color)
	if np.random.uniform() < 0.4:
		img[sub_height:sub_height*2, :sub_width, :] = change_intensity(img[sub_height:sub_height*2, :sub_width, :], color, sub_color)

	return img



def create_desktop_glitch_two(img):
	height, width, channel = img.shape
	sub_height = np.random.randint(int(height / 40), int(height / 30))
	sub_width = sub_height

	count = 0
	for i in range(0, height,  sub_height):
		generate_pattern = False
		for j in range(0, width, sub_width * 4):
			if count % 4 == 0:
				if np.random.uniform() < 0.05:
					generate_pattern = True
				else:
					generate_pattern = False

			if generate_pattern:
				img[i:i+sub_height, j:j+sub_width, :] = form_combined_pattern(img[i:i+sub_height, j:j+sub_width, :])
				if np.random.uniform() < 0.1:
					generate_pattern = False
			count += 1

	return img




