import cv2  
import os
import sys
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


def merge(orig, img, square_width):
	height, width, channel = img.shape
	lower_bound = int(height/3)
	new_height = np.random.randint(int(height / 4), int(height / 3))
	for i in range(0, width, 4 * square_width):
		# new_height = lower_bound
		img[0:new_height,i:i+4*square_width,:] = orig[0:new_height,i:i+4*square_width,:]
		if np.random.uniform() < 0.4:
			new_height = max(0, int(new_height - square_width * 0.8 * (np.random.uniform())))
	return img


def add_single_color(patch, color, is_right_col, density, intensity):
	height, width, channel = patch.shape
	col_width = int(width / 8)
	for j in range(is_right_col * col_width, width, col_width * 2):
		for j2 in range(col_width):
			for i in range(height):
				if np.random.uniform() < density:
					try:
						patch[i,j+j2,color] = min(patch[i,j+j2,color] + np.random.randint(100, 256 * intensity), 255 * intensity)
					except Exception as e: 
						pass

	return patch

def add_glitch(img, square_width):
	# img = np.zeros([target_height, target_width, 3])
	target_height, target_width, channel = img.shape

	for i in range(0, target_height, 2 * square_width):
			#for j in range(target_width):
			img[i:i+square_width,:,:] = 0
			img[i+square_width:i+square_width*2,:,:] = 0


	for i in range(0, target_height, 2 * square_width):
		for i2 in range(square_width):
			for j in range(target_width):
				color = {0,1,2}
				selected_color = np.random.randint(0,3)
				color.remove(selected_color)
				sum = 0
				try:
					for rem_co in color:
						img[i+i2,j,rem_co] = np.random.randint(20,80)
						sum += img[i+i2,j,rem_co]

					img[i+i2,j,selected_color] = np.random.randint(min(384 -sum, 255), 256)
				except:
					pass


	high_intensity = 1
	low_intensity = 0.3

	low_val = [0.002, 0.014, 0.08, 0.4]
	high_val = [0.002, 0.014, 0.08, 0.7]

	try:
		for j in range(0, target_width, square_width * 4):
			for p in range(4):
				img[:, j + p * square_width: j + (p+1)* square_width, :] = add_single_color(img[:, j + p * square_width: j + (p+1)* square_width, :], 0, 0, low_val[p], low_intensity)
				img[:, j + p * square_width: j + (p+1)* square_width, :] = add_single_color(img[:, j + p * square_width: j + (p+1)* square_width, :], 0, 1, low_val[p], high_intensity)
	except Exception as e: 
		pass

	try:
		for j in range(0, target_width, square_width * 4):
			for p in range(4):
				img[:, j + p * square_width: j + (p+1)* square_width, :] = add_single_color(img[:, j + p * square_width: j + (p+1)* square_width, :], 1, 1, low_val[p], low_intensity)
				img[:, j + p * square_width: j + (p+1)* square_width, :] = add_single_color(img[:, j + p * square_width: j + (p+1)* square_width, :], 1, 0, low_val[p], high_intensity)
	except Exception as e: 
		pass
		
	try:
		for j in range(0, target_width, square_width * 4):
			for p in range(4):
				img[:, j + p * square_width: j + (p+1)* square_width, :] = add_single_color(img[:, j + p * square_width: j + (p+1)* square_width, :], 2, 0, low_val[p], low_intensity)
				img[:, j + p * square_width: j + (p+1)* square_width, :] = add_single_color(img[:, j + p * square_width: j + (p+1)* square_width, :], 2, 1, low_val[p], high_intensity)
	except Exception as e: 
		pass

	return img

def create_desktop_glitch_one(orig_img):
	height, width, _ = orig_img.shape
	x0 = npr.randint(0, int(height / 6))
	y0 = npr.randint(0, int(width / 6))
	x1 = npr.randint(int(5 * height / 6), height)
	y1 = npr.randint(int(5 * width / 6), width)

	copy = np.copy(orig_img[x0:x1,y0:y1,:])


	square_width =  np.random.randint(50, 80)
	img = np.empty_like(copy)
	height, width, channel = img.shape
	img = add_glitch(img, square_width) 

	subwidth = int(width / 3)
	subheight = int(height / 3)

	sub_orig_x = np.random.randint(int(subheight / 2), int(subheight * 3 /2))
	sub_orig_y = np.random.randint(0, subwidth)

	added_patch = add_glitch(img[sub_orig_x:sub_orig_x+subheight, sub_orig_y:sub_orig_y+subwidth, :], square_width)
	img[sub_orig_x:sub_orig_x+subheight, sub_orig_y:sub_orig_y+subwidth, :] = np.clip(added_patch,0,255)
	img = merge(copy, img, square_width)

	orig_img[x0:x1,y0:y1,:] = img

	return orig_img



