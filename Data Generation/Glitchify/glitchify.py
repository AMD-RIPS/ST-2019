import matplotlib.pyplot as plt
import cv2
import os
from functools import partial
import numpy as np
import numpy.random as npr
import sys
import argparse
from desktop_glitch.desktop_glitch_one import *
from desktop_glitch.desktop_glitch_two import create_desktop_glitch_two
import random
import imutils
import ou_glitch.ou_glitch as og
from triangulation.triangulation import triangulation


def get_random_color():
	b_int = npr.randint(0,256)
	g_int = npr.randint(0,256)
	r_int = npr.randint(0,256)
	return b_int, g_int, r_int

def add_vertical_pattern(img):
	color = np.random.randint(0,256,size = 3)
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

		if np.random.uniform() < 0.92:
			row_count += 1
			continue
		for x in range(0, height, pattern_length):
			if np.random.uniform() < 0.7:
				continue
			img[(vertical_shift + x) % height, y, :] = color
			img[(vertical_shift + x + segment_length) % height, y, :] = color
			img[((vertical_shift + x + 2 * segment_length)% height):((vertical_shift + x + 4 * segment_length)% height), y, :] = color
			img[(vertical_shift + x + 5 * segment_length)% height, y, :] = color
			img[(vertical_shift + x + 6 * segment_length)% height, y, :] = color
		row_count += 1
	return img

def blurring(img):
	height, width, _ = img.shape
	x0 = npr.randint(0, int(height / 3))
	y0 = npr.randint(0, int(width / 3))
	x1 = npr.randint(int(2 * height / 3), height)
	y1 = npr.randint(int(2 * width / 3), width)

	copy = np.copy(img[x0:x1,y0:y1,:])

	p1 = npr.randint(80, 90)
	p2 = npr.randint(80, 90)
	p3 = npr.randint(80, 90)
	bilFilter = cv2.bilateralFilter(copy, p1, p2, p3)
	img[x0:x1,y0:y1,:] = bilFilter
	return img

def create_discoloration(img):
	threshold = npr.randint(100, 140)
	new_intesity = npr.randint(200, 256)

	color = npr.randint(0, 6)
	if color == 0:
		img[img[:,:,2] > threshold] = [0,0,new_intesity]
	elif color == 1:
		img[img[:,:,1] > threshold] = [0,new_intesity,0]
	elif color == 2:
		img[img[:,:,0] > threshold] = [new_intesity,0,0]
	else:
		b_int = npr.randint(new_intesity,256)
		g_int = npr.randint(new_intesity,256)
		r_int = npr.randint(new_intesity,256)
		img[img[:,:,0] > threshold] = [b_int, g_int, r_int]

	return img

def add_random_patches(im, lo = 3, hi = 20):
	color = npr.randint(0, 6)
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	contours.sort(key = len)
	patch_number = np.random.randint(lo,hi+1)
	b_int, g_int, r_int = get_random_color()
	for i in range(patch_number):
		if color == 0:
			cv2.drawContours(im, contours,len(contours) - 1 - i , (0,0,250), -1)
		elif color == 1:
			cv2.drawContours(im, contours,len(contours) - 1 - i , (0,250,0), -1)
		elif color == 2:
			cv2.drawContours(im, contours,len(contours) - 1 - i , (250,0,0), -1)
		else:
			cv2.drawContours(im, contours,len(contours) - 1 - i , (b_int,g_int,r_int), -1)

	return im


def add_shapes(im, lo = 2, hi = 5):
	h, w, _ = im.shape
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# Find the darkest region of the image
	grid = (-1,-1)
	mean_shade = np.mean(im)
	x_step, y_step = int(w/6), int(h/4)
	for x in range(0, w, x_step):
		for y in range(0, h, y_step):
			new_shade = np.mean(im[x:x+x_step, y:y+y_step])
			if  new_shade <= mean_shade:
				mean_shade = new_shade
				grid = (x,y)

	# Add shapes
	minLoc = (np.random.randint(grid[0], min(grid[0]+x_step, w)), np.random.randint(grid[1], min(grid[1]+x_step, h)))
	num_shapes = np.random.randint(lo,hi+1)
	for i in range(num_shapes):
		stretch = np.random.randint(40, 100)
		diff1, diff2 = np.random.randint(-5,5), np.random.randint(-5,5)
		x1, y1 = minLoc[0] +  diff1* stretch  , minLoc[1] + diff2 * stretch
		x2, y2 = x1 + np.random.randint(1,12)/5 * diff1 * stretch  , y1 + np.random.randint(1,12)/5 * diff2* stretch
		pts = np.array((minLoc, (x1, y1), (x2, y2)), dtype=int)

		c1, c2, c3 = np.random.randint(0,50),np.random.randint(0,50),np.random.randint(0,50)
		cv2.fillConvexPoly(im, pts, color= (c1,c2,c3))

	return im


def add_triangles(im, lo = 1, hi = 3):
	h, w, _ = im.shape
	colors = np.array((
                   (250,206,135),
                   (153,255, 255),
                   (255, 203, 76)),dtype = int) #maybe expand this list of colors

	output = im.copy()
	overlay = im.copy()

	x_0, y_0 = np.random.randint(w), np.random.randint(h)
	x_1, y_1 = np.random.randint(w), np.random.randint(h)
	x_2, y_2 = np.random.randint(w), np.random.randint(h)
	pts = np.array(((x_0, y_0), (x_1, y_1), (x_2, y_2)), dtype=int)

	b_int, g_int, r_int = get_random_color()
	cv2.fillConvexPoly(overlay, pts, color= tuple([b_int, g_int, r_int]) )

	num_shapes = np.random.randint(lo, hi + 1)
	alpha = .95
	for i in range(num_shapes):
		x_1, y_1 = np.mean([x_1, x_0]) + np.random.randint(-60,60), np.mean([y_1,y_0])+ np.random.randint(-60,60)
		x_2, y_2 = np.mean([x_2, x_0]) + np.random.randint(-60,60), np.mean([y_2,y_0])+ np.random.randint(-60,60)
		
		pts = np.array(((x_0, y_0), (x_1, y_1), (x_2, y_2)), dtype=int) 
		# if not is_random:
		# 	cv2.fillConvexPoly(overlay, pts, color= tuple([int(x) for x in colors[np.random.randint(3)]]) )

		b_int, g_int, r_int = get_random_color()
		cv2.fillConvexPoly(overlay, pts, color= tuple([b_int, g_int, r_int]) )

	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

	return output



#this function returns a color blend of the overlay and the original image. angle = 0 means the overlay
#will fade down and angle = 180 will cause fade up:
##################################
def gradient(img, overlay, angle = 0):
	alpha = 1


	img = imutils.rotate_bound(img, angle)
	overlay = imutils.rotate_bound(overlay, angle)


	for x in range(1, img.shape[0],10 ):
		cv2.addWeighted(overlay[x:x+10,:,:], alpha, img[x:x+10,:,:] , 1 - alpha, 0, img[x:x+10,:,:])
		alpha *= .98


	img = imutils.rotate_bound(img, -1 * angle)
	return img

#########
def color_blend(img, overlay1, overlay2, angle = 0):
	alpha = 1

	img = imutils.rotate_bound(img, angle)
	overlay1 = imutils.rotate_bound(overlay1, angle)
	overlay2 = imutils.rotate_bound(overlay2, angle)

	for x in range(1, overlay1.shape[0],10 ):
		cv2.addWeighted(overlay1[x:x+10,:,:], alpha, overlay2[x:x+10,:,:] , 1 - alpha, 0, img[x:x+10,:,:])
		alpha *= .95

	img = imutils.rotate_bound(img, -1 * angle)

	return img
#############################


def add_shaders(im, lo = 1, hi = 3):
	angles = np.array([0,90,180,270])

	h,w,_ = im.shape
	output = im.copy()
	overlay1 = im.copy()
	overlay2 = im.copy()

	#####big shaders in forms of n-gons
	num_shapes = np.random.randint(lo,hi+1)
	for i in range(num_shapes):
		x_0, y_0 = np.random.randint(w), np.random.randint(h)
		x_1, y_1 = np.random.randint(-300,w+300), np.random.randint(-300,h+300)
		x_2, y_2 = np.random.randint(-300,w+300), np.random.randint(-300,h+300)

		pts = np.array(((x_0, y_0), (x_1, y_1), (x_2, y_2)), dtype=int)

		extra_n = np.random.randint(4)

		for i in range(extra_n): #extra number of points to make an n_gon
			pts = np.append(pts, [[np.random.randint(-300,h+300), np.random.randint(-300,w+300)]], axis = 0)

		alpha = 1

		colors = np.empty([2, 3])
		colors[0,:] = npr.randint(0, 256, size = 3)
		colors[1,:] = colors[0,:] + npr.randint(0, 100, size = 3)
		colors[1,:] = np.clip(colors[1,:], 0, 255)

		
		cv2.fillConvexPoly(overlay1, pts, color= tuple([int(x) for x in colors[0]]) )
		cv2.fillConvexPoly(overlay2, pts, color= tuple([int(x) for x in colors[1]]) )


	############
	a1, a2 = random.choice(angles), random.choice(angles)

	return gradient(output, color_blend(im, overlay1, overlay2, a1), a2)


def write_files(original_img, img, is_margin_specified, filename):
	if is_margin_specified:
		original_img[x0:x1, y0:y1, :] = img
	else:
		original_img = img

	cv2.imwrite(filename, original_img)







if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Values
	parser.add_argument('-o', '--output', dest='output_foldername')
	parser.add_argument('-i', '--input', dest='input_foldername')
	parser.add_argument('-t', '--type', dest='glitch_type')
	parser.add_argument('-lo', dest='arg1')
	parser.add_argument('-hi', dest='arg2')

	parser.add_argument('-x0', dest = 'x0')
	parser.add_argument('-y0', dest = 'y0')
	parser.add_argument('-x1', dest = 'x1')
	parser.add_argument('-y1', dest = 'y1')
	parser.add_argument('-interval', dest = 'interval', default=10)


	options = parser.parse_args()
	global arg1, arg2, x0, y0, x1, y1

	is_bound_specified = False
	is_margin_specified = False
	interval = int(options.interval)

	if options.arg1 is not None and options.arg2 is not None:
		is_bound_specified = True
		arg1 = int(options.arg1)
		arg2 = int(options.arg2)


	if options.x0 is not None and options.y0 is not None and options.x1 is not None and options.y1 is not None:
		is_margin_specified = True
		x0 = int(options.x0)
		y0 = int(options.y0)
		x1 = int(options.x1)
		y1 = int(options.y1)

	count = 0

	if options.input_foldername is None or options.output_foldername is None:
		print("Please specify input and output folder name")


	for video_path in os.listdir(options.input_foldername):
		if not video_path.endswith('.mp4') and not video_path.endswith('.avi'):
			continue

		cap = cv2.VideoCapture(os.path.join(options.input_foldername, video_path))

		this_count = 0
		global prev_img
		while(True):
			ret, original_img = cap.read()
			if not ret:
				break


			if this_count % interval != 0:
				this_count += 1
				continue

			img = np.copy(original_img)

			if is_margin_specified:
				img = original_img[x0:x1, y0:y1, :]

			if this_count == 0:
				prev_img = img

			if options.glitch_type is None:
				output_name = str(count) + "_normal.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				cv2.imwrite(output_filename, img)
				count += 1

			if options.glitch_type == 'screen_tearing':
				if this_count == 0:
					this_count += 1
					continue
				
				height, width, channels = img.shape
				r = np.random.rand(1) * 0.8 + 0.1

				new_img = np.copy(img)
				if np.random.uniform() < 0.6:
					target_height = np.rint(r * height)
					target_height = target_height.astype(int)
					diff = (img[0:target_height[0], :, :] - prev_img[0:target_height[0], :, :]) / 255
					area = width * target_height[0]

					ssq = np.sum(diff**2) / area
					if ssq < 0.4:
						this_count += 1
						continue

					new_img[0:target_height[0], :, :] = prev_img[0:target_height[0], :, :]
				else:
					target_width = np.rint(r * width)
					target_width = target_width.astype(int)
					diff = (img[:, 0:target_width[0], :] - prev_img[:, 0:target_width[0], :]) / 255
					area = height * target_width[0]

					ssq = np.sum(diff**2) / area
					if ssq < 0.4:
						this_count += 1
						continue

					new_img[:, 0:target_width[0], :] = prev_img[:, 0:target_width[0], :]

				prev_img = img
				output_name = str(count) + "_screen_tearing.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, new_img, is_margin_specified, output_filename)
				count += 1

			if options.glitch_type == "desktop_glitch_one":
				# print(img.shape)
				new_img = create_desktop_glitch_one(img)

				output_name = str(count) + "_desktop_glitch_one.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				cv2.imwrite(output_filename, new_img)
				count += 1

			if options.glitch_type == "desktop_glitch_two":
				img = create_desktop_glitch_two(img)

				output_name = str(count) + "_desktop_glitch_two.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				cv2.imwrite(output_filename, img)
				count += 1

			if options.glitch_type == "discoloration":
				img = create_discoloration(img)

				output_name = str(count) + "_discoloration.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				# cv2.imwrite(output_filename, img)
				write_files(original_img, img, is_margin_specified, output_filename)
				count += 1

			if options.glitch_type == "random_patch":
				if is_bound_specified:
					img = add_random_patches(img, arg1, arg2)
				else:
					img = add_random_patches(img)

				output_name = str(count) + "_random_patch.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename)
				count += 1

			if options.glitch_type == 'shape':
				if is_bound_specified:
					img = add_shapes(img, arg1, arg2)
				else:
					img = add_shapes(img)

				output_name = str(count) + "_shape.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename)
				count += 1

			if options.glitch_type == 'triangle':
				if is_bound_specified:
					img = add_triangles(img, arg1, arg2)
				else:
					img = add_triangles(img)

				output_name = str(count) + "_triangle.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename)
				count += 1


			if options.glitch_type == 'shader':
				if is_bound_specified:
					img = add_shaders(img, arg1, arg2)
				else:
					img = add_shaders(img)

				output_name = str(count) + "_shader.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename)
				count += 1

			if options.glitch_type == 'dotted_line':
				if is_bound_specified:
					img = og.dotted_lines(img, arg1, arg2)
				else:
					img = og.dotted_lines(img)

				output_name = str(count) + "_dotted_line.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename)
				count += 1

			if options.glitch_type == 'parallel_line':
				if is_bound_specified:
					img = og.parallel_lines(img, arg1, arg2)
				else:
					img = og.parallel_lines(img)

				output_name = str(count) + "_parallel_line.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename)
				count += 1

			if options.glitch_type == 'square_patch':
				if is_bound_specified:
					img = og.square_patches(img, arg1, arg2)
				else:
					img = og.square_patches(img)

				output_name = str(count) + "_square_patch.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename)
				count += 1


			if options.glitch_type == 'texture_popin':
				img = blurring(img)

				output_name = str(count) + "_texture_popin.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename)
				count += 1

			if options.glitch_type == 'random_triangulation':
				img = triangulation(os.path.join(options.input_foldername, image_path), True)

				output_name = str(count) + "_random_triangulation.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename)
				count += 1

			if options.glitch_type == 'regular_triangulation':
				img = triangulation(os.path.join(options.input_foldername, image_path), False)

				output_name = str(count) + "_regular_triangulation.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename)
				count += 1

			if options.glitch_type == 'morse_code':
				img = add_vertical_pattern(img)

				output_name = str(count) + "_morse_code.jpg"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename)
				count += 1

			this_count += 1

		cap.release()








