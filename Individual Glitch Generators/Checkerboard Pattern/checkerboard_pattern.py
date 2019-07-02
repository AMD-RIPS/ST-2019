import cv2  
import os
import sys
import random

input_path = sys.argv[1]
output_path = sys.argv[2]

glitch_width_ratio = float(sys.argv[3])
glitch_height_ratio = float(sys.argv[4])
is_pattern_intermittent = int(sys.argv[5])
is_pattern_square = int(sys.argv[6])
is_random = int(sys.argv[7])
max_num = int(sys.argv[8])


count = 0
for image_path in os.listdir(input_path):
	if count >= max_num:
		break
	
	print("Processing: " + image_path)
	img = cv2.imread(os.path.join(input_path, image_path))
	ending = str(count) + "_checkerboard.png"
	this_output_path = os.path.join(output_path, ending)
	count += 1

	if not image_path.endswith('.png') and not image_path.endswith('.jpg') and not image_path.endswith('.jpeg'):
		continue

	(height, width, channel) = img.shape
	pattern_width = int(width * glitch_width_ratio)
	pattern_height = int(height * glitch_height_ratio)

	if is_pattern_square:
		pattern_height = pattern_width

	print("pattern_width: " + str(pattern_width))
	print("pattern_height: " + str(pattern_height))
	
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

		#img[:, p:(p+4*pattern_dist), :] = add_vertical_patterns(img[:, p:(p+4*pattern_dist), :], pattern_dist, pattern_length)

	cv2.imwrite(this_output_path, img)
