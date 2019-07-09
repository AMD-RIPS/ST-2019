import cv2  
import os
import sys
import random

input_path = sys.argv[1]
output_path = sys.argv[2]

glitch_dist_ratio = float(sys.argv[3])
glitch_length_ratio = float(sys.argv[4])
max_num = int(sys.argv[5])

count = 0
for image_path in os.listdir(input_path):
	if count >= max_num:
		break
	
	print("Processing: " + image_path)
	img = cv2.imread(os.path.join(input_path, image_path))
	ending = str(count) + "_vertical_pattern.png"
	this_output_path = os.path.join(output_path, ending)
	count += 1

	if not image_path.endswith('.png') and not image_path.endswith('.jpg') and not image_path.endswith('.jpeg'):
		continue

	(height, width, channel) = img.shape
	pattern_dist = int(width * glitch_dist_ratio)
	pattern_length = int(height * glitch_length_ratio)
	segment_length = pattern_length // 8

	print("pattern_length: " + str(pattern_length))
	print("pattern_dist: " + str(pattern_dist))
	print("segment_length: " + str(segment_length))
	
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

		#img[:, p:(p+4*pattern_dist), :] = add_vertical_patterns(img[:, p:(p+4*pattern_dist), :], pattern_dist, pattern_length)

	cv2.imwrite(this_output_path, img)
