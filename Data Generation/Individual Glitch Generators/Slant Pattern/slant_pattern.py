import cv2  
import os
import sys
import random

input_path = sys.argv[1]
output_path = sys.argv[2]

#glitch_dist_ratio = float(sys.argv[3])
glitch_length_ratio = float(sys.argv[3])
max_num =int(sys.argv[4])


count = 0
for image_path in os.listdir(input_path):
	if count >= max_num:
		break

	print("Processing: " + image_path)
	img = cv2.imread(os.path.join(input_path, image_path))
	ending = str(count) + "_slant_pattern.png"
	this_output_path = os.path.join(output_path, ending)
	count += 1


	if not image_path.endswith('.png') and not image_path.endswith('.jpg') and not image_path.endswith('.jpeg'):
		continue

	(height, width, channel) = img.shape
	pattern_length = int(width * glitch_length_ratio)

	print("pattern_length: " + str(pattern_length))
	
	horizontal_shift = 0
	row_count = 0
	for y in range(0, width, pattern_length):
		for x in range(row_count % 3, height, 3):
			try:
				img[x, y:(y+pattern_length), :] = 255
			except:
				pass
		row_count += 1

		#img[:, p:(p+4*pattern_dist), :] = add_vertical_patterns(img[:, p:(p+4*pattern_dist), :], pattern_dist, pattern_length)
	cv2.imwrite(this_output_path, img)

