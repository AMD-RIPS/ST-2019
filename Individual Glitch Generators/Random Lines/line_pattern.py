import cv2  
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

input_path = sys.argv[1]
output_path = sys.argv[2]

line_width = int(sys.argv[3])
line_type = sys.argv[4]
glitch_freq = float(sys.argv[5])
max_num = int(sys.argv[6])


count = 0
for image_path in os.listdir(input_path):
	if count >= max_num:
		break
	
	if not image_path.endswith('.png') and not image_path.endswith('.jpg') and not image_path.endswith('.jpeg'):
		continue

	print("Processing: " + image_path)
	img = cv2.imread(os.path.join(input_path, image_path))
	ending = str(count) + "_vertical_line.png"

	# print(img.shape)

	if line_type == "horizontal":
		img = img.transpose(1,0,2)
		ending = str(count) + "_horizontal_line.png"


	(height, width, channel) = img.shape

	this_output_path = os.path.join(output_path, ending)
	count += 1

	for y in range(0, width - line_width, line_width):
		if random.uniform(0, 1) < glitch_freq:
			img[:, y:(y + line_width), 0] = random.randint(0, 255)
			img[:, y:(y + line_width), 1] = random.randint(0, 255)
			img[:, y:(y + line_width), 2] = random.randint(0, 255)

	if line_type == "horizontal":
		img = img.transpose(1,0,2)

	cv2.imwrite(this_output_path, img)
