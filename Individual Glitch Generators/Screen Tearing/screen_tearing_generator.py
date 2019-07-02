import numpy as np
import cv2
import sys
import os
from numpy import linalg as LA

video_path = sys.argv[1]
output_path = sys.argv[2]
interval = int(sys.argv[3])
max_num_pics = int(sys.argv[4])
glitch_type = sys.argv[5]

cap = cv2.VideoCapture(video_path)
count = 0
num_pics = 0

while(True):
	ret, frame = cap.read()  # ret = 1 if the video is captured; frame is the image
	if not ret:
		break

	global prev_img, image

	if count == 0:
		prev_img = frame
		count += 1
		continue
	elif count % interval == 0:
		img = frame
	else:
		count += 1
		continue


	height, width, channels = img.shape
	r = np.random.rand(1) * 0.8 + 0.1
	# r = np.random.rand(1)
	
	diff = img - prev_img
	diff = diff[:,:,0]

	# skip this iteration if two images are too similar
	if LA.norm(diff, "fro") < 1000:
		continue

	new_img = np.copy(img)

	if glitch_type == "horizontal":
		target_height = np.rint(r * height)
		target_height = target_height.astype(int)
		new_img[0:target_height[0], :, :] = prev_img[0:target_height[0], :, :]
	else:
		target_width = np.rint(r * width)
		target_width = target_width.astype(int)
		new_img[:, 0:target_width[0], :] = prev_img[:, 0:target_width[0], :]


	count += 1
	prev_img = img

	if num_pics < max_num_pics:
		#cv2.imshow('Video Capture', new_img)
		ending = str(num_pics) + "_" + glitch_type + ".png"
		this_output_path = os.path.join(output_path, ending)
		cv2.imwrite(this_output_path, new_img)
		num_pics += 1
		print("Writing " + str(num_pics) + "_" + glitch_type + ".png")
	else:
		break

cap.release()
