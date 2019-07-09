import numpy as np
import cv2
import sys
import os
from numpy import linalg as LA
import random

video_path = sys.argv[1]
glitch_type = sys.argv[2]
starting_frame = int(sys.argv[3])
ending_frame = int(sys.argv[4])

cap = cv2.VideoCapture(video_path)
count = 0

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 
out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while(True):
	ret, frame = cap.read()  # ret = 1 if the video is captured; frame is the image
	if not ret:
		break

	global prev_img

	if count == 0:
		prev_img = frame
		count += 1
		continue
	
	if not (count >= starting_frame and count <= ending_frame):
		count += 1
		continue


	height, width, channels = frame.shape
	r = np.random.rand(1) * 0.8 + 0.1

	new_img = np.copy(frame)

	if glitch_type == "horizontal":
		target_height = np.rint(r * height)
		target_height = target_height.astype(int)
		new_img[0:target_height[0], :, :] = prev_img[0:target_height[0], :, :]
		# print("?")
	else:
		target_width = np.rint(r * width)
		target_width = target_width.astype(int)
		new_img[:, 0:target_width[0], :] = prev_img[:, 0:target_width[0], :]

	out.write(new_img)

	count += 1
	prev_img = frame

cap.release()
out.release()
