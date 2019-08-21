import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import cv2
from scipy.stats import entropy, kurtosis, skew
import random


input_path = sys.argv[1]
img = cv2.imread(input_path)
img = img[:,:,[2,1,0]]
color_img = np.copy(img)


h,w,_ = img.shape



data = []
orig_img = np.copy(img)


filter_width = 8


# horizontal
for i in range(0, h, filter_width):
	img = orig_img[i:i+filter_width,:,:]

	height, width, _ = img.shape
	area = height * width

	img = np.reshape(img, [area*3])
	O3 = np.var(img)
	# O3 = skew(img)

	if (np.abs(O3)> 4000):
		color_img[i:i+10,:,0] = 255



plt.imshow(color_img)
plt.show()





















