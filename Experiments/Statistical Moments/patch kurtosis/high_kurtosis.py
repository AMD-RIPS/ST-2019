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
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


h,w = img.shape
orig_img = np.copy(img)


filter_width = 128
interval = filter_width//4
# anomaly = []

for i in range(0,h - filter_width, interval):
	for j in range(0,w - filter_width, interval):
		img = orig_img[i:i+filter_width, j:j+filter_width]

		height, width = img.shape
		area = height * width

		img = np.reshape(img, [area])
		O3 = kurtosis(img)

		if (np.abs(O3)>100):
			color_img[i:i+filter_width, j:j+filter_width, :] = 255



plt.imshow(color_img)
plt.show()






















