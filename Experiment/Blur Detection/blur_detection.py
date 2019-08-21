import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import cv2
from scipy.stats import entropy, kurtosis, skew
import random
from skimage.feature import hog


input_path = sys.argv[1]
img = cv2.imread(input_path)
img = img[:,:,[2,1,0]]
color_img = np.copy(img)

h,w,_ = img.shape


kernel_size = 3
scale = 1
delta = 0
ddepth = cv2.CV_16S



data = []
orig_img = np.copy(img)


filter_width = 32
interval = int(filter_width/2)
anomaly = []


count = 0.0
total = 0.0
for i in range(0,h - filter_width, interval):
	for j in range(0,w - filter_width, interval):
		img = orig_img[i:i+filter_width, j:j+filter_width, :]
		h1, w1, _ = img.shape

		p = np.empty(3)
		for i in range(3):
			p[i] = np.var(img[:,:,i].reshape(-1))

		fd = np.reshape(img, [-1])

		var = np.var(fd)

		if (var < 1 and p[0] > 5 and p[1] > 5 and p[2] > 5):
			color_img[i:i+filter_width, j:j+filter_width, :] = 255
			count += 1

		total += 1

		data.append(var)


print("ratio of number of corrupted tiles over total number of tiles " + str(count/total))


plt.imshow(color_img)
plt.show()























