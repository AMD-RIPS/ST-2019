import cv2
import matplotlib.pyplot as plt
import skimage as skimage
import skimage.feature as  ski_f
from skimage import data, color, exposure
from functools import partial
import numpy as np
from numpy import linalg as LA
import sys
import time
import cvxpy as cp
import timeit

new_img = np.empty([10, 10, 3])
for i in range(10):
	for j in range(10):
		new_img[i,j,:] = 0

new_img[2, 3, :] = np.array([255,255,255])
print(new_img.shape)

cv2.imwrite("abnormal_img.jpg", new_img)

plt.imshow(new_img)
plt.show()
