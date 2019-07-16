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

input_path = sys.argv[1]
img = cv2.imread(input_path)
# plt.imshow(img)
# plt.show()

width, height, channel = img.shape
num_pixels = height * width
feature_size = 3

mu = np.average(np.average(img, axis = 0), axis = 0)
alpha = float(np.average(mu))

def compute_salience(L, pixel):
	mean_centered_pixel = pixel - mu
	return LA.multi_dot([mean_centered_pixel, L, mean_centered_pixel])

W = np.zeros([feature_size, feature_size])
for i in range(feature_size):
	for j in range(feature_size):
		if i == j:
			continue
		else:
			W[i, j] = 1.0 / (1 + np.power((mu[i] - mu[j]) / alpha, 2))

degree_arr = np.sum(W, axis = 0)
D = np.diag(degree_arr)
D_sqrt_inv = np.diag(1.0 / np.sqrt(degree_arr))
L = D - W
L_sym = LA.multi_dot([D_sqrt_inv, L, D_sqrt_inv])

# print(L_sym)

# start = timeit.default_timer()
# img_vec = img.reshape([-1, 3])
# compute_salience_given_L = partial(compute_salience, L_sym)
# salience_matrix = np.apply_along_axis(compute_salience_given_L, 1, img_vec).reshape([width, height])
# stop = timeit.default_timer()
# print('Time: ', stop - start)

salience_matrix = np.empty([width, height])
for i in range(width):
	for j in range(height):
		salience_matrix[i, j] = compute_salience(L_sym, img[i,j,:])

print(salience_matrix)