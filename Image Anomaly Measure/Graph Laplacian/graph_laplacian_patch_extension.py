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

width, height, channel = img.shape
patch_width = 32
patch_height = 32

patch_matrix_width = width / patch_width
patch_matrix_height = height / patch_height

p = ski_f.hog(np.zeros([patch_width,patch_height,3]), orientations=9, pixels_per_cell=(16, 16),
	cells_per_block=(2, 2), visualize=False, multichannel=True, feature_vector=True)
feature_size = p.shape[0]

patch_matrix = np.zeros([patch_matrix_width, patch_matrix_height, feature_size])

for i in range(patch_matrix_width):
	for j in range(patch_matrix_height):
		patch_matrix[i,j,:] = ski_f.hog(img[i*patch_width:(i+1)*patch_width, j*patch_height:(j+1)*patch_height,:], orientations=9, 
			pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False, multichannel=True, feature_vector=True)
		# print(sum(patch_matrix[i,j,:]))

mu = np.average(np.average(patch_matrix, axis = 0), axis = 0)
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

salience_matrix = np.empty([patch_matrix_width,patch_matrix_height])
for i in range(patch_matrix_width):
	for j in range(patch_matrix_height):
		salience_matrix[i, j] = compute_salience(L_sym, patch_matrix[i,j,:])

print(salience_matrix)


