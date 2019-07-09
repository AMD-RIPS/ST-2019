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

mu_r = np.average(np.average(img[:,:,0]))
mu_g = np.average(np.average(img[:,:,1]))
mu_b = np.average(np.average(img[:,:,2]))
alpha = (mu_r + mu_g + mu_b) / 3.0

def compute_Cauchy_weight(a, b, alpha):
	alpha = float(alpha)
	return 1.0 / (1 + np.power((a-b)/alpha, 2))

def compute_salience(L, pixel):
	mean_centered_pixel = pixel - np.array([mu_r, mu_g, mu_b])
	return LA.multi_dot([mean_centered_pixel, L, mean_centered_pixel])\

w_rg = compute_Cauchy_weight(mu_r, mu_g, alpha)
w_rb = compute_Cauchy_weight(mu_r, mu_b, alpha)
w_gb = compute_Cauchy_weight(mu_g, mu_b, alpha)

degree_arr = np.array([w_rg + w_rb, w_rg + w_gb, w_gb + w_rb])

W = np.array([[0, w_rg, w_rb], [w_rg, 0, w_gb], [w_rb, w_gb, 0]])
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