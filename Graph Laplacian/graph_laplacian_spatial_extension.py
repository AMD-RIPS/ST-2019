import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
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

# dx = ndimage.sobel(img, 0)
# dy = ndimage.sobel(img, 1)

# print(dx)

mu_r = np.average(np.average(img[:,:,0]))
mu_g = np.average(np.average(img[:,:,1]))
mu_b = np.average(np.average(img[:,:,2]))
alpha = (mu_r + mu_g + mu_b) / 3.0

padded_img = np.zeros([width+2, height+2, 3])
padded_img[1:-1,1:-1,:] = img


def compute_Cauchy_weight(a, b, alpha):
	alpha = float(alpha)
	return 1.0 / (1 + np.power((a-b)/alpha, 2))

def compute_salience(L, avg, i, j):
	s = np.concatenate((padded_img[i+1,j+1,:], padded_img[i+2,j+1,:], padded_img[i+1,j,:], padded_img[i+2,j+1,:], padded_img[i+1,j+2,:]), axis = 0)
	s -= avg
	return LA.multi_dot([s, L, s])

w_rg = compute_Cauchy_weight(mu_r, mu_g, alpha)
w_rb = compute_Cauchy_weight(mu_r, mu_b, alpha)
w_gb = compute_Cauchy_weight(mu_g, mu_b, alpha)

degree_arr = np.zeros(15)
degree_arr[0] = w_rg + w_rb + 4
degree_arr[1] = w_rg + w_gb + 4
degree_arr[2] = w_rb + w_gb + 4
for p in range(4):
	degree_arr[p * 3 + 3] = degree_arr[0] - 3
	degree_arr[p * 3 + 4] = degree_arr[1] - 3
	degree_arr[p * 3 + 5] = degree_arr[2] - 3

W = np.zeros([15, 15])
for i in range(3):
	for j in range(4):
		W[i,i+j*3+3] = 1
		W[i+j*3+3, i] = 1

for i in range(5):
	W[3*i,3*i+1] = w_rg
	W[3*i+1, 3*i] = w_rg

	W[3*i+1,3*i+2] = w_gb
	W[3*i+2,3*i+1] = w_gb

	W[3*i,3*i+2] = w_rb
	W[3*i+2, 3*i] = w_rb

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

sum = np.concatenate((padded_img[i+1,j+1,:], padded_img[i+2,j+1,:], padded_img[i+1,j,:], padded_img[i+2,j+1,:], padded_img[i+1,j+2,:]), axis = 0)
p = sum.shape[0]
sum = np.zeros(p)
for i in range(width):
	for j in range(height):
		sum += np.concatenate((padded_img[i+1,j+1,:], padded_img[i+2,j+1,:], padded_img[i+1,j,:], padded_img[i+2,j+1,:], padded_img[i+1,j+2,:]), axis = 0)

avg = sum / float(num_pixels)


salience_matrix = np.empty([width, height])
for i in range(width):
	for j in range(height):
		salience_matrix[i, j] = compute_salience(L_sym, avg, i, j)

print(salience_matrix)