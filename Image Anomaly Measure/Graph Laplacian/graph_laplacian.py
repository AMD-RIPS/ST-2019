import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import sys
import cvxpy as cp
import timeit
from scipy import stats

input_path = sys.argv[1]
img = cv2.imread(input_path)

h,w,_ = img.shape
target_width = h
target_height = w

ori_img = np.copy(img)


# Compute the abnormality score for each pixel in the input img
def compute_salience_matrix(img):
	width, height,_ = img.shape
	# num_pixels = width * height
	feature_size = 3

	mu = np.average(np.average(img, axis = 0), axis = 0)
	alpha = float(np.average(mu))

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

	img = img.astype(float)

	img[:,:,0] = img[:,:,0] - mu[0]
	img[:,:,1] = img[:,:,1] - mu[1]
	img[:,:,2] = img[:,:,2] - mu[2]


	z2 = np.einsum("ijk, lk -> ijl", img, L)
	salience_matrix = np.einsum("ijk, ijk -> ij", img, z2)

	return salience_matrix






salience_matrix = compute_salience_matrix(img)
salience_list = np.reshape(salience_matrix, [h*w])
var = np.var(salience_list)
std = np.sqrt(var)
salience_threshold = max(np.mean(salience_list) + 2 * std, 6000)

newimg = np.copy(ori_img)

newimg[salience_matrix <= salience_threshold] = 0


# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()



ax[0].imshow(ori_img)
ax[0].set_title('Input image')
ax[1].imshow(newimg)
ax[1].set_title('Pixels with high anomaly measure')
# ax[2].imshow(ori_img)
# ax[2].set_title('Pixels with high anomaly measure (in original color)')
plt.show()












