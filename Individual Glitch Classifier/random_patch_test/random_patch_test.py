import cv2, os
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import sys
import timeit
from scipy import stats
from sklearn.metrics import confusion_matrix
import time


# Compute the abnormality score for each pixel in the input img
def compute_salience_matrix(img):
	width = h
	height = w
	num_pixels = area
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

	salience_matrix = np.empty([width, height])




	img = img.astype(float)



	img[:,:,0] = img[:,:,0] - mu[0]
	img[:,:,1] = img[:,:,1] - mu[1]
	img[:,:,2] = img[:,:,2] - mu[2]


	z2 = np.einsum("ijk, lk -> ijl", img, L)
	salience_matrix = np.einsum("ijk, ijk -> ij", img, z2)



	return salience_matrix


def is_glitched(img):
	h,w,_ = img.shape

	M = int(h / 21)
	N = int(w / 38)

	count = 0

	for i in range(0,h-M,M):
		for j in range(0, w-N, N):
			tile = np.mean(img[i:i+M, j:j+N, :], axis = 2)

			if np.all(tile == tile[0,0]) and tile[0,0] > 40:
				count += 1
				if count > 20:
					return 1


	return 0


h, w = 270, 480
area = h * w


def classify(img):
	img = cv2.resize(img, (w, h))


	salience_matrix = compute_salience_matrix(img)

	salience_list = np.reshape(salience_matrix, [area])
	var = np.var(salience_list)
	std = np.sqrt(var)
	salience_threshold = np.mean(salience_list) + 2 * std

	# print(salience_threshold)


	img[np.where(salience_matrix <= salience_threshold)] = 0

	# for i in range(h):
	# 	for j in range(w):
	# 		if salience_matrix[i,j] <= salience_threshold:
	# 			img[i,j,:] = 0


	
	flat = img.flatten()
	non_zero_flat = flat[np.nonzero(flat)]
	total = non_zero_flat.shape[0]
	p = stats.mode(non_zero_flat)

	if total == 0:
		return 0


	if p[1] > float(total) / 12:
		return 1
	return 0


def test(X):
	n = X.shape[0]
	y = np.empty(n)

	for i in range(n):

		# st = time.time()

		if is_glitched(X[i, :, :, :]):
			y[i] = 1
		elif classify(X[i, :, :, :]):
			y[i] = 1
		else:
			y[i] = 0

	return y





# wd = "np_data"

# glitch_type = "random_patch"

# X_train_1 = np.load("/home/IPAMNET/kjiang/Desktop/glitch_classifiers/normal_data/X_test_normal.npy")
# X_train_3 = np.load(os.path.join(wd, "X_test_" + glitch_type + ".npy"))

# # X_train_1 = X_train_1[:40,:,:,:]
# # X_train_3 = X_train_3[:40,:,:,:]

# y_1 = np.zeros(X_train_1.shape[0])
# y_3 = np.ones(X_train_3.shape[0])


# X_train = np.concatenate((X_train_1, X_train_3 ))
# y_train = np.concatenate((y_1, y_3))


# y_pred = np.empty(X_train.shape[0])


# idx = np.random.permutation(X_train.shape[0])
# X_train, y_train = X_train[idx], y_train[idx]



# st = time.time()
# y_pred = test(X_train)
# ed = time.time()

# print("total", ed - st)



# matrix = confusion_matrix(y_train, y_pred)

# print(matrix)














