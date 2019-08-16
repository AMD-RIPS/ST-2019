import cv2, os
import numpy as np
from matplotlib import pyplot as plt
import sys, random, time
from random import shuffle
from sklearn.metrics import confusion_matrix
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from numpy import linalg as LA
import time
from functools import partial





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



def add_neighbors(a, salience_threshold, interval):
	z = np.zeros_like(a)

	length = a.shape[0]
	for i in range(length):
		if a[i] > salience_threshold:
			z[i-interval:i+interval] = 1

	return np.count_nonzero(z) / float(length)



def dilatation(src, dilatation_size):
	#max_elem = 2
	#max_kernel_size = 21
	#dilatation_size = 10
	#dilatation_type = 0

	# dilatation_type = cv2.MORPH_RECT

	element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))

	dilatation_dst = cv2.dilate(src, element)
	return dilatation_dst



def classify(img):
	h,w,_ = img.shape
	h1 = float(h)
	w1 = float(w)
	p = np.zeros([h,w])
	

	salience_matrix = compute_salience_matrix(img)

	salience_list = np.reshape(salience_matrix, [-1])
	var = np.var(salience_list)
	std = np.sqrt(var)
	salience_threshold = max(np.mean(salience_list) + 2 * std, 6000)


	interval = h // 50

	p[salience_matrix > salience_threshold] = 1


	p = dilatation(p, interval)

	a = np.amax(np.count_nonzero(p, axis = 0) / h1)
	b = np.amax(np.count_nonzero(p, axis = 1) / w1)

	return max(a,b)



def test(X):
	n= X.shape[0]
	rt = []

	for i in range(n):
		# print(classify(X[i,:,:,:]))
		rt.append((classify(X[i,:,:,:]) > 0.85)*1)

	return np.asarray(rt)






# # img = cv2.imread("pixelation2.bmp")
# # print(test(np.array([img])))



# def main():
# 	glitch_type = "line_pixelation"
# 	wd = "np_data"

# 	X_test_1 = np.load("/home/IPAMNET/kjiang/Desktop/glitch_classifiers/normal_data/X_test_normal.npy")
# 	X_test_2 = np.load(os.path.join(wd, "X_test_" + glitch_type + ".npy"))

# 	# test(X_test_2)

# 	y_1 = np.zeros(X_test_1.shape[0])
# 	y_2 = np.ones(X_test_2.shape[0])

# 	X_test = np.concatenate((X_test_1, X_test_2))
# 	y_test = np.concatenate((y_1, y_2))

# 	st = time.time()
# 	y_pred = test(X_test)
# 	ed = time.time()
# 	print(ed - st)

# 	print(confusion_matrix(y_test, y_pred))





# if __name__ == '__main__':
# 	main()