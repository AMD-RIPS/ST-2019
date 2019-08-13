import cv2, os
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import sys
import timeit
from scipy import stats
from sklearn.metrics import confusion_matrix


# Compute the abnormality score for each pixel in the input img
def compute_salience_matrix(img):
	width, height, channel = img.shape
	num_pixels = height * width
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
	for i in range(width):
		for j in range(height):
			mean_centered_pixel = img[i,j,:] - mu
			salience_matrix[i, j] = LA.multi_dot([mean_centered_pixel, L, mean_centered_pixel])

	return salience_matrix


def is_glitched(img):
    h,w,_ = img.shape

    # M = 50
    # N = 50

    M = int(h / 21)
    N = int(w / 38)


    result = np.zeros_like(img)
    count = 0

    for i in range(0,h-M,M):
        for j in range(0, w-N, N):
            tile = np.mean(img[i:i+M, j:j+N, :], axis = 2)


            if np.all(tile == tile[0,0]) and tile[0,0] > 40:
                result[i:i+M, j:j+N,:]=255
                count += 1

    if count > 20:
    	return 1
    else:
    	return 0

# def is_glitched(img):
#     h,w,_ = img.shape

#     M = 30
#     N = 30


#     result = np.zeros_like(img)
#     count = 0

#     for i in range(0,h-M,M):
#         for j in range(0, w-N, N):
#             tile = np.mean(img[i:i+M, j:j+N, :], axis = 2)


#             if np.all(tile == tile[0,0]) and tile[0,0] > 40:
#                 result[i:i+M, j:j+N,:]=255
#                 count += 1

#     if count > 50:
#     	return 1, result, tile[0,0]
#     else:
#     	return 0, result, tile[0,0]



def classify(img):
	img = cv2.resize(img, (480, 270))
	# test_img = np.zeros_like(img)
	h,w,_ = img.shape
	area = h * w



	salience_matrix = compute_salience_matrix(img)
	salience_list = np.reshape(salience_matrix, [area])
	var = np.var(salience_list)
	std = np.sqrt(var)
	salience_threshold = np.mean(salience_list) + 2 * std

	for i in range(h):
		for j in range(w):
			if salience_matrix[i,j] <= salience_threshold:
				img[i,j,:] = 0
			# else:
			# 	test_img[i,j,:] = 255


	# plt.imshow(img)
	# plt.show()

	# plt.imshow(test_img)
	# plt.show()

	
	flat = img.flatten()
	non_zero_flat = flat[np.nonzero(flat)]
	total = non_zero_flat.shape[0]
	p = stats.mode(non_zero_flat)


	if p[1] > float(total) / 12:
		return 1
	return 0


def test(X):
	n = X.shape[0]
	y = np.empty(n)

	for i in range(n):
		y[i] = classify(X[i,:,:,:]) | is_glitched(X[i,:,:,:])

	return y



# wd = "np_data"

# glitch_type = "random_patch"

# X_train_1 = np.load("/home/IPAMNET/kjiang/Desktop/glitch_classifiers/normal_data/X_test_normal.npy")
# X_train_3 = np.load(os.path.join(wd, "X_test_" + glitch_type + ".npy"))

# X_train_1 = X_train_1[:40,:,:,:]
# X_train_3 = X_train_3[:40,:,:,:]

# y_1 = np.zeros(X_train_1.shape[0])
# y_3 = np.ones(X_train_3.shape[0])


# X_train = np.concatenate((X_train_1, X_train_3 ))
# y_train = np.concatenate((y_1, y_3))


# y_pred = np.empty(X_train.shape[0])


# idx = np.random.permutation(X_train.shape[0])
# X_train, y_train = X_train[idx], y_train[idx]

# y_pred = test(X_train)



# matrix = confusion_matrix(y_train, y_pred)

# print(matrix)


















# h, w = 270, 480
# area = h * w



# correct = 0.0
# total = 0.0

# for i in range(X_train.shape[0]):
# 	img = X_train[i,:,:,:]
# 	p, t, l = classify(img)
# 	q, r, color = is_glitched(img)

# 	y_pred[i] = ((p + q) > 0) * 1
# 	# print(y_train[i], p, q)


# 	# if y_train[i] != y_pred[i]:
# 	# 	print(color, l)
# 	# 	plt.imshow(img)
# 	# 	plt.show()

# 	# 	plt.imshow(r)
# 	# 	plt.show()

# 	# 	plt.imshow(t)
# 	# 	plt.show()
# 	total += 1
# 	if y_train[i] == y_pred[i]:
# 		correct += 1

# 	if i % 20 == 0:
# 		print("accuracy", total, correct / total)



# matrix = confusion_matrix(y_train, y_pred)

# print(matrix)


