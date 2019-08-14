import cv2, os
import numpy as np
from matplotlib import pyplot as plt
import sys, random, time
from random import shuffle
from sklearn.metrics import confusion_matrix
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from numpy import linalg as LA


# h, w = 1080, 1920
# area = h * w

# h1, w1 = 1080.0, 1920.0


def compute_salience_matrix(img):
	width, height, _ = img.shape
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

	interval = 10
	for i in range(h):
		for j in range(w):
			if salience_matrix[i, j] > salience_threshold:
				p[i - interval :i+ interval,j] = 1
				p[i,j-interval:j+interval] = 1

	p[salience_matrix > salience_threshold] = 1


	a = np.amax(np.count_nonzero(p, axis = 0) / h1)
	b = np.amax(np.count_nonzero(p, axis = 1) / w1)

	return max(a,b)


	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# f = np.fft.fft2(gray)
	# fshift = np.fft.fftshift(f)
	# magnitude_spectrum = 20*np.log(np.abs(fshift))


	# plt.imshow(magnitude_spectrum)
	# plt.show()

	
	# flat = img.flatten()
	# non_zero_flat = flat[np.nonzero(flat)]
	# total = non_zero_flat.shape[0]
	# p = stats.mode(non_zero_flat)

	# if total == 0:
	# 	return 0


	# if p[1] > float(total) / 12:
	# 	return 1
	# return 0


def test(X):
	n= X.shape[0]
	rt = []

	for i in range(n):
		# print(classify(X[i,:,:,:]))
		rt.append(classify(X[i,:,:,:]) > 0.6)

	return np.asarray(rt)










def main():
	# glitch_type = "line_pixelation"
	# wd = "np_data"

	# X_test_1 = np.load("/home/IPAMNET/kjiang/Desktop/glitch_classifiers/normal_data/X_test_normal.npy")
	# #X_test_1 = transform(X_test_1,)


	# X_test_2 = np.load(os.path.join(wd, "X_test_" + glitch_type + ".npy"))
	# #X_test_2 = transform(X_test_2, 1)


	# test(X_test_2)



	# y_1 = np.zeros(X_test_1.shape[0])
	# y_2 = np.ones(X_test_2.shape[0])

	# X_test = np.concatenate((X_test_1, X_test_2))
	# y_test = np.concatenate((y_1, y_2))


	# y_pred = test(X_test)

	# print(confusion_matrix(y_test, y_pred))




	# # clf = SVC(gamma='auto')
	# clf.fit(X_train, Y_train) 
	# # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
	# # 	decision_function_shape='ovr', degree=1, gamma='auto', kernel='rbf',
	# # 	max_iter=-1, probability=False, random_state=None, shrinking=True,
	# # 	tol=0.001, verbose=False)


	# # from joblib import dump, load
	# # dump(clf, 'blur_lr.joblib') 

	# st = time.time()
	# Y_pred = clf.predict(X_test)
	# ed = time.time()
	# print(ed-st, "time")




if __name__ == '__main__':
	main()