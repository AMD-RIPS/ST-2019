import cv2  
import numpy as np
from generator_functions import *
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pickle
import skimage as skimage
import skimage.feature as  ski_f
import timeit

# file = open('X_train.pickle', 'rb')
# data = pickle.load(file)
# x_train_list = []
# for i in range(735):
# 	cp = np.copy(data[i])
# 	x_train_list.append(cp)
# x_train_list = np.array(x_train_list)


# file = open('Y_train.pickle', 'rb')
# data = pickle.load(file)
# y_train_list = []
# for i in range(735):
# 	cp = np.copy(data[i])
# 	y_train_list.append(cp)
# y_train_list = np.array(y_train_list)


# file = open('X_test.pickle', 'rb')
# data = pickle.load(file)
# x_test_list = []
# for i in range(182):
# 	cp = np.copy(data[i])
# 	x_test_list.append(cp)
# x_test_list = np.array(x_test_list)


# file = open('Y_test.pickle', 'rb')
# data = pickle.load(file)
# y_test_list = []
# for i in range(182):
# 	cp = np.copy(data[i])
# 	y_test_list.append(cp)
# y_test_list = np.array(y_test_list)


# X_train, X_test, Y_train, Y_test = add_screen_tearing_data(x_train_list, x_test_list, y_train_list, y_test_list, "bd_screen_tearing 2", 0.8, 224, 224, 200)

# plt.imshow(X_train[-1])
# plt.show()

# plt.imshow(X_test[-1])
# plt.show()

# print(Y_test[-5:])

# np.save('X_train.npy', X_train)
# np.save('X_test.npy', X_test)
# np.save('Y_train.npy', Y_train)
# np.save('Y_test.npy', Y_test)

# exit()

hog = cv2.HOGDescriptor()

# Computing HOG vectors
def transform_features(X):
	p = ski_f.hog(X[0,:,:,:], orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), visualize=False, multichannel=True, feature_vector=True)

	# p = hog.compute(X[0,:,:,:])
	new_feature_size = p.shape[0]
	X_transform = np.empty([X.shape[0], new_feature_size])

	for i in range(X.shape[0]):
		# im = cv2.imread(X[i,:,:,:])
		X_transform[i,:] = ski_f.hog(X[i,:,:,:], orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), visualize=False, multichannel=True, feature_vector=True)

	return X_transform

# #generating data
# imgs_path = "imgs/normal_imgs/Mixture"
# screen_tearing_path = "imgs/screen_tearing"
# num_screen_tearing = 100
# percent_training = 0.5
# target_width = 224
# target_height =  224
# num_limit = 400

# X_train, X_test, Y_train, Y_test = create_data(imgs_path, percent_training, target_height, target_width, num_limit)
# X_train, X_test, Y_train, Y_test = add_screen_tearing_data(X_train, X_test, Y_train, Y_test, screen_tearing_path, percent_training, target_height, target_width, num_screen_tearing)


# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)


# np.save('X_train.npy', X_train)
# np.save('X_test.npy', X_test)
# np.save('Y_train.npy', Y_train)
# np.save('Y_test.npy', Y_test)

# exit()

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
Y_train = np.load('Y_train.npy')
Y_test = np.load('Y_test.npy')


# for i in range(500, X_train.shape[0],50):
# 	print(Y_train[i])
# 	plt.imshow(X_train[i])
# 	plt.show()


# X_train = X_train / 255
# X_test = X_test / 255

print("transforming training data")

X_train_transform = transform_features(X_train)
# u, s, vh = np.linalg.svd(X_train_transform, full_matrices=False)
# X_train_transform = np.dot(u[:,100], np.diag(s[100]))

print("transforming testing data")

X_test_transform = transform_features(X_test)
# u, s, vh = np.linalg.svd(X_test_transform, full_matrices=False)
# X_test_transform = np.dot(u[:,100], np.diag(s[100]))


print(X_train_transform.shape)
print(X_test_transform.shape)

start = timeit.default_timer()

clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
clf.fit(X_train_transform, Y_train)

stop = timeit.default_timer()
print('Traing Time: ', stop - start)

start = timeit.default_timer()

Y_pred = clf.predict(X_test_transform)

stop = timeit.default_timer()
print('Testing Time: ', stop - start)

print(confusion_matrix(Y_test, Y_pred))




