import numpy as np
import scipy, cv2
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import time

def test(X):
	st =  time.time()
	X = transform(X)
	ed = time.time()

	print("transform time: " + str(ed - st))
	
	clf = load('texture_popin_lr.joblib') 

	Y_pred = clf.predict(X)
	return Y_pred


def transform(X):
	n,_,_,_ = X.shape
	rt = []

	for i in range(n):
		img = X[i,:,:,:]

		lap = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F))
		perc = np.percentile(lap, 90)
		var_lap = np.var(lap.flatten())

		rt.append(np.array([perc, var_lap]))

	return np.asarray(rt)


# X_test_1 = np.load("X_test_normal.npy")
# X_test_2 = np.load("X_test_texture_popin.npy")


# y_1 = np.zeros(X_test_1.shape[0])
# y_2 = np.ones(X_test_2.shape[0])

# X_test = np.concatenate((X_test_1, X_test_2))
# y_test = np.concatenate((y_1, y_2))

# y_pred = test(X_test)
# print(confusion_matrix(y_test, y_pred))

