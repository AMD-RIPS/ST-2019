import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import time
import pickle
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def get_flat_FFT(X_test):
  dim = (480,270)

  X_list = []

  for i in range(X_test.shape[0]):
    ori_img = X_test[i]
    gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    img = cv2.resize(magnitude_spectrum, dim)
    X_list.append(img)
	
  X_resized = np.asarray(X_list)
	

  test_list = [] 
  for i in range(X_resized.shape[0]):
    img = X_resized[i].flatten()
    test_list.append(img)

  X_test = np.asarray(test_list)

  return X_test


def test(X_test):
	# Apply FFT and flatten the matrix 
  t1 = time.time()
  X_test_FFT = get_flat_FFT(X_test)
  t2 = time.time()
  print('Transform Time for triangulation: ', t2-t1)

	# Load PCA / CLF models
  with open('triang-PCA-400.pkl', 'rb') as file:
    pca = pickle.load(file)
  with open('triang-LDA-400.pkl', 'rb') as file:
    clf = pickle.load(file)

  # Apply PCA and make predictions
  t1 = time.time()
  X_pca = pca.transform(X_test_FFT)
  y_pred = clf.predict(X_pca)
  t2 = time.time()
  print('Remaining Time for triangulation: ', t2-t1)

  return y_pred  	

