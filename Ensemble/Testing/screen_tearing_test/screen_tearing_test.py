import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score
from sklearn.utils.extmath import randomized_svd
import time
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
import sys
import pickle




dim = (480, 270)
resize = lambda img: cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)


def HOG(img, norm = 'L2', ppc = 10, cpb = 3):
    pff = hog(img, orientations= 12, pixels_per_cell=(ppc, ppc),
              cells_per_block=(cpb, cpb), multichannel=True, transform_sqrt=True, block_norm= norm)
    return pff



def test(images):
    glitch = "screen_tearing_test/screen_tearing"
    pkl_filename = "screen_tearing_test/screen_tearing_LR_85.pkl"
    images = np.array([resize(img) for img in images])
    images = np.array([HOG(img) for img in images])

    # Load from file
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    
    # Calculate the accuracy score and predict target values
    Y = pickle_model.predict(images)
    return Y

