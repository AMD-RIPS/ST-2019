import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import random
from functools import partial
import numpy as np
import sys
from generator_functions import *


imgs_path = sys.argv[1]
percent_training = float(sys.argv[2])
num_limit = int(sys.argv[3])
target_width = int(sys.argv[4])
target_height = int(sys.argv[5])

X_train, X_test, Y_train, Y_test = create_data(imgs_path, percent_training, target_height, target_width, num_limit)

print(X_train.shape)
print(X_test.shape)

print(Y_train)
print(Y_test)


for i in range(X_train.shape[0]):
	plt.imshow(X_train[i])
	plt.show()