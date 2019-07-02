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

num_limit = 50
target_width = 224
target_height = 224

X_train, X_test, Y_train, Y_test = create_data(imgs_path, percent_training, target_height, target_width, num_limit)


