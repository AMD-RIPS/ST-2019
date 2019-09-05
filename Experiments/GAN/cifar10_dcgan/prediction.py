import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import numpy.random as npr
import argparse, random, imutils
import pickle
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import cv2
from condensed_glitchify import *
from dcgan import Discriminator, Generator
from skimage import data, img_as_float
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

num_gpu = 1 if torch.cuda.is_available() else 0


# This is the threshold for prediction. If the output of the discriminator is greater than the 
# threshold, then an image is classified as fake.
THRESHOLD = 0.5

# Number of training data to be used. Should be at most 10000 (b/c the input data contains 10000 instances)
NUM_TRAINING_DATA = 100


# Name of the training dataset. There are 5 datasets in the folder:
# data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5
TRAINING_DATASET = "data_batch_1"



# load the models
D = Discriminator(ngpu=1).eval()
G = Generator(ngpu=1).eval()

D = D.double()


# load weights
D.load_state_dict(torch.load('weights/netD_epoch_199.pth', map_location='cpu'))
G.load_state_dict(torch.load('weights/netG_epoch_199.pth', map_location='cpu'))
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def produce_noise(img):
	new_img = np.random.randint(0,255, size = [32,32,3])
	return np.uint8(new_img)



# Functions that generates the corresponding glitches
funcs = {triangulation:"triangulation", parallel_lines: "parallel line", add_shaders: "shader", 
add_shapes: "shape", create_discoloration: "discoloration", add_random_patches: "random patch", 
produce_stuttering:"stuttering", line_pixelation: "line pixelation", dotted_lines: "dotted line",
dotted_lines_radial:"radial dotted line", blur: "blurring", square_patches: "square patch", 
produce_noise: "random noise"}


# Load data
img_dict = unpickle(TRAINING_DATASET)



for transform_function in funcs:
	# This is the actual number of corrputed images produced. It may be less than NUM_TRAINING_DATA
	# b/c the glitchfy functions may fail to insert artifacts to certain images.
	actual_num_instances = 0

	true_positive = 0
	for i in range(NUM_TRAINING_DATA):
		try:
			# Load the image
			img = img_dict[b'data'][i,:]

			colored_img= np.empty([32,32,3])
			colored_img[:,:,0] = img[:1024].reshape([32,32])
			colored_img[:,:,1] = img[1024:2048].reshape([32,32])
			colored_img[:,:,2] = img[2048:].reshape([32,32])
			colored_img = np.uint8(colored_img)


			# Visualize the image
			# plt.imshow(colored_img)
			# plt.show()


			# Insert artifacts
			new_img =transform_function(colored_img)


			# Visualize the resulting image
			# plt.imshow(new_img)
			# plt.show()


			# Prediction using the discriminator
			new_img = np.array([new_img.astype(float)])
			new_img_torch = torch.from_numpy(new_img.transpose(0,3,1,2))
			prediction = D(new_img_torch).detach().numpy()


			actual_num_instances += 1

			if prediction > THRESHOLD:
				true_positive += 1

		except Exception as e: 
			print(e)


	print("For " + funcs[transform_function] + ", ")
	print("The total number of corrupted images is: " + str(actual_num_instances))
	print("The number of true positive is: " + str(true_positive))
	print("The true positive rate is: " + str(true_positive / actual_num_instances))
	print("----------------------------------------------")















