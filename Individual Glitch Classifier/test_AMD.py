import cv2
import numpy as np
import sys
import pickle
import importlib
import time
import glob
import tensorflow as tf
import os, os.path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

normal_directory = sys.argv[1]
corrupted_directory = sys.argv[2]


valid_images = [".jpg",".gif",".png",".tga"]

tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print('loading the images...')
corrupted_images = np.array([cv2.imread(img) for img in glob.glob(corrupted_directory + "/*.png")])
normal_images =  np.array([cv2.imread(img) for img in glob.glob(normal_directory + "/*.png")])

print("loaded ", (normal_images.shape), " normal images and ", (corrupted_images.shape), " corrupted images...")
images = np.concatenate((normal_images[0:100], corrupted_images[0:100]), axis = 0)#############################
labels = np.concatenate((np.zeros(200), np.ones(0)), axis = 0)#####################################3

pred = np.zeros(len(images))

glitches = [ "morse_code", "screen_tearing", "parallel_lines", 
            "shader", 
            "line_pixelation",  "dotted_line",
             "stuttering", "triangulation"]
#"discoloration", "random_patch", "square_patch", "shape", "texture_popin",

gbl = globals()
start = time.time()
for glitch in glitches:
    filename =  glitch + "_test." +  glitch + "_test"
    gbl[filename] = importlib.import_module(filename)
    labels = gbl[filename].test(images)
    pred = np.logical_or(pred, labels)
    print('tested ', glitch, ' which predicted ', np.sum(labels) , '. So far the model has predicted ', np.sum(pred), ' corrupted images.')
    
print("It took ", (time.time()-start)/60, " minutes to pass ", len(images), " test images into the ensemble model.")
np.save("test_result.npy", pred)
print("saved the result")

print(classification_report(labels, pred))
print(confusion_matrix(labels, pred))
print("Accuracy of the model is: ", accuracy_score(labels, pred))


