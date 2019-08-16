import cv2
import numpy as np
import sys
import pickle
import importlib
import time
import glob
import tensorflow as tf
import os, os.path
import gc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
"""
normal_directory = sys.argv[1]
corrupted_directory = sys.argv[2]
"""
normal_directory = "/home/IPAMNET/pdavarmanesh/Documents/images/new_heldout_testing/normal/"
corrupted_directory = "/home/IPAMNET/pdavarmanesh/Documents/images/new_heldout_testing/glitched/"


glitches = ["shape", "line_pixelation",  "shader", "morse_code", "parallel_lines", "dotted_line",
             "stuttering", "triangulation","discoloration",
              "random_patch",  "screen_tearing"] ##screen tearing must come last

normal_list = []
corrupted_list = []
true_labels = []

print('loading the images...')
normal = ([cv2.imread(img) for img in glob.glob(normal_directory + "/*.png")])
for i in range(0,1650,150):
    normal_list.append(np.array(normal[i:i+150]))
del normal
gc.collect()
true_labels.append(np.zeros(1650))


for i, glitch in enumerate(glitches[:-1]):
    with open(corrupted_directory + glitch + '/X.pkl', 'rb') as file:
        corrupted_images = pickle.load(file)
        corrupted_list.append(np.array(corrupted_images[:150]))
        print(glitch, i+1)
        true_labels = np.append(true_labels, np.ones(150)*(i+1))


ST = ([cv2.imread(img) for img in glob.glob(corrupted_directory + 'screen_tearing' + "/*.png")])
true_labels = np.append(true_labels, np.ones(150)*11)
corrupted_list.append(np.array(ST[:150]))
del ST
gc.collect()

#print(np.unique(true_labels))
print('Extracted {} normal emages and {} of each glitch'.format(1650, 150))
print(len(normal_list), len(corrupted_list))
print(len(true_labels))

print("loaded the images...")
ens_pred = np.zeros((len(glitches),len(true_labels) ))

gbl = globals()

for i, glitch in enumerate(glitches):
    print('testing for ', glitch)
    start = time.time()
    filename =  glitch + "_test." +  glitch + "_test"
    gbl[filename] = importlib.import_module(filename)
    pred = np.array([])
    for images in normal_list:
         pred = np.append(pred, gbl[filename].test(images)) 
    for images in corrupted_list:
         pred = np.append(pred, gbl[filename].test(images))
         print(pred.shape)
         print(pred)
    
    ens_pred[i] = pred
    
    print('tested ', glitch, ' which predicted ', np.sum(pred) , 'in {} minutes'.format((time.time()-start)/60))
    
    print(confusion_matrix(true_labels, pred))
    
    labels = (true_labels == (i+1)) * 1 #bc i starts at 0 glitch labels start at 1
    print(confusion_matrix(labels, pred))
    idx = np.arange(len(labels))
    print(len(idx), len(pred), len(labels))
    false_neg_idx = idx[(pred - labels) == -1]
    false_pos_idx = idx[(pred - labels) == 1]
    """
    print(pred[0:20])
    print(labels[0:20])
    print(np.sum(labels))
    print(len(false_neg_idx))
    print(len(false_pos_idx))
    print(np.sum(np.logical_and(pred == labels, pred == 1)))
    
    for idx in false_neg_idx[:10]:
        list_idx = idx // 200
        img_idx = idx % 200
        cv2.imwrite('false_negatives/'+ glitch + "_"+ str(idx) +'.jpg', images_list[list_idx][img_idx])
    
    for idx in false_pos_idx[:10]:
        list_idx = idx // 200
        img_idx = idx % 200
        cv2.imwrite('false_positives/'+ glitch + "_"+ str(idx) +'.jpg', images_list[list_idx][img_idx])
    """
  

np.save("ens_pred.npy", ens_pred)
print("saved the result")  

true_labels = (true_labels != 0) * 1.0
ens_pred = np.transpose(ens_pred)

X_train, X_test, y_train, y_test = train_test_split(ens_pred, true_labels, test_size=0.25, random_state=42)
start = time.time()
LR = LogisticRegression(solver = 'lbfgs', multi_class = 'auto', max_iter = 1000)
LR.fit(X_train, y_train)
print('LR training time: ', time.time() - start)
y_pred = LR.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("Accuracy of the model is: ", accuracy_score(y_test, y_pred))








"""
print(classification_report(labels, pred))
print(confusion_matrix(labels, pred))
print("Accuracy of the model is: ", accuracy_score(labels, pred))

"""
