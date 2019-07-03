#### Description:

This program generates training and testing data. 


#### Usage: 

from generator_functions import *
X_train, X_test, Y_train, Y_test = create_data(imgs_path, percent_training, target_height, target_width, num_limit)\\


imgs_path is the path for the image folder
percentage_training is the percentage of overall data used for training
target_height, target width determine the dimension of training and testing data (images are resized to that dimension)
num_limit is the maximum number of samples (including both training and testing) to produce\\

X_train, X_test, Y_train, Y_test are numpy arrays with the following dimension:

X_train: n_train * target_width * target_height * 3, where n_train is the number of training samples
X_test: n_test * target_width * target_height * 3, where n_test is the number of testing samples
Y_train: n_train
Y_test: n_test

The labels are integers from 0 to c - 1, where c is the number of types of glitches determined by the operation_list (line147 in generator_functions.py). Right now the labels are:

0: normal image
1: vertical pattern (Morse code pattern)
2: slant pattern
3: checkerboard
4: blurring
5: vertical line (Random color and position)

#### Changing the Type and Frequency of Glitches

Modify the weight_list (line 151 in generator_functions.py) to change the relative frequency of different glitches. 
Modify the operation_list (line147 in generator_functions.py) to include different types of glitches.