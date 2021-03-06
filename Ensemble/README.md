## Description:

This program can train and test the ensemble model which is built upon individual models for 10 types of artifact. 


## Examples:

### For training:

python3  train_test_ensemble.py  <normal_images_directory> <glitched_images_directory> train

### For testing:

python3  train_test_ensemble.py  <normal_images_directory> <glitched_images_directory> test


## Important Note (Please READ):

Some model files (e.g. PCA) are missing from the folders because the file size is too big.

For the model to run without error, please go to:

https://drive.google.com/open?id=1j9NC1SFPMl6BIZFIeQQjyLmk3VEaqGhf

Download all the .pkl files, and put them into the correct folders respectively.



### PCA models

discolor-PCA-500.pkl -> Put into folder "discoloration_test"

dots-PCA-300.pkl -> Put into folder "dotted_line_test"

paraline-PCA-300.pkl -> Put into folder "parallel_lines_test"

stut-PCA-300.pkl -> Put into folder "stuttering_test"

triang-PCA-400.pkl -> Put into folder "triangulation_test"



### SVM models

modelSVMmorse.pkl -> Put into folder "morse_code_test"

