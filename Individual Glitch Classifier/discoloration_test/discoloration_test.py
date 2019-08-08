import sys
import logging, cv2
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
# import cnn_discoloration

MODEL_PATH = "cnn_discoloration"

n_epoch = 180
new_x = 300
new_y = 300
BATCH_SIZE = 32
nb_classes = 2


IMAGE_INPUT_SIZE = (new_x, new_y)
transpose_size = (new_y, new_x)

def build_model():
    convnet = input_data(shape=[None, new_x, new_y, 3], name='input')
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, 256, activation='relu')
    convnet = fully_connected(convnet, nb_classes, activation='softmax')
    convnet = regression(convnet, optimizer='sgd', learning_rate=0.01, 
        loss='binary_crossentropy', name='targets')

    return convnet


def transform(X):
    n,h,w,_ = X.shape
    new_X = np.empty([n,new_x,new_y, 3])
    radius = 1
    n_points = 1

    for i in range(n):
        img = cv2.resize(X[i,:,:,:],(new_y, new_x))

        new_X[i,:,:,:] = img

    return new_X



def test(X):
    model = build_model()
    model = tflearn.DNN(model)

    model.load( MODEL_PATH + '.model')

    X_test = transform(X)

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis = 1)

    return y_pred

# def load_test(test_normal_path, test_glitched_path):
#     X_test_1 = np.load(test_normal_path)
#     X_test_2 = np.load(test_glitched_path)


#     y_1 = np.zeros(X_test_1.shape[0])
#     y_2 = np.ones(X_test_2.shape[0])

#     X_test = np.concatenate((X_test_1, X_test_2))
#     y_test = np.concatenate((y_1, y_2))


#     idx = np.random.permutation(X_test.shape[0])
#     X_test, y_test = X_test[idx], y_test[idx]


#     return X_test, y_test


# def main():
#     test_normal_path = "/home/IPAMNET/kjiang/Desktop/glitch_classifiers/normal_data/X_test_normal.npy"
#     test_glitched_path = "/home/IPAMNET/kjiang/Desktop/glitch_classifiers/discoloration/discoloration/np_data/X_test_discoloration.npy"

#     X_test, Y_test = load_test(test_normal_path, test_glitched_path)

#     y_pred = test(X_test)
#     matrix = confusion_matrix(Y_test, y_pred)
#     print(matrix)


# if __name__ == '__main__':
#   main()

