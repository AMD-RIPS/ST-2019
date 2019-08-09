import numpy as np
import cv2
import sklearn
import sys
import os
import pickle

array=sys.argv[1]
scaler=sys.argv[2]
model=sys.argv[3]

def fouriertransform(img):
    buff=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    f=np.fft.fft2(buff)
    fshift=np.fft.fftshift(f)
    magnitude_spectrum=20*np.log(np.abs(fshift))
    return magnitude_spectrum

def pred(array,scale,model):
    buff=np.load(array)
    test=np.zeros((len(buff),1920*1080//16),dtype=np.uint8)

    for i in list(range(len(buff))):
        img=buff[i]
        img_fft=fouriertransform(img)
        img_fft=cv2.resize(img_fft,(img_fft.shape[1]//4,img_fft.shape[0]//4))
        test[i]=img_fft.flatten()

    model=pickle.load(open(model,'rb'))
    predictions=model.predict(test)
    return predictions

pred(array,scaler,model)
