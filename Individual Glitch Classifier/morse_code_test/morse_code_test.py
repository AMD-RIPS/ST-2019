import numpy as np
import cv2
import sklearn
import sys
import os
import pickle

MODEL_NAME = 'morse_code_test/modelSVMmorse.pkl'
def test(images):
    model_f = MODEL_NAME  #change this for the actual path + name
    return pred(images,model_f)

def fouriertransform(img):
    buff=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    f=np.fft.fft2(buff)
    fshift=np.fft.fftshift(f)
    magnitude_spectrum=20*np.log(np.abs(fshift))
    return magnitude_spectrum

def pred(buff ,model_f):
    
    test=np.zeros((len(buff),1920*1080//16),dtype=np.uint8)

    for i in list(range(len(buff))):
        img=buff[i]
        img_fft=fouriertransform(img)
        img_fft=cv2.resize(img_fft,(480,270))
        test[i]=img_fft.flatten()

    model=pickle.load(open(model_f,'rb'))
    predictions=model.predict(test)
    return predictions
