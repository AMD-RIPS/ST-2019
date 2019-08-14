import numpy as np
import cv2
import sklearn
import sys
import os
import pickle
from PIL import Image


def test(images):
    model_f='modelLogisticLinePix.pkl'
    pred(images,model_f)

def fouriertransform(img):
    f=np.fft.fft2(img)
    fshift=np.fft.fftshift(f)
    magnitude_spectrum=20*np.log(np.abs(fshift))
    return magnitude_spectrum
def fourierwindow(img):
    f=np.fft.fft2(img)
    fshift=np.fft.fftshift(f)
    newarray=np.zeros((1080,1920),dtype=complex)
    for y in list(range(img.shape[0])):
        for x in list(range(img.shape[1])):
            if((y<200 or y>900) and (x<200 or x>1700)):
                newarray[y,x]=fshift[y,x]
    newarray=np.fft.ifftshift(newarray)
    newarray=np.fft.ifft2(newarray)
    image=Image.fromarray(newarray.astype(np.uint8))
    image=np.array(image)
    return image

def pred(array_f,model_f):
    buff=np.load(array_f)
    test=np.zeros((len(buff),2*480*270),dtype=np.uint8)

    for i in list(range(len(buff))):
        img=buff[i]
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_fft=fouriertransform(img)
        img_fft=cv2.resize(img_fft,(img_fft.shape[1]//4,img_fft.shape[0]//4))
        img=fourierwindow(img)
        img=cv2.resize(img,(img.shape[1]//4,img.shape[0]//4))
        test[i]=np.append(img_fft.flatten(),img.flatten())

    model=pickle.load(open(model_f,'rb'))
    predictions=model.predict(test)
    return predictions

