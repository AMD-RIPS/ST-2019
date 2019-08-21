import cv2
import numpy as np
import random

# in_name='bd-original-14.jpg'
# out_name='stuttering_test.png'

def divisors(n):
	divisors=[]
	for i in list(range(1,int(n/2)+1)):
		if (n%i==0):
			divisors.append(i)
	divisors.append(n)
	return divisors

def swap(image,pix1,pix2):
	dummyr=image[pix1[0],pix1[1],0]
	dummyg=image[pix1[0],pix1[1],1]
	dummyb=image[pix1[0],pix1[1],2]
	image[pix1[0],pix1[1],0]=image[pix2[0],pix2[1],0]
	image[pix1[0],pix1[1],1]=image[pix2[0],pix2[1],1]
	image[pix1[0],pix1[1],2]=image[pix2[0],pix2[1],2]
	image[pix2[0],pix2[1],0]=dummyr
	image[pix2[0],pix2[1],1]=dummyg
	image[pix2[0],pix2[1],2]=dummyb

def stutter(img,sizex,sizey):
	vstripes=int(img.shape[1]/sizex)
	hstripes=int(img.shape[0]/sizey)
	for k in list(range(0,img.shape[1],2*int(img.shape[1]/vstripes))):
		for i in list(range(int(img.shape[1]/vstripes))):
			for j in list(range(img.shape[0])):
				if(i+k+int(img.shape[1]/vstripes)<img.shape[1]):
					swap(img,[j,i+k],[j,int(img.shape[1]/vstripes)+i+k])
	for k in list(range(0,img.shape[0],2*int(img.shape[0]/hstripes))):
		for i in list(range(int(img.shape[0]/hstripes))):
			for j in list(range(img.shape[1])):
				if(i+k+int(img.shape[0]/hstripes)<img.shape[0]):
					swap(img,[i+k,j],[i+k+int(img.shape[0]/hstripes),j])
	return img


def produce_stuttering(img):
	vdivisors=divisors(img.shape[0])
	hdivisors=divisors(img.shape[1])
	iterations=np.random.choice(np.arange(1,5),p=[0.65,0.2,0.1,0.05])
	for i in list(range(iterations)):
		sizex=random.choice(hdivisors)
		sizey=random.choice(vdivisors)
		img=stutter(img,sizex,sizey)
	return img


# img=np.array(cv2.imread(in_name,1))
# vdivisors=divisors(img.shape[0])
# hdivisors=divisors(img.shape[1])
# iterations=np.random.choice(np.arange(1,5),p=[0.65,0.2,0.1,0.05])
# for i in list(range(iterations)):
# 	sizex=random.choice(hdivisors)
# 	sizey=random.choice(vdivisors)
# 	img=stutter(img,sizex,sizey)
# cv2.imwrite(out_name,img)