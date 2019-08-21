import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import cv2
from scipy.stats import entropy, kurtosis, skew
import random
import scipy
from scipy import signal
from scipy import misc # pip install Pillow
from numpy import pi
from numpy import r_

input_path = sys.argv[1]
im = cv2.imread(input_path)
im = im[:,:,[2,1,0]]
color_img = np.copy(im)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


h,w= im.shape
M = h//4
N = w//4



def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0 ), axis=1 )



for x in range(0,im.shape[0],M):
	for y in range(0,im.shape[1],N):
		img = im[x:x+M,y:y+N]
		plt.imshow(color_img[x:x+M,y:y+N,:])
		plt.show()


		imsize = img.shape
		dct = np.zeros(imsize)

		data = []

		width = 8

		for i in r_[:imsize[0]:width]:
		    for j in r_[:imsize[1]:width]:
		        dct[i:(i+width),j:(j+width)] = dct2( img[i:(i+width),j:(j+width)] )
		        ## Non-DCT coefficients
		        data.append(dct[i,j+1])

		o3 = skew(data)
		o4 = kurtosis(data)


		print("skewnes: " + str(o3), "kurtosis: " +  str(o4))

		weights = np.ones_like(data)/float(len(data))
		plt.hist(data, weights=weights, bins=200)
		plt.show()







# imsize = im.shape
# dct = np.zeros(imsize)

# data = []

# width = 8

# for i in r_[:imsize[0]:width]:
#     for j in r_[:imsize[1]:width]:
#         dct[i:(i+width),j:(j+width)] = dct2( im[i:(i+width),j:(j+width)] )
#         data.append(dct[i,j+1])

# o2 = np.var(data)
# o3 = skew(data)
# o4 = kurtosis(data)


# # print(data)



# # print(o2, o3, o4)

# weights = np.ones_like(data)/float(len(data))
# hist, bins, _ = plt.hist(data, weights=weights, bins=50)
# plt.show()

# hist = hist[hist != 0]



# # plt.hist(hist2, bins=200)
# # plt.show()


# # weights = np.ones_like(data)/float(len(data))
# # hist, bins, _ = plt.hist(data, weights=weights, bins=50)
# # plt.show()

# # # histogram on log scale. 
# # # Use non-equal bin sizes, such that they look equal on log scale.
# #  #print()


# # logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
# # plt.hist(data, bins=logbins)
# # plt.xscale('log')
# # plt.show()


# # plt.hist(hist, bins=200)
# # plt.show()




