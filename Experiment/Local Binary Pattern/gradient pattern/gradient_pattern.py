import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
from skimage.feature import local_binary_pattern
from scipy import ndimage
from skimage import feature
from skimage.feature import hog
from scipy.stats import entropy, kurtosis, skew


input_path = sys.argv[1]


radius = 3
n_points = 1

img = cv2.imread(input_path)
img = img[:,:,[2,1,0]]
c_img = np.copy(img)

w,h,_ = img.shape

img = np.mean(img, axis = 2)

kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
ky = np.array([[1,2,1] ,[0,0,0], [-1,-2,-1]])

x=ndimage.convolve(img,kx)
y=ndimage.convolve(img,ky)

g = np.sqrt(x**2+y**2)

lbp = local_binary_pattern(img, n_points, radius, "uniform")

# print(lbp > 0)
p = (lbp > 0)

plt.imshow(p)
plt.show()
