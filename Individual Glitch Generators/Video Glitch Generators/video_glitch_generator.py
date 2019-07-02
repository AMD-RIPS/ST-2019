import cv2
import matplotlib.pyplot as plt
import skimage as skimage
import skimage.feature as  ski_f
from skimage import data, color, exposure
import numpy as np
import sys
import time
from generator_functions import *

input_path = sys.argv[1]
starting_frame = int(sys.argv[2])
ending_frame = int(sys.argv[3])

import numpy as np
import cv2
cap = cv2.VideoCapture(input_path)


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

count = 0
while(1):
    ret,frame = cap.read()

    if not ret:
        break
    
    if count >= starting_frame and count <= ending_frame:
    	frame = add_checkerboard_pattern(1, 0, frame)

    out.write(frame)

    count += 1
cv2.destroyAllWindows()
out.release()
cap.release()