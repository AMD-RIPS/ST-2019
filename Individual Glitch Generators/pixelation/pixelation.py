import cv2
import numpy as np
import os
import sys
import random

input_path =  sys.argv[1]
output_path = sys.argv[2]
max_blob_radius = 49 sys.argv[3]

count = 0
for image_path in os.listdir(input_path):
    print("Processing: " + image_path)
    img = cv2.imread(os.path.join(input_path, image_path))
    ending = str(count) + "_pixelation.png"
    this_output_path = os.path.join(output_path, ending)
    count += 1
    
    if not image_path.endswith('.png') and not image_path.endswith('.jpg') and not image_path.endswith('.jpeg'):
        continue
    width, height, channel = img.shape
    square_size = int(height / 32)
    x_start = np.random.randint(1, width - max_blob_radius - 1)
    y_start = np.random.randint(1, height - max_blob_radius - 1)
    brighter = np.random.randint(50, 150)
    
    blob_width = square_size * np.random.randint(1, max_blob_radius/square_size) #in pixels
    color = np.minimum(img[x_start, y_start, :]+ brighter, 255)
    for x in range(x_start, x_start + blob_width, square_size):
        blob_height = np.random.randint(1, max_blob_radius/square_size)
        for y in range(y_start, y_start + blob_height*square_size, square_size):
            try:
                img[x:x+square_size , y:y+square_size , :] = color
            except:
                print(width, height, x, y, x_start, y_start, blob_width)
                raise(error)


cv2.imwrite(this_output_path, img)

