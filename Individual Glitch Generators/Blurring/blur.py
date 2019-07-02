import cv2  
import os
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]
blur_type = sys.argv[3]
max_num = int(sys.argv[4])
count = 0
p1 = 0
p2 = 0
p3 = 0


if blur_type == "avg":
	p1 = int(sys.argv[4])
	p2 = int(sys.argv[5])
elif blur_type == "bilateral":
	p1 = int(sys.argv[4])
	p2 = int(sys.argv[5])
	p3 = int(sys.argv[6])

for image_path in os.listdir(input_path):
	if count >= max_num:
		break
	print("Processing: " + image_path)
	img = cv2.imread(os.path.join(input_path, image_path))
	ending = str(count) + "_" + blur_type + "_" + ".png"
	this_output_path = os.path.join(output_path, ending)
	count += 1

	if blur_type == "avg":
		avging = cv2.blur(img,(p1, p2)) 
		cv2.imwrite(this_output_path, avging)
	elif blur_type == "bilateral":
		bilFilter = cv2.bilateralFilter(img, p1, p2, p3) 
		cv2.imwrite(this_output_path, bilFilter)

