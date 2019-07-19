import cv2
import sys
import os

input_video_path = sys.argv[1]
output_image_path = sys.argv[2]
output_type = sys.argv[3]
interval = int(sys.argv[4])
# max_num = int(sys.argv[5])

cap = cv2.VideoCapture(input_video_path)

count = 0
num_produced = 0

if not os.path.isdir(output_image_path):
	os.mkdir(output_image_path)

while(1):
    ret,frame = cap.read()

    if not ret:
        break

    if count % interval == 0:
        output_name = os.path.join(output_image_path, str(count) + '.' + output_type)
        cv2.imwrite(output_name, frame)
        num_produced += 1

    count += 1
cap.release()
print("Number of images extracted: " + str(num_produced))

print("Rename the output folder or choose a different output folder before running the program again (to prevent overwriting exsiting images)")