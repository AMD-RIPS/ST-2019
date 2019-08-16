#### Description:

This program extracts frames from input video file. The input can either be an individual video file, or a folder containing video files. The program can detect whether the input is a file or a folder, so the input type need not to be specified.


#### Usage: 

python extract_images.py input_video_path output_folder_path output_type interval

input_video_path: path of individual video file, OR path of folder containing video files.

output_folder_path: path of the output folder (the program will create the folder if it does not exist)

output_type: jpg or png

interval: number of frames skipped between two consecutive samples



#### Examples

python extract_images.py sample.avi output_folder png 60

python extract_images.py input_folder output_folder png 60