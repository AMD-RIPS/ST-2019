#### Description:

This program superimposes nearby frames in a video to produce images with sceen-tearing defect.


#### Usage: 

python screen_tearing_generator.py input_video_path output_folder frame_interval max_num glitch_type

input_video_path: path for the video from which glitched images are generated
output_folder: path for output images
frame_interval: how many frames are skipped till the next source frame
max_num: maximum number of images to be generated
glitch_type: "horizontal" or "vertical"

#### Examples

python screen_tearing_generator.py short_sample.mp4 output_examples 30 100 vertical



