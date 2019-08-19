#### Examples

python3 glitchify.py -i input_folder -t screen_tearing

python3 glitchify.py  -i input_folder -o output_folder -ot video  -interval 60 -t discoloration -output_array True

python3 glitchify.py  -i input_folder -o output_folder -ot video  -interval 60 -t discoloration -output_array True -is_output_resized True new_height 224 new_width 256


#### Description:

This program takes in videos and/or images, and adds selected glitches to the selected regions in the frames. The outputs are images (in png format and numpy arrays) or videos (in avi format). The types of glitches are listed below:

1. discoloration

Randomly changes the color intensity of any pixel whose intensity exceeds the threshold.

2. random_patch

Generates a random number (3~20) of patches of random colors and positions.

3. shape

Adds a random number (2~5) of polygonal shapes to the darkest region of the input frames.

4. triangle

Adds a random number (1~3) of overlaid triangular shapes of random color to the input frames.

5. shader

Adds a random number (1~3) of shades of random color to the input frames.

6. dotted_line

Adds a random number (15~35) of dotted lines of random color to the input frames.

7. radial_dotted_line

Adds a random number (30~60) of dotted lines of random color and radiation shape to the input frames.

8. square_patch

Adds a random number (2~15) of little square patches (pixelation) of random color to the input frames.

9. parallel_line

Adds a random number (60~100) of parallel lines to the input frames.

10. texture_popin

Randomly selects a region and then applies bilateral filter (blurring) on it.


12. regular_triangulation

Randomly selects a region and then divides the region into isosceles right triangles. The color of each triangle is determined by the weighted average of pixels within the triangle.

14. morse_code

Adds morse code pattern of random color and position to the input frames.


15. desktop_glitch_one

Randomly selects a region and adds the first kind of desktop glitch (see examples) to the input frames.


16. desktop_glitch_two

Adds the second kind of desktop glitch (see examples) of random colors and positions the input frames.

17. screen_tearing

Combines two frames to form a new frame with vertical or horizontal screen-tearing. The interval between two frames is determined by the "interval" input value.

18. stuttering

Permutes rows and columns to produce stuttering frames.

19. line_pixelation

Adds random line pixelations to the input frames.


#### Input Arguments:

-o: name of output folder\
-i: name of input folder. The folder can contain images or videos or both.

-t: type of glitches.  Choose from the list: \
[discoloration, random_patch, shape, triangle, shader, dotted_line, radial_dotted_line, square_patch, parallel_line, texture_popin, random_triangulation, regular_triangulation, morse_code, desktop_glitch_one, desktop_glitch_two, screen_tearing, stuttering, line_pixelation]


The inputs below are optional:

-ot: output type, either video or image. Default is image.\
-interval: the number of frames skipped till the next glitch is added. Default value is 10.

-is_output_resized: True or False.\
-new_width: width of the resized output\
-new_height: height of the resized output

-lo: lower bound of number of glitches\
-hi: upper bound of number of glitches

(x0,y0) and (x1,y1) defines the sub-region where the selected transformation takes place.\
-x0: x-coordinate of the top left corner\
-y0: y-coordinate of the top left corner\
-x1: x-coordinate of the bottom right corner\
-y1: y-coordinate of the bottom right corner

-output_array: whether outputs np arrays that corresponds to glitched images. Either True or False. Default value is False. 

If set True, then the program will store two arrays X_orig.npy and X_glitched.npy in output_folder/np_array. X_gliched.npy contains glitched images, and X_orig.npy contains the corresponding non-glitched images (i.e. images before the glitches are added).






