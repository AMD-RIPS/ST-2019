## Glitchify

#### Examples

python3 glitchify.py -i input_folder -t discoloration

python3 glitchify.py  -i input_folder -o output_folder -ot video  -interval 60 -t discoloration -output_array True

python3 glitchify.py  -i input_folder -o output_folder -ot video  -interval 60 -t discoloration -output_array True -is_output_resized True -new_height 224 -new_width 256


#### Description:

This program takes in videos and/or images, and adds selected glitches to the selected regions in the frames. The outputs are images (in png format and numpy arrays). The types of glitches are listed below:

1. discoloration

Randomly changes the color intensity of any pixel whose intensity exceeds the threshold.

2. random_patch

Generates a random number (3~20) of patches of random colors and positions.

3. shape

Adds a random number (2~5) of polygonal shapes to the darkest region of the input frames.


4. shader

Adds a random number (1~3) of shades of random color to the input frames.

5. dotted_line

Adds a random number (15~35) of dotted lines of random color to the input frames.

6. radial_dotted_line

Adds a random number (30~60) of dotted lines of random color and radiation shape to the input frames.

7. square_patch

Adds a random number (2~15) of little square patches (pixelation) of random color to the input frames.

8. parallel_line

Adds a random number (60~100) of parallel lines to the input frames.

9. texture_popin

Randomly selects a region and then applies bilateral filter (blurring) on it.


10. regular_triangulation

Randomly selects a region and then divides the region into isosceles right triangles. The color of each triangle is determined by the weighted average of pixels within the triangle.

11. morse_code

Adds morse code pattern of random color and position to the input frames.


12. desktop_glitch_one

Randomly selects a region and adds the first kind of desktop glitch (see examples) to the input frames.


13. desktop_glitch_two

Adds the second kind of desktop glitch (see examples) of random colors and positions the input frames.

14. screen_tearing

Combines two frames to form a new frame with vertical or horizontal screen-tearing. The interval between two frames is determined by the "interval" input value.

15. stuttering

Permutes rows and columns to produce stuttering frames.

16. line_pixelation

Adds random number of pixelated lines to random columns or rows in the input frames.


#### Input Arguments:

-o: name of output folder\
-i: name of input folder. The folder can contain images or videos or both.

-t: type of glitches.  Choose from the list: \
[discoloration, random_patch, shape, shader, dotted_line, radial_dotted_line, square_patch, parallel_line, texture_popin, regular_triangulation, morse_code, desktop_glitch_one, desktop_glitch_two, screen_tearing, stuttering, line_pixelation]


The inputs below are optional:
-interval: the number of frames skipped till the next glitch is added. Default value is 10.

-is_output_resized: True or False.\
-new_width: width of the resized output\
-new_height: height of the resized output

-output_array: whether outputs np arrays that corresponds to glitched images. Either True or False. Default value is False. 

If set True, then the program will store two arrays X_orig.npy and X_glitched.npy in output_folder/np_array. X_gliched.npy contains glitched images, and X_orig.npy contains the corresponding non-glitched images (i.e. images before the glitches are added).






