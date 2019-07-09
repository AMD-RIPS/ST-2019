#### Description:

This program adds checkerboard patterns to input images


#### Usage: 

python checkerboard_pattern.py input_path glitch_width_ratio glitch_height_ratio is_pattern_intermittent is_random
is_pattern_square max_num

glitch_width_ratio: width of a single glitch divided by width of image

glitch_height_ratio: height of a single glitch divided by height of image

is_pattern_intermittent: is the glitch intermittent (0 for solid glitch, 1 for intermittent glitch. See demo for examples)

is_pattern_square: is the glitch square

is_random: is glitch randomly generated

max_num: maximum number of images to be generated

#### Examples

python checkerboard_pattern.py input_examples output_examples 0.03 0.03 1 1 0 100



