#### Description:

This program blurs input images.


#### Usage: 

python screen_tearing_generator.py input_path output_path blur_type parameters max_num

blur_type:
	1. avg: blurs an image using the normalized box filter. Parameters are filter height and width
	2. bilateral: applies the bilateral filter to an image, which keeps the edges fairly sharp. Parameters are filter_size, sigma_color, sigma_size. Large filters (d > 5) are slow. If Sigma Values are small (< 10), the filter will not have much effect, whereas if they are large (> 150), they will have a very strong effect.
	
max_num: maximum number of images to be generated

#### Examples

python blur.py input_examples output_examples bilateral 10 80 80 100
python blur.py input_examples output_examples avg 5 5 100



