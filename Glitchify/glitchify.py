import matplotlib.pyplot as plt
import cv2
import os
from functools import partial
import numpy as np
import numpy.random as npr
import sys
import argparse
from desktop_glitch.desktop_glitch_one import *
from desktop_glitch.desktop_glitch_two import create_desktop_glitch_two
import random
import imutils
import ou_glitch.ou_glitch as og
from stuttering.stuttering import produce_stuttering
from line_pixelation.line_pixelation import line_pixelation


def get_random_color():
	b_int = npr.randint(0,256)
	g_int = npr.randint(0,256)
	r_int = npr.randint(0,256)
	return b_int, g_int, r_int

def add_vertical_pattern(img):
	color = np.random.randint(0,256,size = 3)
	(height, width, channel) = img.shape
	pattern_dist = int(width * 0.01)
	pattern_length = int(height * 0.04)
	segment_length = pattern_length // 8

	row_count = 0
	start_row_index = 0
	horizontal_shift = random.randint(0, pattern_dist)
	for y in range(horizontal_shift, width, pattern_dist):
		if row_count % 4 == 0:
			vertical_shift = random.randint(0, pattern_length)

		if np.random.uniform() < 0.75:
			row_count += 1
			continue
		for x in range(0, height, pattern_length):
			if np.random.uniform() < 0.4:
				continue
			img[(vertical_shift + x) % height, y, :] = color
			img[(vertical_shift + x + segment_length) % height, y, :] = color
			img[((vertical_shift + x + 2 * segment_length)% height):((vertical_shift + x + 4 * segment_length)% height), y, :] = color
			img[(vertical_shift + x + 5 * segment_length)% height, y, :] = color
			img[(vertical_shift + x + 6 * segment_length)% height, y, :] = color
		row_count += 1
	return img

def blurring(img):
	cp = np.copy(img)
	cp2 = np.copy(img)
	blur = cv2.bilateralFilter(img, 40, 100, 100)
	return blur

def create_discoloration(img):
	threshold = npr.randint(100, 140)
	new_intesity = npr.randint(200, 256)

	color = npr.randint(0, 6)
	if color == 0:
		img[img[:,:,2] > threshold] = [0,0,new_intesity]
	elif color == 1:
		img[img[:,:,1] > threshold] = [0,new_intesity,0]
	elif color == 2:
		img[img[:,:,0] > threshold] = [new_intesity,0,0]
	else:
		b_int = npr.randint(new_intesity,256)
		g_int = npr.randint(new_intesity,256)
		r_int = npr.randint(new_intesity,256)
		img[img[:,:,0] > threshold] = [b_int, g_int, r_int]

	return img

def triangulation(img):
	h,w,_ = img.shape
	grid_length = int(np.random.uniform(1.0 / 40, 1.0 / 25) * w)
	half_grid = grid_length // 2

	triangles = []

	for i in range(0,h,grid_length):
		for j in range(0,w,grid_length):
			pt1, pt2 = np.array([i,j]), np.array([i,min(j+ grid_length, w-1)])
			pt3 = np.array([min(i+half_grid, h-1),min(j+half_grid, w-1)])
			pt4, pt5 = np.array([min(i+grid_length,  h-1),j]), np.array([min(i+grid_length, h-1),min(j+grid_length, w-1)])


			pt1 = pt1[[1,0]]
			pt2 = pt2[[1,0]]
			pt3 = pt3[[1,0]]
			pt4 = pt4[[1,0]]
			pt5 = pt5[[1,0]]

			triangles.append(np.array([pt1,pt2,pt3]))
			triangles.append(np.array([pt1,pt4,pt3]))
			triangles.append(np.array([pt5,pt2,pt3]))
			triangles.append(np.array([pt5,pt4,pt3]))


	for t in triangles:
		mid_pt = ((t[0] + t[1] + t[2])/3).astype(int)

		mid_pt = mid_pt[[1,0]]

		color = img[mid_pt[0], mid_pt[1],:]*0.85 + 0.05 * img[t[0,1], t[0,0], :] + 0.05 * img[t[1,1], t[1,0], :] + 0.05 * img[t[2,1], t[2,0], :] 
		color = np.uint8(color)
		c = tuple(map(int, color))

		p = cv2.drawContours(img, [t], -1, c, -1)

	return p

def add_random_patches(im, lo = 3, hi = 20):
	color = npr.randint(0, 6)
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	contours.sort(key = len)
	patch_number = np.random.randint(lo,hi+1)
	b_int, g_int, r_int = get_random_color()
	for i in range(patch_number):
		if color == 0:
			cv2.drawContours(im, contours,len(contours) - 1 - i , (0,0,250), -1)
		elif color == 1:
			cv2.drawContours(im, contours,len(contours) - 1 - i , (0,250,0), -1)
		elif color == 2:
			cv2.drawContours(im, contours,len(contours) - 1 - i , (250,0,0), -1)
		else:
			cv2.drawContours(im, contours,len(contours) - 1 - i , (b_int,g_int,r_int), -1)

	return im


def add_shapes(im, lo = 2, hi = 5):
	h, w, _ = im.shape
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# Find the darkest region of the image
	grid = (-1,-1)
	mean_shade = np.mean(im)
	x_step, y_step = int(w/6), int(h/4)
	for x in range(0, w, x_step):
		for y in range(0, h, y_step):
			new_shade = np.mean(im[x:x+x_step, y:y+y_step])
			if  new_shade <= mean_shade:
				mean_shade = new_shade
				grid = (x,y)

	# Add shapes
	minLoc = (np.random.randint(grid[0], min(grid[0]+x_step, w)), np.random.randint(grid[1], min(grid[1]+x_step, h)))
	num_shapes = np.random.randint(lo,hi+1)
	for i in range(num_shapes):
		stretch = np.random.randint(40, 100)
		diff1, diff2 = np.random.randint(-5,5), np.random.randint(-5,5)
		x1, y1 = minLoc[0] +  diff1* stretch  , minLoc[1] + diff2 * stretch
		x2, y2 = x1 + np.random.randint(1,12)/5 * diff1 * stretch  , y1 + np.random.randint(1,12)/5 * diff2* stretch
		pts = np.array((minLoc, (x1, y1), (x2, y2)), dtype=int)

		c1, c2, c3 = np.random.randint(0,50),np.random.randint(0,50),np.random.randint(0,50)
		cv2.fillConvexPoly(im, pts, color= (c1,c2,c3))

	return im


def add_triangles(im, lo = 1, hi = 3):
	h, w, _ = im.shape
	colors = np.array((
                   (250,206,135),
                   (153,255, 255),
                   (255, 203, 76)),dtype = int) #maybe expand this list of colors

	output = im.copy()
	overlay = im.copy()

	x_0, y_0 = np.random.randint(w), np.random.randint(h)
	x_1, y_1 = np.random.randint(w), np.random.randint(h)
	x_2, y_2 = np.random.randint(w), np.random.randint(h)
	pts = np.array(((x_0, y_0), (x_1, y_1), (x_2, y_2)), dtype=int)

	b_int, g_int, r_int = get_random_color()
	cv2.fillConvexPoly(overlay, pts, color= tuple([b_int, g_int, r_int]) )

	num_shapes = np.random.randint(lo, hi + 1)
	alpha = .95
	for i in range(num_shapes):
		x_1, y_1 = np.mean([x_1, x_0]) + np.random.randint(-60,60), np.mean([y_1,y_0])+ np.random.randint(-60,60)
		x_2, y_2 = np.mean([x_2, x_0]) + np.random.randint(-60,60), np.mean([y_2,y_0])+ np.random.randint(-60,60)
		
		pts = np.array(((x_0, y_0), (x_1, y_1), (x_2, y_2)), dtype=int) 
		# if not is_random:
		# 	cv2.fillConvexPoly(overlay, pts, color= tuple([int(x) for x in colors[np.random.randint(3)]]) )

		b_int, g_int, r_int = get_random_color()
		cv2.fillConvexPoly(overlay, pts, color= tuple([b_int, g_int, r_int]) )

	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

	return output



#this function returns a color blend of the overlay and the original image. angle = 0 means the overlay
#will fade down and angle = 180 will cause fade up:
##################################
def gradient(img, overlay, angle = 0):
	alpha = 1


	img = imutils.rotate_bound(img, angle)
	overlay = imutils.rotate_bound(overlay, angle)


	for x in range(1, img.shape[0],10 ):
		cv2.addWeighted(overlay[x:x+10,:,:], alpha, img[x:x+10,:,:] , 1 - alpha, 0, img[x:x+10,:,:])
		alpha *= .98


	img = imutils.rotate_bound(img, -1 * angle)
	return img

#########
def color_blend(img, overlay1, overlay2, angle = 0):
	alpha = 1

	img = imutils.rotate_bound(img, angle)
	overlay1 = imutils.rotate_bound(overlay1, angle)
	overlay2 = imutils.rotate_bound(overlay2, angle)

	for x in range(1, overlay1.shape[0],10 ):
		cv2.addWeighted(overlay1[x:x+10,:,:], alpha, overlay2[x:x+10,:,:] , 1 - alpha, 0, img[x:x+10,:,:])
		alpha *= .95

	img = imutils.rotate_bound(img, -1 * angle)

	return img
#############################


def add_shaders(im, lo = 1, hi = 3):
	angles = np.array([0,90,180,270])

	h,w,_ = im.shape
	output = im.copy()
	overlay1 = im.copy()
	overlay2 = im.copy()

	#####big shaders in forms of n-gons
	num_shapes = np.random.randint(lo,hi+1)
	for i in range(num_shapes):
		x_0, y_0 = np.random.randint(w), np.random.randint(h)
		x_1, y_1 = np.random.randint(-300,w+300), np.random.randint(-300,h+300)
		x_2, y_2 = np.random.randint(-300,w+300), np.random.randint(-300,h+300)

		pts = np.array(((x_0, y_0), (x_1, y_1), (x_2, y_2)), dtype=int)

		extra_n = np.random.randint(4)

		for i in range(extra_n): #extra number of points to make an n_gon
			pts = np.append(pts, [[np.random.randint(-300,h+300), np.random.randint(-300,w+300)]], axis = 0)

		alpha = 1

		colors = np.empty([2, 3])
		start_x = min(max(0, x_0), h-1)
		start_y = min(max(0, y_0), w-1)

		colors[0, :] = im[start_x, start_y,:] + npr.randint(-30, 30, size = [3])
		mid_x = (x_1+x_2)//2
		mid_y = (y_1+y_2)//2

		mid_x = min(max(0, mid_x), h-1)
		mid_y = min(max(0, mid_y), w-1)

		colors[1, :] = im[mid_x,mid_y,:] + npr.randint(-30, 30, size = [3])

		colors = np.clip(colors, a_min = 0, a_max = 255) 

		# colors[0,:] = npr.randint(0, 256, size = 3)
		# colors[1,:] = colors[0,:] + npr.randint(0, 100, size = 3)
		# colors[1,:] = np.clip(colors[1,:], 0, 255)

		
		cv2.fillConvexPoly(overlay1, pts, color= tuple([int(x) for x in colors[0]]) )
		cv2.fillConvexPoly(overlay2, pts, color= tuple([int(x) for x in colors[1]]) )


	############
	a1, a2 = random.choice(angles), random.choice(angles)

	return gradient(output, color_blend(im, overlay1, overlay2, a1), a2)


def write_files(original_img, img, is_margin_specified, filename, out, is_video, append_to_arr):
	if append_to_arr:
		if not is_output_resized:
			X_orig_list.append(original_img)
		else:
			original_img = cv2.resize(original_img, (new_width, new_height))
			X_orig_list.append(original_img)

	if is_margin_specified:
		original_img[x0:x1, y0:y1, :] = img
	else:
		original_img = img

	if not is_video:
		if not is_output_resized:
			cv2.imwrite(filename, original_img)
		else:
			original_img = cv2.resize(original_img ,(new_width, new_height))
			cv2.imwrite(filename, original_img)
	else:
		if not is_output_resized:
			out.write(original_img)
		else:
			original_img = cv2.resize(original_img, (new_width, new_height))
			out.write(original_img)


	if append_to_arr:
		if not is_output_resized:
			X_glitched_list.append(original_img)
		else:
			original_img = cv2.resize(original_img ,(new_width, new_height))
			X_glitched_list.append(original_img)



def is_video_file(filename):
	video_file_extensions = (
	'.264', '.3g2', '.3gp', '.3gp2', '.3gpp', '.3gpp2', '.3mm', '.3p2', '.60d', '.787', '.89', '.aaf', '.aec', '.aep', '.aepx',
	'.aet', '.aetx', '.ajp', '.ale', '.am', '.amc', '.amv', '.amx', '.anim', '.aqt', '.arcut', '.arf', '.asf', '.asx', '.avb',
	'.avc', '.avd', '.avi', '.avp', '.avs', '.avs', '.avv', '.axm', '.bdm', '.bdmv', '.bdt2', '.bdt3', '.bik', '.bin', '.bix',
	'.bmk', '.bnp', '.box', '.bs4', '.bsf', '.bvr', '.byu', '.camproj', '.camrec', '.camv', '.ced', '.cel', '.cine', '.cip',
	'.clpi', '.cmmp', '.cmmtpl', '.cmproj', '.cmrec', '.cpi', '.cst', '.cvc', '.cx3', '.d2v', '.d3v', '.dat', '.dav', '.dce',
	'.dck', '.dcr', '.dcr', '.ddat', '.dif', '.dir', '.divx', '.dlx', '.dmb', '.dmsd', '.dmsd3d', '.dmsm', '.dmsm3d', '.dmss',
	'.dmx', '.dnc', '.dpa', '.dpg', '.dream', '.dsy', '.dv', '.dv-avi', '.dv4', '.dvdmedia', '.dvr', '.dvr-ms', '.dvx', '.dxr',
	'.dzm', '.dzp', '.dzt', '.edl', '.evo', '.eye', '.ezt', '.f4p', '.f4v', '.fbr', '.fbr', '.fbz', '.fcp', '.fcproject',
	'.ffd', '.flc', '.flh', '.fli', '.flv', '.flx', '.gfp', '.gl', '.gom', '.grasp', '.gts', '.gvi', '.gvp', '.h264', '.hdmov',
	'.hkm', '.ifo', '.imovieproj', '.imovieproject', '.ircp', '.irf', '.ism', '.ismc', '.ismv', '.iva', '.ivf', '.ivr', '.ivs',
	'.izz', '.izzy', '.jss', '.jts', '.jtv', '.k3g', '.kmv', '.ktn', '.lrec', '.lsf', '.lsx', '.m15', '.m1pg', '.m1v', '.m21',
	'.m21', '.m2a', '.m2p', '.m2t', '.m2ts', '.m2v', '.m4e', '.m4u', '.m4v', '.m75', '.mani', '.meta', '.mgv', '.mj2', '.mjp',
	'.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd', '.moff', '.moi', '.moov', '.mov', '.movie', '.mp21',
	'.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1', '.mpeg4', '.mpf', '.mpg', '.mpg2', '.mpgindex', '.mpl',
	'.mpl', '.mpls', '.mpsub', '.mpv', '.mpv2', '.mqv', '.msdvd', '.mse', '.msh', '.mswmm', '.mts', '.mtv', '.mvb', '.mvc',
	'.mvd', '.mve', '.mvex', '.mvp', '.mvp', '.mvy', '.mxf', '.mxv', '.mys', '.ncor', '.nsv', '.nut', '.nuv', '.nvc', '.ogm',
	'.ogv', '.ogx', '.osp', '.otrkey', '.pac', '.par', '.pds', '.pgi', '.photoshow', '.piv', '.pjs', '.playlist', '.plproj',
	'.pmf', '.pmv', '.pns', '.ppj', '.prel', '.pro', '.prproj', '.prtl', '.psb', '.psh', '.pssd', '.pva', '.pvr', '.pxv',
	'.qt', '.qtch', '.qtindex', '.qtl', '.qtm', '.qtz', '.r3d', '.rcd', '.rcproject', '.rdb', '.rec', '.rm', '.rmd', '.rmd',
	'.rmp', '.rms', '.rmv', '.rmvb', '.roq', '.rp', '.rsx', '.rts', '.rts', '.rum', '.rv', '.rvid', '.rvl', '.sbk', '.sbt',
	'.scc', '.scm', '.scm', '.scn', '.screenflow', '.sec', '.sedprj', '.seq', '.sfd', '.sfvidcap', '.siv', '.smi', '.smi',
	'.smil', '.smk', '.sml', '.smv', '.spl', '.sqz', '.srt', '.ssf', '.ssm', '.stl', '.str', '.stx', '.svi', '.swf', '.swi',
	'.swt', '.tda3mt', '.tdx', '.thp', '.tivo', '.tix', '.tod', '.tp', '.tp0', '.tpd', '.tpr', '.trp', '.ts', '.tsp', '.ttxt',
	'.tvs', '.usf', '.usm', '.vc1', '.vcpf', '.vcr', '.vcv', '.vdo', '.vdr', '.vdx', '.veg','.vem', '.vep', '.vf', '.vft',
	'.vfw', '.vfz', '.vgz', '.vid', '.video', '.viewlet', '.viv', '.vivo', '.vlab', '.vob', '.vp3', '.vp6', '.vp7', '.vpj',
	'.vro', '.vs4', '.vse', '.vsp', '.w32', '.wcp', '.webm', '.wlmp', '.wm', '.wmd', '.wmmp', '.wmv', '.wmx', '.wot', '.wp3',
	'.wpl', '.wtv', '.wve', '.wvx', '.xej', '.xel', '.xesc', '.xfl', '.xlmv', '.xmv', '.xvid', '.y4m', '.yog', '.yuv', '.zeg',
	'.zm1', '.zm2', '.zm3', '.zmv'  )

	if filename.endswith((video_file_extensions)):
		return True
	else:
		return False



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	global X_orig_list, X_glitched_list
	X_orig_list, X_glitched_list = [], []


	# Values
	parser.add_argument('-o', '--output', dest='output_foldername')
	parser.add_argument('-i', '--input', dest='input_foldername')
	parser.add_argument('-t', '--type', dest='glitch_type')
	parser.add_argument('-lo', dest='arg1')
	parser.add_argument('-hi', dest='arg2')

	parser.add_argument('-x0', dest = 'x0')
	parser.add_argument('-y0', dest = 'y0')
	parser.add_argument('-x1', dest = 'x1')
	parser.add_argument('-y1', dest = 'y1')
	parser.add_argument('-interval', dest = 'interval', default=10)

	parser.add_argument('-ot', '--output_type', dest = 'output_type', default= 'image')
	parser.add_argument('-save_normal_frames', dest= 'save_normal_frames', default = 'False')
	parser.add_argument('-output_array', dest = 'output_array', default = 'False')
	parser.add_argument('-is_output_resized', dest = 'resize_output', default = 'False')
	parser.add_argument('-new_height', dest = 'new_height', default = 224)
	parser.add_argument('-new_width', dest = 'new_width', default = 224)


	options = parser.parse_args()
	global arg1, arg2, x0, y0, x1, y1, is_output_resized, new_height, new_width

	is_bound_specified = False
	is_margin_specified = False
	is_video = False
	output_array = False
	is_output_resized = False
	interval = int(options.interval)
	new_height = 224
	new_width = 224

	if options.resize_output == 'True' or options.resize_output == 'true':
		is_output_resized = True
		new_height = int(options.new_height)
		new_width = int(options.new_width)

	if options.output_type == 'video' or options.output_type == 'Video':
		is_video = True

	if options.arg1 is not None and options.arg2 is not None:
		is_bound_specified = True
		arg1 = int(options.arg1)
		arg2 = int(options.arg2)


	if options.x0 is not None and options.y0 is not None and options.x1 is not None and options.y1 is not None:
		is_margin_specified = True
		x0 = int(options.x0)
		y0 = int(options.y0)
		x1 = int(options.x1)
		y1 = int(options.y1)


	if options.output_array == 'True' or options.output_array == 'true':
		output_array = True

	count = 0

	if options.input_foldername is None:
		print("Please specify input and output folder name")

	if options.output_foldername is None:
		options.output_foldername = "new_output_folder"

	if not os.path.isdir(options.output_foldername):
		os.mkdir(options.output_foldername)


	for video_path in os.listdir(options.input_foldername):
		is_image = False

		if not is_video_file(video_path):
			if video_path.endswith('.jpg') or video_path.endswith('.png'):
				is_image = True
			else:
				continue

		if is_image and is_video:
			print("Input imgaes are skipped when producing glitched videos")
			continue

		cap = None
		frame_width = 0
		frame_height = 0 
		out = None

		if not is_image:
			cap = cv2.VideoCapture(os.path.join(options.input_foldername, video_path))
			frame_width = int(cap.get(3))
			frame_height = int(cap.get(4))

			if is_output_resized:
				frame_width = new_width
				frame_height = new_height

		save_normal_frames = False

		if is_video:
			if options.glitch_type is None:
				out = cv2.VideoWriter(os.path.join(options.output_foldername,str(count) + '_normal_video.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), 60, (frame_width,frame_height))
			else:
				out = cv2.VideoWriter(os.path.join(options.output_foldername,str(count) + '_' + str(options.glitch_type) +'_video.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), 60, (frame_width,frame_height))

		if options.save_normal_frames == 'True' or options.save_normal_frames == 'true' or is_video:
			save_normal_frames  = True

		if not is_video and save_normal_frames and not os.path.isdir(os.path.join(options.output_foldername, 'normal')):
			os.mkdir(os.path.join(options.output_foldername, 'normal'))

		if output_array and not os.path.isdir(os.path.join(options.output_foldername, 'np_array')):
			os.mkdir(os.path.join(options.output_foldername, 'np_array'))

		this_count = 0
		global prev_img
		while(True):
			ret  =  False
			original_img = None

			if is_image:
				ret = True
				original_img = cv2.imread(os.path.join(options.input_foldername, video_path))
			else:
				ret, original_img = cap.read()
				if not ret:
					break

			img = np.copy(original_img)

			if is_margin_specified:
				img = original_img[x0:x1, y0:y1, :]

			if this_count % interval != 0:
				this_count += 1
				if save_normal_frames:
					output_name = str(count) + "_normal.png"
					if is_video:
						output_name = str(count)  + '_' + str(this_count)+ "_normal.png"
					output_filename = os.path.join(options.output_foldername, 'normal')
					output_filename = os.path.join(output_filename, output_name)
					# print(output_filename)
					new_img = img
					write_files(original_img, new_img, is_margin_specified, output_filename, out, is_video, False)

					if not is_video:
						count += 1
				continue

			if this_count == 0:
				prev_img = img

			if options.glitch_type is None:
				output_name = str(count) + "_normal.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				# X_orig_list.append()
				if not is_video:
					count += 1

			if options.glitch_type == 'screen_tearing':
				if is_image:
					print("Single input image is skipped when producing screen tearing glitches")
					break

				if this_count == 0:
					this_count += 1
					if save_normal_frames:
						output_name = str(count) + "_normal.png"
						if is_video:
							output_name = str(count)  + '_' + str(this_count)+ "_normal.png"
						output_filename = os.path.join(options.output_foldername, 'normal')
						output_filename = os.path.join(output_filename, output_name)
						new_img = img
						write_files(original_img, new_img, is_margin_specified, output_filename, out, is_video, False)
					continue
				
				height, width, channels = img.shape
				r = np.random.rand(1) * 0.8 + 0.1

				new_img = np.copy(img)
				if np.random.uniform() < 0.6:
					target_height = np.rint(r * height)
					target_height = target_height.astype(int)
					diff = (img[0:target_height[0], :, :] - prev_img[0:target_height[0], :, :]) / 255
					area = width * target_height[0]

					ssq = np.sum(diff**2) / area
					if ssq < 0.4:
						this_count += 1
						continue

					new_img[0:target_height[0], :, :] = prev_img[0:target_height[0], :, :]
				else:
					target_width = np.rint(r * width)
					target_width = target_width.astype(int)
					diff = (img[:, 0:target_width[0], :] - prev_img[:, 0:target_width[0], :]) / 255
					area = height * target_width[0]

					ssq = np.sum(diff**2) / area
					if ssq < 0.4:
						this_count += 1
						continue

					new_img[:, 0:target_width[0], :] = prev_img[:, 0:target_width[0], :]

				prev_img = img
				output_name = str(count) + "_screen_tearing.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, new_img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == "desktop_glitch_one":
				# print(img.shape)
				new_img = create_desktop_glitch_one(img)

				output_name = str(count) + "_desktop_glitch_one.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, new_img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == "desktop_glitch_two":
				img = create_desktop_glitch_two(img)

				output_name = str(count) + "_desktop_glitch_two.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == "discoloration":
				# print(img.shape)
				img = create_discoloration(img)

				output_name = str(count) + "_discoloration.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				# cv2.imwrite(output_filename, img)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == "random_patch":
				if is_bound_specified:
					img = add_random_patches(img, arg1, arg2)
				else:
					img = add_random_patches(img)

				output_name = str(count) + "_random_patch.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'shape':
				if is_bound_specified:
					img = add_shapes(img, arg1, arg2)
				else:
					img = add_shapes(img)

				output_name = str(count) + "_shape.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'triangle':
				if is_bound_specified:
					img = add_triangles(img, arg1, arg2)
				else:
					img = add_triangles(img)

				output_name = str(count) + "_triangle.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1


			if options.glitch_type == 'shader':
				if is_bound_specified:
					img = add_shaders(img, arg1, arg2)
				else:
					img = add_shaders(img)

				output_name = str(count) + "_shader.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'dotted_line':
				if is_bound_specified:
					img = og.dotted_lines(img, arg1, arg2)
				else:
					img = og.dotted_lines(img)

				output_name = str(count) + "_dotted_line.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'radial_dotted_line':
				if is_bound_specified:
					img = og.dotted_lines_radial(img, arg1, arg2)
				else:
					img = og.dotted_lines_radial(img)

				output_name = str(count) + "_radial_dotted_line.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'parallel_line':
				if is_bound_specified:
					img = og.parallel_lines(img, arg1, arg2)
				else:
					img = og.parallel_lines(img)

				output_name = str(count) + "_parallel_line.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1


			if options.glitch_type == 'square_patch':
				if is_bound_specified:
					img = og.square_patches(img, arg1, arg2)
				else:
					img = og.square_patches(img)

				output_name = str(count) + "_square_patch.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1


			if options.glitch_type == 'texture_popin':
				img = blurring(img)

				output_name = str(count) + "_texture_popin.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'random_triangulation':
				print("Random Triangulation is removed from the list of glitches")


			if options.glitch_type == 'regular_triangulation':
				img = triangulation(img, False)

				output_name = str(count) + "_regular_triangulation.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'morse_code':
				img = add_vertical_pattern(img)

				output_name = str(count) + "_morse_code.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'stuttering':
				img = produce_stuttering(img)

				output_name = str(count) + "_stuttering.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1

			if options.glitch_type == 'line_pixelation':
				img = line_pixelation(img)

				output_name = str(count) + "_line_pixelation.png"
				output_filename = os.path.join(options.output_foldername, output_name)
				write_files(original_img, img, is_margin_specified, output_filename, out, is_video, True)
				if not is_video:
					count += 1
		
			this_count += 1

			if is_image:
				break

		if is_video:
			count += 1
		if cap is not None:
			cap.release()
		if out is not None:
			out.release()


	if output_array:
		output_folder = os.path.join(options.output_foldername, 'np_array')
		X_orig = np.asarray(X_orig_list)
		X_glitched = np.asarray(X_glitched_list)

		print("Numpy arrays are saved in " +  output_folder)
		# print("Number of dimensions of saved arrays are :" + str(X_orig.ndim) + ", and " + str(X_glitched.ndim))
		if X_orig.ndim ==  1 or X_glitched.ndim == 1:
			print("Either the input is empty, or the input images and/or videos frames are not of the same dimension. Consider resizing the outputs.")
		# print(X_glitched.ndim)

		np.save(os.path.join(output_folder, 'X_orig.npy'), X_orig)
		np.save(os.path.join(output_folder, 'X_glitched.npy'), X_glitched)





