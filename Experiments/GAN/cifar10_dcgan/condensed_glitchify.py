import numpy as np
import cv2
import numpy.random as npr
import argparse, random, imutils
import random
import math
from PIL import Image


def divisors(n):
	divisors=[]
	for i in list(range(1,int(n/2)+1)):
		if (n%i==0):
			divisors.append(i)
	divisors.append(n)
	return divisors

def swap(image,pix1,pix2):
	dummyr=image[pix1[0],pix1[1],0]
	dummyg=image[pix1[0],pix1[1],1]
	dummyb=image[pix1[0],pix1[1],2]
	image[pix1[0],pix1[1],0]=image[pix2[0],pix2[1],0]
	image[pix1[0],pix1[1],1]=image[pix2[0],pix2[1],1]
	image[pix1[0],pix1[1],2]=image[pix2[0],pix2[1],2]
	image[pix2[0],pix2[1],0]=dummyr
	image[pix2[0],pix2[1],1]=dummyg
	image[pix2[0],pix2[1],2]=dummyb

def stutter(img,sizex,sizey):
	vstripes=int(img.shape[1]/sizex)
	hstripes=int(img.shape[0]/sizey)
	for k in list(range(0,img.shape[1],2*int(img.shape[1]/vstripes))):
		for i in list(range(int(img.shape[1]/vstripes))):
			for j in list(range(img.shape[0])):
				if(i+k+int(img.shape[1]/vstripes)<img.shape[1]):
					swap(img,[j,i+k],[j,int(img.shape[1]/vstripes)+i+k])
	for k in list(range(0,img.shape[0],2*int(img.shape[0]/hstripes))):
		for i in list(range(int(img.shape[0]/hstripes))):
			for j in list(range(img.shape[1])):
				if(i+k+int(img.shape[0]/hstripes)<img.shape[0]):
					swap(img,[i+k,j],[i+k+int(img.shape[0]/hstripes),j])
	return img


def produce_stuttering(img):
	vdivisors=divisors(img.shape[0])
	hdivisors=divisors(img.shape[1])
	iterations=np.random.choice(np.arange(1,5),p=[0.65,0.2,0.1,0.05])
	for i in list(range(iterations)):
		sizex=random.choice(hdivisors)
		sizey=random.choice(vdivisors)
		img=stutter(img,sizex,sizey)
	return img

def check_val(value):
	if value > 255:
		value = 255
	if value < 0:
		value = 0
	return value

def dotted_lines(picture, lo = 15, hi = 35):
	pic = picture.copy()
	height = pic.shape[0]
	width = pic.shape[1]
	angle = random.randint(15,345)
	number_of_lines = random.randint(lo,hi+1)

	ox = random.randint(int(0.2*width), int(0.8*width))
	oy = random.randint(int(0.2*height),int(0.8*height))

	r = random.randint(0,255)
	g = random.randint(0,255)
	b = random.randint(0,255)

	for i in np.arange(number_of_lines):
		x = ox + random.randint(-int(0.2*width), int(0.2*width))
		y = oy + random.randint(-int(0.2*height),int(0.2*height))
		theta = angle + random.randint(-20,20)
		tangent = np.tan(theta/180*np.pi)
		hstep = random.choice([-4,4,-5, 5,-3,3,-6,6])
		vstep = hstep*tangent
		for j in np.arange(random.randint(20,50)):
			px = int(x + j*hstep)
			py = int(y + j*vstep)
			if px >= 0 and px <= width-1 and py >= 0 and py <= height-1:
				u = random.uniform(0, 1)
				if u > 0.9:
					nx = max(px-1, 0)
					pic[py,nx] = [r,g,b]
				if u < 0.1:
					ny = max(py-1, 0)
					pic[ny,px] = [r,g,b]
				pic[py,px] = [r,g,b]
			else:
				break   
	return pic   

def dotted_lines_radial(picture, lo = 30, hi = 60):
    pic = picture.copy()
    height = pic.shape[0]
    width = pic.shape[1]
   # angle = random.randint(15,345)
    number_of_lines = random.randint(lo,hi)

    x = random.randint(int(0.2*width), int(0.8*width))
    y = random.randint(int(0.2*height),int(0.8*height))

    r = np.random.randint(0,255)
    g = np.random.randint(0,255)
    b = np.random.randint(0,255)

    angle_step = np.floor(360 / number_of_lines)
    initial_angle = random.randint(-10,10)


    for i in np.arange(number_of_lines):
        theta = initial_angle + angle_step * i + random.randint(-5,5)
        radian = theta/180*np.pi
        if np.cos(radian) >= 0:
            hstep = random.choice([4,5,3,6])
        else:
            hstep = random.choice([4,5,3,6]) * -1
        vstep = hstep*np.tan(radian)
        for j in np.arange(random.randint(20,50)):
            px = int(x + j*hstep)
            py = int(y + j*vstep)
            if px >= 0 and px <= width-1 and py >= 0 and py <= height-1:
                u = random.uniform(0, 1)
                if u > 0.9:
                    nx = max(px-1, 0)
                    pic[py,nx] = [r,g,b]
                if u < 0.1:
                    ny = max(py-1, 0)
                    pic[ny,px] = [r,g,b]
                pic[py,px] = [r,g,b]
            else:
                break   
    return pic


def square_patches(picture, lo = 2, hi = 15):
	pic = picture.copy()
	height = pic.shape[0]
	width = pic.shape[1]
	number_of_patches = random.randint(lo,hi+1)

	first_y = -1
	first_x = -1

	r = int(random.uniform(0, 1)*255)
	g = int(random.uniform(0, 1)*255)
	b = int(random.uniform(0, 1)*255)

	for i in range(number_of_patches):
		size = random.randint(2,5)
		red = check_val(r + random.randint(-30,30))
		green = check_val(g + random.randint(-30,30))
		blue = check_val(b + random.randint(-30,30))
		color = [blue, green, red]
		if first_y < 0:
			first_y = random.randint(int(height*0.2), int(height*0.8))
			first_x = random.randint(int(width*0.2), int(width*0.8))
			pic[first_y:(first_y+size), first_x:(first_x+size)] = color
		else:
			y = first_y +  random.randint(-int(height*0.1), int(height*0.1))
			x = first_x +  random.randint(-int(width*0.1), int(width*0.1))
			pic[y:(y+size), x:(x+size)] = color

	return pic

def parallel_lines(picture, lo = 60, hi = 100):
	pic = picture.copy()
	height = pic.shape[0]
	width = pic.shape[1]
	number_of_lines = np.random.randint(lo,hi+1)
	theta = np.random.randint(10,35)
	angle = np.tan(theta/180*np.pi)
	u = np.random.uniform(0,1)
	sign = random.choice([1,-1])

	while number_of_lines > 0:
		x1 = random.randint(int(0.3*width),int(0.6*width))
		y1 = random.randint(int(0.2*height),int(0.8*height))
		if u < 0.5:
			x2 = 0
			y2 = y1 + sign*int(x1*angle)
			if y2 >= height or y2 < 0:
				continue
		else:
			x2 = width-1
			y2 = y1 + sign*int((width-x1-1)*angle)
			if y2 >= height or y2 < 0:
				continue     
		lineThickness = random.randint(1,3)
		colors = pic[y1,x1].astype(float)
		cv2.line(pic, (x1,y1), (x2,y2), colors, lineThickness)
		number_of_lines -= 1

	return pic 



def triangulation(img):
	h,w,_ = img.shape
	grid_length = 5
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

		color = img[mid_pt[0], mid_pt[1],:]*0.85 + 0.05 * img[t[0,1], t[0,0], :] 
		+ 0.05 * img[t[1,1], t[1,0], :] + 0.05 * img[t[2,1], t[2,0], :] 
		color = np.uint8(color)
		c = tuple(map(int, color))

		cv2.drawContours(img, [t], -1, c, -1)

		# plt.imshow(img)
		# plt.show()

	# print(p)

	return img


def blur(img):
	cp = np.copy(img)
	cp2 = np.copy(img)
	blur = cv2.bilateralFilter(img, 40, 100, 100)
	return blur


def tri_bunch(X):
	X_transformed = np.empty_like(X)
	for i in range(X.shape[0]):
		X_transformed[i,:,:,:] = blur(X[i,:,:,:])

	return X_transformed


def create_discoloration(img):
	threshold = npr.randint(100, 140)
	new_intesity = npr.randint(200, 255)

	color = npr.randint(0, 6)
	if color == 0:
		img[img[:,:,2] > threshold] = [0,0,new_intesity]
	elif color == 1:
		img[img[:,:,1] > threshold] = [0,new_intesity,0]
	elif color == 2:
		img[img[:,:,0] > threshold] = [new_intesity,0,0]
	else:
		b_int = npr.randint(new_intesity,255)
		g_int = npr.randint(new_intesity,255)
		r_int = npr.randint(new_intesity,255)
		img[img[:,:,0] > threshold] = [b_int, g_int, r_int]

	return img

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
	minLoc = (np.random.randint(grid[0], min(grid[0]+x_step, w)), 
		np.random.randint(grid[1], min(grid[1]+x_step, h)))
	num_shapes = np.random.randint(lo,hi+1)
	for i in range(num_shapes):
		stretch = np.random.randint(40, 100)
		diff1, diff2 = np.random.randint(-5,5), np.random.randint(-5,5)
		x1, y1 = minLoc[0] +  diff1* stretch  , minLoc[1] + diff2 * stretch
		x2, y2 = x1 + np.random.randint(1,12)/5 * diff1 * stretch,  y1 + np.random.randint(1,12)/5 * diff2* stretch
		pts = np.array((minLoc, (x1, y1), (x2, y2)), dtype=int)

		c1, c2, c3 = np.random.randint(0,50),np.random.randint(0,50),np.random.randint(0,50)
		cv2.fillConvexPoly(im, pts, color= (c1,c2,c3))

	return im

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


def get_random_color():
	b_int = npr.randint(0,255)
	g_int = npr.randint(0,255)
	r_int = npr.randint(0,255)
	return b_int, g_int, r_int


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


		
		cv2.fillConvexPoly(overlay1, pts, color= tuple([int(x) for x in colors[0]]) )
		cv2.fillConvexPoly(overlay2, pts, color= tuple([int(x) for x in colors[1]]) )


	############
	a1, a2 = random.choice(angles), random.choice(angles)

	return gradient(output, color_blend(im, overlay1, overlay2, a1), a2)


def add_vertical_pattern(img):
	color = np.random.randint(0,255,size = 3)
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
			img[((vertical_shift + x + 2 * segment_length)% height):((vertical_shift +
			 x + 4 * segment_length)% height), y, :] = color
			img[(vertical_shift + x + 5 * segment_length)% height, y, :] = color
			img[(vertical_shift + x + 6 * segment_length)% height, y, :] = color
		row_count += 1
	return img



def vrglow(image,pixel,color,radius):
	transparent=np.zeros((image.shape[0],image.shape[1],4),dtype=np.uint8)
	for i in list(range(-radius,radius+1)):
		for j in list(range(1,radius+1)):
			if(((pixel[0]+i)<transparent.shape[0]) and ((pixel[0]+i)>=0) and
			 ((pixel[1]+j)<transparent.shape[1])):
				transparent[pixel[0]+i,pixel[1]+j]=[color[0],color[1],color[2],
				max(255-(255*(math.sqrt(i**2+j**2))/radius),0)]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2RGBA)
	foreground=Image.fromarray(transparent,mode="RGBA")
	background=Image.fromarray(image,mode="RGBA")
	background.paste(foreground,(0,0),foreground)
	image=np.array(background)
	image=cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
	return image

def vlglow(image,pixel,color,radius):
	transparent=np.zeros((image.shape[0],image.shape[1],4),dtype=np.uint8)
	for i in list(range(-radius,radius+1)):
		for j in list(range(-radius,0)):
			if(((pixel[0]+i)<transparent.shape[0]) and ((pixel[0]+i)>=0) and
			 ((pixel[1]+j)<transparent.shape[1])):
				transparent[pixel[0]+i,pixel[1]+j]=[color[0],color[1],color[2],
				max(255-(255*(math.sqrt(i**2+j**2))/radius),0)]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2RGBA)
	foreground=Image.fromarray(transparent,mode="RGBA")
	background=Image.fromarray(image,mode="RGBA")
	background.paste(foreground,(0,0),foreground)
	image=np.array(background)
	image=cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
	return image

def hbglow(image,pixel,color,radius):
	transparent=np.zeros((image.shape[0],image.shape[1],4),dtype=np.uint8)
	for j in list(range(-radius,radius+1)):
		for i in list(range(1,radius+1)):
			if(((pixel[0]+i)<transparent.shape[0]) and ((pixel[0]+i)>=0) and 
				((pixel[1]+j)<transparent.shape[1])):
				transparent[pixel[0]+i,pixel[1]+j]=[color[0],color[1],color[2],
				max(255-(255*(math.sqrt(i**2+j**2))/radius),0)]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2RGBA)
	foreground=Image.fromarray(transparent,mode="RGBA")
	background=Image.fromarray(image,mode="RGBA")
	background.paste(foreground,(0,0),foreground)
	image=np.array(background)
	image=cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
	return image

def htglow(image,pixel,color,radius):
	transparent=np.zeros((image.shape[0],image.shape[1],4),dtype=np.uint8)
	for j in list(range(-radius,radius+1)):
		for i in list(range(-radius,0)):
			if(((pixel[0]+i)<transparent.shape[0]) and ((pixel[0]+i)>=0) and
			 ((pixel[1]+j)<transparent.shape[1])):
				transparent[pixel[0]+i,pixel[1]+j]=[color[0],color[1],color[2],
				max(255-(255*(math.sqrt(i**2+j**2))/radius),0)]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2RGBA)
	foreground=Image.fromarray(transparent,mode="RGBA")
	background=Image.fromarray(image,mode="RGBA")
	background.paste(foreground,(0,0),foreground)
	image=np.array(background)
	image=cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
	return image

def line_pixelation(img):
	while(True):
		vertical=0
		horizontal=1

		# PARAMETERS
		skipstripe=random.randrange(0,2)
		orientation=random.randrange(0,2)
		brightness=random.randrange(0,2)
		monobias=random.randrange(0,3)
		biasval=random.randrange(0,11)
		glow=random.randrange(0,8)

		image=np.copy(img)
		height=max(abs(int(np.random.normal(int(image.shape[0]/200),2))),1)
		width=max(abs(int(np.random.normal(height,2))),1)
		if(orientation==vertical):
			if(width<image.shape[1]):
				indent=random.randrange(0,image.shape[1]-width)
			else:
				print("Error: 'width' is too large for input dimensions. ")
				continue
		if(orientation==horizontal):
			indent=random.randrange(0,image.shape[0]-height)
			if(height<image.shape[0]):
				indent=random.randrange(0,image.shape[0]-height)
			else:
				print("Error: 'height' is too large for input dimensions.")
				continue
		stripes=random.randrange(1,max(1+abs(int(np.random.normal(20,20))), 2))
		ss=np.ones(stripes)
		if(skipstripe==1):
			ss=[1]
			for i in list(range(stripes-2)):
				ss.append(random.randrange(0,2))
			ss.append(1)
		if(monobias==1):
			monocolor=[0,0,0]
		if(monobias==2):
			monocolor=[255,255,255]

		if(orientation==vertical):
			for n in list(range(stripes)):
				if (ss[n]==1):
					for i in list(range(0,image.shape[0],height)):
						color=np.array([random.randrange(0,255),
							random.randrange(0,255),random.randrange(0,255)])
						mono=0
						if(monobias>0):
							mono=random.randrange(1,11)
						if(glow==6 and n==0 and random.randrange(1,10)<random.randrange(1,3)):
							image=vlglow(image,[i*height+int(height/2),indent],
								color,random.randrange(5,4+4*height))
						if(glow==7 and n==(len(ss)-1) and random.randrange(1,10)<random.randrange(1,4)):
							image=vrglow(image,[i*height+int(height/2),indent+n*width],
								color,random.randrange(5,70))
						for j in list(range(height)):
							for k in list(range(width)):
								localcolor=np.array(color)
								if(((i+j)<image.shape[0]) and (indent+k+n*width<image.shape[1])):
									if(brightness==1 and mono<=biasval):
										seed=int(np.random.normal(0,10))
										localcolor[0]=max(min(color[0]+seed,255),0)
										localcolor[1]=max(min(color[1]+seed,255),0)
										localcolor[2]=max(min(color[2]+seed,255),0)
									elif(mono>biasval):
										localcolor=monocolor
									image[i+j,indent+(k+n*width)]=localcolor

		if(orientation==horizontal):
			for n in list(range(stripes)):
				if (ss[n]==1):
					for i in list(range(0,image.shape[1],width)):
						color=np.array([random.randrange(0,255),random.randrange(0,255),
							random.randrange(0,255)])
						mono=0
						if(monobias>0):
							mono=random.randrange(1,11)
						if(glow==6 and n==0 and random.randrange(1,10)<random.randrange(1,3)):
							image=htglow(image,[indent,i*width+int(width/2)],
								color,random.randrange(5,4+4*width))
						if(glow==7 and n==(len(ss)-1) and 
							random.randrange(1,10)<random.randrange(1,4)):
							image=hbglow(image,[indent+height*n,i*width+int(width/2)],
								color,random.randrange(5,70))
						for j in list(range(width)):
							for k in list(range(height)):
								localcolor=np.array(color)
								if(((k+n*height+indent)<image.shape[0]) and (i+j<image.shape[1])):
									if(brightness==1 and mono<=biasval):
										seed=int(np.random.normal(0,10))
										localcolor[0]=max(min(color[0]+seed,255),0)
										localcolor[1]=max(min(color[1]+seed,255),0)
										localcolor[2]=max(min(color[2]+seed,255),0)
									elif(mono>biasval):
										localcolor=monocolor
									image[indent+k+(n*height),i+j]=localcolor

		if(not np.array_equal(img,image)):
			return image








































