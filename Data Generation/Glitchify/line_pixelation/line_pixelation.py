import os
import random
import cv2
import numpy as np
import math
from PIL import Image


def vrglow(image,pixel,color,radius):
	transparent=np.zeros((image.shape[0],image.shape[1],4),dtype=np.uint8)
	for i in list(range(-radius,radius+1)):
		for j in list(range(1,radius+1)):
			if(((pixel[0]+i)<transparent.shape[0]) and ((pixel[0]+i)>=0) and ((pixel[1]+j)<transparent.shape[1])):
				transparent[pixel[0]+i,pixel[1]+j]=[color[0],color[1],color[2],max(255-(255*(math.sqrt(i**2+j**2))/radius),0)]
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
			if(((pixel[0]+i)<transparent.shape[0]) and ((pixel[0]+i)>=0) and ((pixel[1]+j)<transparent.shape[1])):
				transparent[pixel[0]+i,pixel[1]+j]=[color[0],color[1],color[2],max(255-(255*(math.sqrt(i**2+j**2))/radius),0)]
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
			if(((pixel[0]+i)<transparent.shape[0]) and ((pixel[0]+i)>=0) and ((pixel[1]+j)<transparent.shape[1])):
				transparent[pixel[0]+i,pixel[1]+j]=[color[0],color[1],color[2],max(255-(255*(math.sqrt(i**2+j**2))/radius),0)]
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
			if(((pixel[0]+i)<transparent.shape[0]) and ((pixel[0]+i)>=0) and ((pixel[1]+j)<transparent.shape[1])):
				transparent[pixel[0]+i,pixel[1]+j]=[color[0],color[1],color[2],max(255-(255*(math.sqrt(i**2+j**2))/radius),0)]
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
		stripes=random.randrange(1,1+abs(int(np.random.normal(20,20))))
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
						color=np.array([random.randrange(0,256),random.randrange(0,256),random.randrange(0,256)])
						mono=0
						if(monobias>0):
							mono=random.randrange(1,11)
						if(glow==6 and n==0 and random.randrange(1,10)<random.randrange(1,3)):
							image=vlglow(image,[i*height+int(height/2),indent],color,random.randrange(5,4+4*height))
						if(glow==7 and n==(len(ss)-1) and random.randrange(1,10)<random.randrange(1,4)):
							image=vrglow(image,[i*height+int(height/2),indent+n*width],color,random.randrange(5,70))
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
						color=np.array([random.randrange(0,256),random.randrange(0,256),random.randrange(0,256)])
						mono=0
						if(monobias>0):
							mono=random.randrange(1,11)
						if(glow==6 and n==0 and random.randrange(1,10)<random.randrange(1,3)):
							image=htglow(image,[indent,i*width+int(width/2)],color,random.randrange(5,4+4*width))
						if(glow==7 and n==(len(ss)-1) and random.randrange(1,10)<random.randrange(1,4)):
							image=hbglow(image,[indent+height*n,i*width+int(width/2)],color,random.randrange(5,70))
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



# img = cv2.imread("square_patch.jpg")
# img = noisy_line(img)
# cv2.imwrite("result.png", img)



