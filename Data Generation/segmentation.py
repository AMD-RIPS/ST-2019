
# Random Patches: This cell produces random blue patches on the games using contouring (segmentation).
# The starting size and number of patches are adjustable. Currently they are randomly chosen from a range of integers.



import numpy as np
import imutils
import cv2
import random
input_path = 'borderland.jpg'




############### Discoloration
img = cv2.imread('civ.jpg')
img[img[:,:,0] > 120] = [230,0,0]
out_path = '/Users/parmida/Downloads/discoloration_' + input_path
cv2.imwrite(out_path, img)

########################################
##Random Patches
out_path = '/Users/parmida/Downloads/contour_' + input_path

im = cv2.imread(input_path)
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours.sort(key = len)
patch_number = np.random.randint(3,20)    #maybe these two can be turned into parameters
for i in range(patch_number):
    cv2.drawContours(im, contours,len(contours) - 1 - i , (250,0,0), -1)



cv2.imwrite(out_path, im)



#######################################
#Shapes: produces thin dark triangles that all originate from the same point (random point in the darkest region)
#and point to different directions

out_path = '/Users/parmida/Downloads/shapes_' + input_path

im = cv2.imread(input_path)
h, w, _ = im.shape
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


########### this chunk is to find the darkest region of the image##########
grid = (x,y)
mean_shade = np.mean(im)
x_step, y_step = int(w/6), int(h/4)
for x in range(0, w, x_step):
    for y in range(0, h, y_step):
        new_shade = np.mean(im[x:x+x_step, y:y+y_step])
        if  new_shade < mean_shade:
            mean_shade = new_shade
            grid = (x,y)
###########



minLoc = (np.random.randint(grid[0], min(grid[0]+x_step, w)), np.random.randint(grid[1], min(grid[1]+x_step, h)))
num_shapes = np.random.randint(2,5)
for i in range(num_shapes):
    stretch = np.random.randint(40, 100)
    diff1, diff2 = np.random.randint(-5,5), np.random.randint(-5,5)
    x1, y1 = minLoc[0] +  diff1* stretch  , minLoc[1] + diff2 * stretch
    x2, y2 = x1 + np.random.randint(1,12)/5 * diff1 * stretch  , y1 + np.random.randint(1,12)/5 * diff2* stretch
    pts = np.array((minLoc,
                    (x1, y1),
                    (x2, y2)), dtype=int)
        
                    c1, c2, c3 = np.random.randint(0,50),np.random.randint(0,50),np.random.randint(0,50)
                    cv2.fillConvexPoly(im, pts,
                                       color= (c1,c2,c3))



cv2.imwrite(out_path, im)


############################################
#Big Triangles

out_path = '/Users/parmida/Downloads/triangle_' + input_path

im = cv2.imread(input_path)
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
pts = np.array(((x_0, y_0),
                (x_1, y_1),
                (x_2, y_2)), dtype=int)
cv2.fillConvexPoly(overlay, pts,
                   color= tuple([int(x) for x in colors[np.random.randint(3)]]) )



num_shapes = np.random.randint(4)
for i in range(num_shapes):
    
    x_1, y_1 = np.mean([x_1, x_0]) + np.random.randint(-60,60), np.mean([y_1,y_0])+ np.random.randint(-60,60)
    x_2, y_2 = np.mean([x_2, x_0]) + np.random.randint(-60,60), np.mean([y_2,y_0])+ np.random.randint(-60,60)
    
    pts = np.array(((x_0, y_0),
                    (x_1, y_1),
                    (x_2, y_2)), dtype=int)
                    alpha = .95
                    
                    cv2.fillConvexPoly(overlay, pts,
                                       color= tuple([int(x) for x in colors[np.random.randint(3)]]) )


cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                0, output)

cv2.imwrite(out_path, output)



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
def color_blend( img, overlay1, overlay2, angle = 0):
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


#Shaders: produces n_gon hues that go from red to yellow in the 'a1' direction. The fading happens in the 'a2' direction.

out_path = '/Users/parmida/Downloads/shader_' + input_path

im = cv2.imread(input_path)

angles = np.array([0,90,180,270])


h,w,_ = im.shape
output = im.copy()
overlay1 = im.copy()
overlay2 = im.copy()

#####big shaders in forms of n-gons
num_shapes = np.random.randint(1,4)
for i in range(num_shapes):
    x_0, y_0 = np.random.randint(w), np.random.randint(h)
    x_1, y_1 = np.random.randint(-300,w+300), np.random.randint(-300,h+300)
    x_2, y_2 = np.random.randint(-300,w+300), np.random.randint(-300,h+300)
    
    pts = np.array(((x_0, y_0),
                    (x_1, y_1),
                    (x_2, y_2)), dtype=int)
        
                    extra_n = np.random.randint(4)
                    for i in range(extra_n): #extra number of points to make an n_gon
                        pts = np.append(pts, [[np.random.randint(-300,h+300), np.random.randint(-300,w+300)]], axis = 0)

alpha = 1
    
    
    colors = np.array((
                       (0,0,255),
                       (0, 165, 255)),dtype = int)
        
                       
                       cv2.fillConvexPoly(overlay1, pts,
                                          color= tuple([int(x) for x in colors[0]]) )
                       cv2.fillConvexPoly(overlay2, pts,
                                          color= tuple([int(x) for x in colors[1]]) )


############
a1, a2 = random.choice(angles), random.choice(angles)

cv2.imwrite(out_path, gradient(output, color_blend(im, overlay1, overlay2, a1), a2))
