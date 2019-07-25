import os
import numpy as np
import cv2
import random as rand
import numpy.random as npr
import math
import copy
import sys
from matplotlib import path

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps, ImageFile
from functools import *


def pointInTriangle4(p,t):
	m1 = np.array([ [1, t[0][0], t[0][1]],
					[1, t[1][0], t[1][1]],
					[1, p[0]   , p[1]   ]])
	m2 = np.array([ [1, t[1][0], t[1][1]],
					[1, t[2][0], t[2][1]],
					[1, p[0]   , p[1]   ]])
	m3 = np.array([ [1, t[2][0], t[2][1]],
					[1, t[0][0], t[0][1]],
					[1, p[0]   , p[1]   ]])
	r = (np.linalg.det(m1) >= 0) == (np.linalg.det(m2) >= 0) == (np.linalg.det(m3) >= 0)
	return r

def pointInTriangle6(p, t):
	triangle = path.Path([t[0], t[1], t[2]])
	points = np.array([p[0], p[1]]).reshape(1, 2)
	res = triangle.contains_points(points)
	return res[0]


def vecLen(v):
	return math.sqrt(v[0]**2 + v[1]**2)

def euclidDistance(p1, p2):
	p = (p2[0]-p1[0], p2[1]-p1[1])
	return vecLen(p)

def alreadyVisited(t, visited):
	for v in visited:
		if t == None or equal(t,v):
			return True
	return False

def getOuterPoint(nextT, thisT):
	for i in range(3):
		if nextT[i] != thisT[0] and nextT[i] != thisT[1] and nextT[i] != thisT[2]:
			return nextT[i]
	print ("not happening...")

def debugDrawPath(triangles, startT, p):
	im = Image.new('RGB', (2000, 2000))
	draw = ImageDraw.Draw(im)
	lastT = triangles[0]
	for t in triangles[1:]:
		tmpP1 = ((lastT[0][0]+lastT[1][0]+lastT[2][0])/3.0,
				 (lastT[0][1]+lastT[1][1]+lastT[2][1])/3.0)
		tmpP2 = ((t[0][0]+t[1][0]+t[2][0])/3.0, (t[0][1]+t[1][1]+t[2][1])/3.0)
		draw.polygon((t[0],t[1],t[2]), fill=None, outline='blue')
		draw.line([tmpP1, tmpP2],fill='white')
		lastT = t

	tmpP = (int((startT[0][0]+startT[1][0]+startT[2][0])/3.0),
			int((startT[0][1]+startT[1][1]+startT[2][1])/3.0))
	print ("Tuple:","(",startT[0],startT[1],startT[2],")")
	print (tmpP)
	draw.arc((p[0], p[1], p[0]+20,p[1]+20),0,360,fill='red')
	draw.arc((tmpP[0], tmpP[1], tmpP[0]+20,tmpP[1]+20),0,360,fill='green')

	im.save("some_lines.jpg")

def getCenterPoint(t):
	return ((t[0][0]+t[1][0]+t[2][0])/3, (t[0][1]+t[1][1]+t[2][1])/3)

def normVectorFromPoints(p1, p2):
	v = (p2[0]-p1[0], p2[1]-p1[1])
	div = np.linalg.norm(v)
	if abs(div) < 0.00001:
		return v
	else:
		return v / div

def tupleToString(t):
	return "{" + str(t[0]) + ", " + str(t[1]) + ", " + str(t[0]) + "}"

def printTriangleList(l):
	for t in l:
		if t != None:
			print (tupleToString(t))
		else:
			print ("None")
	print ("")

def isEmpty(t):
	return t == None or (t[0] == None and t[1] == None and t[2] == None)

def findTriangle2Rec(p, t, lastT, debug):

	count = 0
	visited = []
	index = 0

	while True:
		upStream = False
		count += 1
		if not isEmpty(t) and pointInTriangle6(p, t):
			return t

		d1 = sys.maxsize
		d2 = sys.maxsize

		t1 = t[3]
		t2 = t[4]
		t3 = t[5]

		if equal(t1, lastT):
			nextT1 = t2
			nextT2 = t3
		else:
			if equal(t2, lastT):
				nextT1 = t1
				nextT2 = t3
			else:
				nextT1 = t1
				nextT2 = t2

		visitedT1 = alreadyVisited(nextT1, visited)
		visitedT2 = alreadyVisited(nextT2, visited)

		if not alreadyVisited(t, visited):
			visited.append(t)

		if nextT1 == None and nextT2 != None:
			if not visitedT2:
				t = nextT2
				continue
			else:
				upStream = True

		if nextT2 == None and nextT1 != None:
			if not visitedT1:
				t = nextT1
				continue
			else:
				upStream = True

		if not upStream and not isEmpty(t):
			herePos   = getCenterPoint(t)
			targetPos = p
			dir1Pos   = getCenterPoint(nextT1)
			dir2Pos   = getCenterPoint(nextT2)
			dir1      = normVectorFromPoints(herePos, dir1Pos)
			dir2      = normVectorFromPoints(herePos, dir2Pos)
			targetVector = normVectorFromPoints(herePos, targetPos)
			dot1 = np.dot(targetVector, dir1)
			dot2 = np.dot(targetVector, dir2)

			if dot1 > dot2 and not visitedT1:
				lastT = t
				t = nextT1
				continue

			if dot2 > dot1 and not visitedT2:
				lastT = t
				t = nextT2
				continue

			if not visitedT1:
				lastT = t
				t = nextT1
				continue

			if not visitedT2:
				lastT = t
				t = nextT2
				continue

		lastT = t
		index -= 1
		if index <= 0:
			index = len(visited)-1
		t = visited[index]

def firstNotNoneTriangle(triangles):
	for t in triangles:
		if not isEmpty(t):
			return t


def findTriangle2(point, triangles, debug):
	t = findTriangle2Rec(point, firstNotNoneTriangle(triangles), None, debug)
	return t

def dist(p1, p2):
	a = (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2
	return math.sqrt(a)

def pointOnLine2(p, p1, p2):
	d1 = dist(p, p1)
	d2 = dist(p, p2)
	d3 = dist(p1,p2)
	return math.fabs(d3 - (d1+d2)) <= 0.0000001

def equal(t1, t2):
	return (t1 == None and t2 == None) or \
		   (t1 != None and t2 != None and (t1[0] == t2[0] and \
			t1[1] == t2[1] and t1[2] == t2[2]))

def removeTriangleFromListImplicit(triangle):

	changeReferenceFromTo(triangle, None, triangle[3])
	changeReferenceFromTo(triangle, None, triangle[4])
	changeReferenceFromTo(triangle, None, triangle[5])
	triangle[3] = None
	triangle[4] = None
	triangle[5] = None
	triangle[0] = None
	triangle[1] = None
	triangle[2] = None

def pointOnLine(p, t):
	if pointOnLine2(p, t[0], t[1]):
		return 2
	if pointOnLine2(p, t[1], t[2]):
		return 0
	if pointOnLine2(p, t[2], t[0]):
		return 1
	return -1

def matchT(t, p1, p2, notp3):
	b1 = False
	b2 = False
	b3 = True
	index = -1
	for i in range(3):
		if t[i] == p1:
			b1 = True
			continue
		if t[i] == p2:
			b2 = True
			continue
		if t[i] == notp3:
			b3 = False
		else:
			index = i
	return (b1 and b2 and b3, index)

def findNeighbourTriangle(triangles, p1, p2, notp3):
	for t in triangles:
		(b, i) = matchT(t, p1, p2, notp3)
		if b:
			return (t, i)
	return (None, -1)

def notValid(t, p):
	matrix = np.array(
			[[t[0][0],t[0][1],t[0][0]**2+t[0][1]**2,1],
			 [t[1][0],t[1][1],t[1][0]**2+t[1][1]**2,1],
			 [t[2][0],t[2][1],t[2][0]**2+t[2][1]**2,1],
			 [p[0]   ,p[1]   ,p[0]**2   +p[1]**2   ,1]])

	return np.linalg.det(matrix) > 0

def pointsMatchTriangle(p1, p2, t):
	return t != None and \
		   (p1 == t[0] or p1 == t[1] or p1 == t[2]) and \
		   (p2 == t[0] or p2 == t[1] or p2 == t[2])

def getReferenceWithPoints(p1, p2, t):
	for i in range(3):
		if pointsMatchTriangle(p1, p2, t[3+i]):
			return t[3+i]
	return None

def legalize(triangles, t):
	nextT = getReferenceWithPoints(t[1], t[2], t)

	if nextT == None:
		return triangles

	(b, nextI) = matchT(nextT, t[1], t[2], t[0])
	p = nextT[nextI]

	if notValid(t, p):

		t1 = [t[0], t[1], p, getReferenceWithPoints(t[0], t[1], t),
							 getReferenceWithPoints(t[1], p, nextT)]
		t2 = [t[0], p, t[2], getReferenceWithPoints(t[0], t[2], t),
							 getReferenceWithPoints(t[2], p, nextT), t1]

		t1.append(t2)

		changeReferenceFromTo(t, t1, t1[3])
		changeReferenceFromTo(nextT, t1, t1[4])

		changeReferenceFromTo(t, t2, t2[3])
		changeReferenceFromTo(nextT, t2, t2[4])

		removeTriangleFromListImplicit(t)
		removeTriangleFromListImplicit(nextT)

		triangles.append(t1)
		triangles.append(t2)

		triangles = legalize(triangles, t1)
		triangles = legalize(triangles, t2)
	return triangles

def changeReferenceFromTo(fromT, toT, t):
	for i in range(3,6):
		if fromT != None and t != None and t[i] != None and equal(t[i], fromT):
			t[i] = toT
			break

def findNextTriangleWithPoint(point, t):
	for i in range(3, 6):
		if t != None and t[i] != None and pointInTriangle4(point, t[i]):
			return t[i]



def insertPointIntoTriangles(point, triangles, debug):
	t = findTriangle2(point, triangles, debug)
	line = pointOnLine(point, t)

	if line == -1:
		t1 = [point, t[0], t[1], getReferenceWithPoints(t[0], t[1], t)]
		t2 = [point, t[1], t[2], getReferenceWithPoints(t[1], t[2], t), t1]
		t3 = [point, t[2], t[0], getReferenceWithPoints(t[2], t[0], t), t1, t2]

		changeReferenceFromTo(t, t1, t1[3])
		changeReferenceFromTo(t, t2, t2[3])
		changeReferenceFromTo(t, t3, t3[3])

		t2.append(t3)
		t1.append(t2)
		t1.append(t3)

		triangles.append(t1)
		triangles.append(t2)
		triangles.append(t3)

		removeTriangleFromListImplicit(t)

		triangles = legalize(triangles, t1)
		triangles = legalize(triangles, t2)
		triangles = legalize(triangles, t3)

	else:
		t2 = findNextTriangleWithPoint(point, t)

		if t2 == None:
			return triangles

		line2 = pointOnLine(point, t2)

		tt1 = [point, t[line], t[(line+1)%3],
			   getReferenceWithPoints(t[line], t[(line+1)%3], t)]
		tt2 = [point, t[(line+2)%3], t[line],
			   getReferenceWithPoints(t[(line+2)%3], t[line], t), tt1]

		tt3 = [point, t2[line2], t2[(line2+1)%3],
			   getReferenceWithPoints(t2[line2], t2[(line2+1)%3], t2)]
		tt4 = [point, t2[(line2+2)%3], t2[line2],
			   getReferenceWithPoints(t2[(line2+2)%3], t2[line2], t2), tt3]

		tt1.append(tt2)
		tt3.append(tt4)

		changeReferenceFromTo(t,  tt1, tt1[3])
		changeReferenceFromTo(t,  tt2, tt2[3])
		changeReferenceFromTo(t2, tt3, tt3[3])
		changeReferenceFromTo(t2, tt4, tt4[3])

		if getReferenceWithPoints(point, t[(line+1)%3], tt3) != None:
			tt1.append(getReferenceWithPoints(point, t[(line+1)%3], tt3))
			tt2.append(getReferenceWithPoints(point, t[(line+2)%3], tt4))
		else:
			tt1.append(getReferenceWithPoints(point, t[(line+1)%3], tt4))
			tt2.append(getReferenceWithPoints(point, t[(line+2)%3], tt3))

		if getReferenceWithPoints(point, t2[(line2+1)%3], tt1) != None:
			tt3.append(getReferenceWithPoints(point, t2[(line2+1)%3], tt1))
			tt4.append(getReferenceWithPoints(point, t2[(line2+2)%3], tt2))
		else:
			tt3.append(getReferenceWithPoints(point, t2[(line2+1)%3], tt2))
			tt4.append(getReferenceWithPoints(point, t2[(line2+2)%3], tt1))

		triangles.append(tt1)
		triangles.append(tt2)
		triangles.append(tt3)
		triangles.append(tt4)

		removeTriangleFromListImplicit(t)
		removeTriangleFromListImplicit(t2)

		triangles = legalize(triangles, tt1)
		triangles = legalize(triangles, tt2)
		triangles = legalize(triangles, tt3)
		triangles = legalize(triangles, tt4)

	return triangles

def createDelaunayTriangulation(points, triangle):
	triangles = [triangle]
	for point in points:
		triangles = insertPointIntoTriangles(point, triangles, False)
	return triangles

def maxCoord(points):
	ma = 0
	mi = 0
	for p in points:
		ma = max(max(p), ma)
		mi = min(min(p), mi)
	return max(ma, math.fabs(mi))

def pointInRange(p, minX, minY, maxX, maxY):
	return p[0] < maxX and p[1] < maxY and p[0] > minX and p[1] > minY

def triangleInRange(t, minX, minY, maxX, maxY):
	return not isEmpty(t) and \
			pointInRange(t[0], minX, minY, maxX, maxY) and \
			pointInRange(t[1], minX, minY, maxX, maxY) and \
			pointInRange(t[2], minX, minY, maxX, maxY)


def removeOutOfBoundsTriangles(triangles, minX, minY, maxX, maxY):
	newTriangles = []
	good = 0
	bad = 0
	for t in triangles:
		if triangleInRange(t, minX, minY, maxX, maxY):
			newT = [t[0],t[1],t[2], t[3], t[4], t[5]]
			for i in range(3,6):
				if not triangleInRange(newT[i], minX, minY, maxX, maxY):
					newT[i] = None
			newTriangles.append(tuple(newT))
			good += 1
		else:
			bad += 1
	return newTriangles

def delaunay(points):
	m = maxCoord(points)
	t = [(-3*m,-3*m), (3*m,0), (0,3*m), None, None, None]

	triangles = createDelaunayTriangulation(points, t)
	triangles = removeOutOfBoundsTriangles(triangles, 0, 0, m, m)
	return triangles


def loadAndFilterImage(img):
	# orig = Image.open(name)
	img = img[:,:,[2,1,0]]
	orig = Image.fromarray(img)

	(width, height) = orig.size
	im = orig.convert("L")
	im = im.filter(ImageFilter.GaussianBlur(radius=5))
	im = im.filter(ImageFilter.FIND_EDGES)
	im = brightenImage(im, 20.0)

	im = im.filter(ImageFilter.GaussianBlur(radius=5))
	return (orig, im)

def generateRandomPoints(count, sizeX, sizeY):
	x_start = 0
	x_end = sizeX
	y_start = 0
	y_end = sizeY

	points = []
	for i in range(count):
		p = (rand.randint(x_start,x_end),rand.randint(y_start,y_end))
		if not p in points:
			points.append(p)

	return points

def generateNonRandomPoints(sizeX, sizeY, width_ratio):
	points = []
	triangle_width = int(sizeX * width_ratio)

	for i in range(0,sizeX,triangle_width):
		for j in range(0, sizeY,triangle_width):
			p = (i,  j)
			points.append(p)
	return points

def brightenImage(im, value):
	enhancer = ImageEnhance.Brightness(im)
	im = enhancer.enhance(value)
	return im

def delaunayFromPoints(points):
	triangles = delaunay(points)
	return triangles

def drawImageColoredTriangles(triangles, origIm, multiplier):
	(sizeX, sizeY) = origIm.size
	im = Image.new('RGB', (sizeX*multiplier, sizeY*multiplier))
	draw = ImageDraw.Draw(im)
	for t in triangles:
		(r,g,b) = getTriangleColor(t, origIm)
		p0 = tuple(map(lambda x:x*multiplier, t[0]))
		p1 = tuple(map(lambda x:x*multiplier, t[1]))
		p2 = tuple(map(lambda x:x*multiplier, t[2]))
		drawT = (p0, p1, p2)
		draw.polygon(drawT, fill=(r,g,b,255))
	im = brightenImage(im, 5.0) 
	return im  

	
def getCenterPoint(t):
	return ((t[0][0]+t[1][0]+t[2][0])/3, (t[0][1]+t[1][1]+t[2][1])/3)

def getTriangleColor(t, im):
	color = []
	for i in range(3):
		p = t[i]
		if p[0] >= im.size[0] or p[0] < 0 or p[1] >= im.size[1] or p[1] < 0:
			continue
		color.append(im.getpixel(p))

	p = getCenterPoint(t)
	if p[0] < im.size[0] and p[0] >= 0 and p[1] < im.size[1] and p[1] >= 0:
		centerPixel = im.getpixel(p)
		color = color + [centerPixel]*3

	div = float(len(color))
	color = reduce(lambda rec, x : ((rec[0]+x[0])/div, (rec[1]+x[1])/div, (rec[2]+x[2])/div), color, (0,0,0))
	color = map(lambda x : int(x), color)
	return color


def triangulation(img, is_random = False):
	(orig, blackIm) = loadAndFilterImage(img)
	orig_np = np.array(orig)
	height, width, _ = orig_np.shape
	x0 = npr.randint(0, int(height / 2))
	y0 = npr.randint(0, int(width / 2))
	x1 = npr.randint(min(x0 + int(height / 4), height), min(x0 + int(height / 2), height))
	y1 = npr.randint(min(y0 + int(height / 4), width), min(y0 + int(height / 2), width))

	cropped_img = orig.crop([y0, x0, y1, x1])
	colorIm = cropped_img

	(width, height) = colorIm.size
	multiplier = 10
	width_ratio = np.random.uniform(0.02, 0.03)
	num_points = np.random.randint(1000, 2000)

	points = None
	if is_random:
		points = generateRandomPoints(num_points, width, height)
	else:
		points = generateNonRandomPoints(width, height, width_ratio)

	triangles = delaunayFromPoints(points)
	img = drawImageColoredTriangles(triangles, colorIm, multiplier)
	img = np.array(img)
	img = cv2.resize(img, (y1-y0, x1-x0), interpolation = cv2.INTER_LINEAR)

	margin_size = int(width * width_ratio * 2.2)
	orig_np[x0+margin_size:x1-margin_size,y0+ margin_size :y1- margin_size , :] = img[margin_size:-margin_size, margin_size:-margin_size, :]
	return orig_np[:,:,[2,1,0]]





