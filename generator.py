#!/usr/bin/python3

# commit and push to git with: git add . ; git commit -m "Next commit3" ; git push origin master

from __future__ import print_function
from __future__ import division
from math import atan2, cos, sin, sqrt, pi


from os import listdir
import numpy as np

from random import randint
from mahotas import features
from sklearn.decomposition import PCA
import math
import cv2


print("x"*50)
print("Image Generator")
print("x"*50)
print()


#Funktionsdefinitionen

def invert_image(img):
	return (255-img)

def add_images(img1, img2):
	temp = invert_image(img1) + invert_image(img2)
	return invert_image(temp)

def add_images4(img1, img2, img3, img4):
	temp = invert_image(img1) + invert_image(img2) + invert_image(img3) + invert_image(img4)
	return invert_image(temp)

def translate(img, x, y):
	rows, cols = img.shape

	M = np.float32([[1,0,x], [0,1,y]])
	img = cv2.warpAffine(img, M, (cols,rows), borderMode = cv2.BORDER_REPLICATE)

	return img

def rotate(img, winkel):
	rows, cols = img.shape

	# Argumente: Center, Angle, Scale
	M = cv2.getRotationMatrix2D((cols/2,rows/2),winkel,1)
	img = cv2.warpAffine(img, M, (cols,rows), borderMode = cv2.BORDER_REPLICATE)

	return img


# Generate Scene
def generateScene():
    line_styles = ["dreieck", "ellipse", "gerade", "rechteck"]
    orientierung = ["links", "oben", "rechts", "unten"]
    path0 = ["", "", "", ""]
    path1 = ["", "", "", ""]

    for num in range(4):
        path0[num] = "Kanten/" + "kante_" + orientierung[num] + "_" + line_styles[randint(0, 3)] + ".png"

    for num in range(4):
        path1[num] = "Kanten/" + "kante_" + orientierung[num] + "_" + line_styles[randint(0, 3)] + ".png"

    path1[1] = path0[3]

    img0 = []
    img1 = []

    for num in range(4):
        img0.append(cv2.imread(path0[num], 0))
        img1.append(cv2.imread(path1[num], 0))

    img1[1] = translate(img1[1], 0, -100)

    obj0 = add_images4(img0[0], img0[1], img0[2], img0[3])
    obj1 = add_images4(img1[0], img1[1], img1[2], img1[3])

    obj1 = translate(obj1, 0, 100)
    
    return add_images(obj0,obj1)


def get_huMoments(img, name):
	# Hu-Momente
	moments = cv2.moments(img, False)
	huMoments = cv2.HuMoments(moments)

	# log Transformation
	for i in range(0,7):
		huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
	print()
	print("HuMoments (log corrected) from "+ name +": {}".format(huMoments))

	return huMoments


def get_matchShapes(img0, img1, name0, name1):
	# MatchShapes
	contours_match = [0, 0, 0]
	contours_match[0] = cv2.matchShapes(img0, img1, cv2.CONTOURS_MATCH_I1, 0)
	contours_match[1] = cv2.matchShapes(img0, img1, cv2.CONTOURS_MATCH_I2, 0)
	contours_match[2] = cv2.matchShapes(img0, img1, cv2.CONTOURS_MATCH_I3, 0)
	print()
	print("ContoursMatch of "+ name0 +" and "+ name1 +": {}".format(contours_match))

	return contours_match


# Flächenschwerpunkt berechnen
def calcCentroid(img):
    moments = cv2.moments(img, False)

    cX = float(moments["m10"] / moments["m00"])
    cY = float(moments["m01"] / moments["m00"])

    print()
    print("Flächenschwerpunkt: {}, {}".format(cX, cY))


# Zernike Moments
def get_zernikeMoments(img, name):
	ordnung = 8
	radius = 200
	zernike = features.zernike_moments(img, radius, ordnung)
	print()
	print("Zernike Momente von "+name+" der Ordnung {}, Radius in px {}: {}".format(ordnung, radius, zernike))

	return zernike


# PCA Achsen Zeichnen
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)


#Ausführbereich

scene = generateScene()
#scene2 = generateScene()

cv2.imshow("Scene", scene)
#cv2.imshow("Scene2", scene2)

get_huMoments(scene, "scene")
calcCentroid(scene)
get_zernikeMoments(scene, "Scene")

output1 = rotate(scene, randint(0, 360))
output1 = translate(output1, randint(-200, 200), randint(-200, 200))

get_huMoments(output1, "output1")
calcCentroid(output1)
get_zernikeMoments(output1, "Output1")

output2 = rotate(scene, randint(0, 360))
output2 = translate(output2, randint(-200, 200), randint(-200, 200))

get_huMoments(output2, "output2")
calcCentroid(output2)
get_zernikeMoments(output2, "Output2")

#get_zernikeMoments(scene2, "Scene2")

get_matchShapes(output1, output2, "output1", "output2")
#get_matchShapes(output1, scene2, "output1", "scene2")

def pca(img):

	# Convert image to binary
	_, bw = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	# PCA for everything
	contour_size = 0

	for i, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		# Ignore contours that are too small or too large
		if area < 1e2 or 1e5 < area:
		    continue

		contour_size += len(contour)

	all_data_pts = np.empty((contour_size, 2), dtype=np.float64)


	k = 0

	for i, contour in enumerate(contours):

		area = cv2.contourArea(contour)
		# Ignore contours that are too small or too large
		if area < 1e2 or 1e5 < area:
		    continue

		sz = len(contour)
		data_pts = np.empty((sz, 2), dtype=np.float64)

		# cv2.drawContours(scene, contours, i, (0, 255, 0), 3)

		for i in range(data_pts.shape[0]):
			data_pts[i,0] = contour[i,0,0]
			data_pts[i,1] = contour[i,0,1]

			all_data_pts[i+k,0] = contour[i,0,0]
			all_data_pts[i+k,1] = contour[i,0,1]

		k += len(contour)



	# import sys
	# np.set_printoptions(threshold=sys.maxsize)
	# print("{}".format(all_data_pts))

	# Perform PCA analysis
	mean = np.empty((0))
	mean, eigenvectors, eigenvalues = cv2.PCACompute2(all_data_pts, mean)
	# Store the center of the object
	cntr = (int(mean[0,0]), int(mean[0,1]))

	cv2.circle(img, cntr, 4, (0, 255, 0), 1)

	# p1 = x1, y1
	# p2 = x2, y2
	p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
	p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])

	# drawAxis(scene, cntr, p1, (0, 255, 0), 1)
	# drawAxis(scene, cntr, p2, (0, 255, 0), 1)

	print()
	print("Eigenvectors: {}".format(eigenvectors))
	print()
	print("Eigenwerte: {}".format(eigenvalues))

	angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians


	rows, cols = img.shape
	count = 0
	for y in range(rows):
		for x in range(cols):
			if bw[y, x] <= 50:
				count += 1

	a = np.empty((count, 2), dtype=np.float64)

	cc = 0
	for y in range(rows):
		for x in range(cols):
			if bw[y, x] <= 50:
				a[cc, 0] = x
				a[cc, 1] = y
				cc += 1

	# PCA mit sklearn
	pca = PCA(n_components=2)
	pca.fit(a)

	print(pca.components_)
	print(pca.explained_variance_)

	eigenvectors = pca.components_
	eigenvalues = pca.explained_variance_

	print()
	print("Eigenvectors: {}".format(eigenvectors))
	print()
	print("Eigenwerte: {}".format(eigenvalues))

	p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0])
	p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1])

	drawAxis(img, cntr, p1, (0, 255, 0), 1)
	drawAxis(img, cntr, p2, (0, 255, 0), 1)

	# PCA mit sklearn
	pca = PCA(n_components=2)
	pca.fit(all_data_pts)

	print(pca.components_)
	print(pca.explained_variance_)

	eigenvectors = pca.components_
	eigenvalues = pca.explained_variance_

	print()
	print("Eigenvectors: {}".format(eigenvectors))
	print()
	print("Eigenwerte: {}".format(eigenvalues))

	p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0])
	p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1])

	# drawAxis(scene, cntr, p1, (0, 255, 0), 1)
	# drawAxis(scene, cntr, p2, (0, 255, 0), 1)

	return img


output1 = pca(output1)
output2 = pca(output2)

cv2.imshow('output1', output1)
cv2.imshow('output2', output2)

cv2.waitKey(0)
cv2.destroyAllWindows()
