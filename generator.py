#!/usr/bin/python3

# commit and push to git with: git add . ; git commit -m "Next commit3" ; git push origin master

from __future__ import print_function
from __future__ import division
from math import atan2, cos, sin, sqrt, pi


from os import listdir
import numpy as np

from random import randint

from mahotas import features

import math

import cv2



print("x"*50)
print("Image Generator")
print("x"*50)
print()

entries = listdir('Kanten/')

image_counter = 0

print("Gefundene Bilder: ")
for file in entries:
	if ".png" in file or ".jpg" in file:
		print(file)
		image_counter += 1

print()
print("Anzahl der gefundenen Bilder: {}".format(image_counter))

# print("Länge: {}".format(len(entries)))


oben_count = 0
unten_count = 0
rechts_count = 0
links_count = 0

for file in entries:
	if "oben" in file:
		oben_count += 1

	if "rechts" in file:
		rechts_count += 1

	if "unten" in file:
		unten_count += 1

	if "links" in file:
		links_count += 1

print()
print("Varianten oben: {}".format(oben_count))
print("Varianten rechts: {}".format(rechts_count))
print("Varianten unten: {}".format(unten_count))
print("Varianten links: {}".format(links_count))

print()
print("Varianten: {}".format(oben_count*rechts_count*unten_count*links_count))

img = []

for num, file in enumerate(entries):
	if ".png" in file or ".jpg" in file:
		print("Reading image: {}, No. {}".format(file, num))
		path = "Kanten/" + file
		img.append(cv2.imread(path, 0))

def invert_image(img):
	return (255-img)

pic0 = invert_image(img[0])
pic1 = invert_image(img[1])

pic_res = pic0 + pic1
pic_res = invert_image(pic_res)

print()
print("Random number between 0 and 10: ")
print(randint(0, 10))

# cv2.imshow('Image 0', pic0)
# cv2.imshow('Image 1', pic1)
# cv2.imshow('Image res', pic_res)

pic_ursprung = pic_res

# Translation
rows, cols = pic_res.shape

x_translate = 100
y_translate = 50

M = np.float32([[1,0,x_translate], [0,1,y_translate]])
pic_res = cv2.warpAffine(pic_res, M, (cols,rows), borderMode = cv2.BORDER_REPLICATE)

print()
print("Verschiebungsmatrix: ")
print(M)

cv2.imshow('Image res - translate', pic_res)

# Rotation
winkel = 23

rows, cols = pic_res.shape

# Argumente: Center, Angle, Scale
M = cv2.getRotationMatrix2D((cols/2,rows/2),winkel,1)
pic_res = cv2.warpAffine(pic_res, M, (cols,rows), borderMode = cv2.BORDER_REPLICATE)

cv2.imshow('Image res - rotate', pic_res)

# Hu-Momente
moments = cv2.moments(pic_res, False)
# print()
# print(moments)
huMoments = cv2.HuMoments(moments)

# log Transformation
for i in range(0,7):
	huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
print()
print("HuMoments (log corrected): {}".format(huMoments))

# MatchShapes
contours_match = [0, 0, 0]
contours_match[0] = cv2.matchShapes(pic_ursprung, pic_res, cv2.CONTOURS_MATCH_I1, 0)
contours_match[1] = cv2.matchShapes(pic_ursprung, pic_res, cv2.CONTOURS_MATCH_I2, 0)
contours_match[2] = cv2.matchShapes(pic_ursprung, pic_res, cv2.CONTOURS_MATCH_I3, 0)
print()
print("ContoursMatch: {}".format(contours_match))

# Zernike Moments
# Flächenschwerpunkt berechnen
cX = int(moments["m10"] / moments["m00"])
cY = int(moments["m01"] / moments["m00"])

print()
print("Flächenschwerpunkt: {}, {}".format(cX, cY))

ordnung = 8
radius = 200
zernike = features.zernike_moments(pic_res, radius, ordnung)
print()
print("Zernike Momente der Ordnung {}, Radius in px {}: {}".format(ordnung, radius, zernike))

# PCA
# mean, eigenVectors = cv2.PCACompute(pic_res, mean=None, maxComponents=2)



def getOrientation(pts, img):

	sz = len(pts)
	data_pts = np.empty((sz, 2), dtype=np.float64)

	for i in range(data_pts.shape[0]):
	    data_pts[i,0] = pts[i,0,0]
	    data_pts[i,1] = pts[i,0,1]

	# Perform PCA analysis
	mean = np.empty((0))
	mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
	# Store the center of the object
	cntr = (int(mean[0,0]), int(mean[0,1]))

	cv2.circle(img, cntr, 4, (0, 255, 0), 1)

	# p1 = x1, y1
	# p2 = x2, y2
	p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
	p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])

	drawAxis(img, cntr, p1, (0, 255, 0), 1)
	drawAxis(img, cntr, p2, (0, 255, 0), 1)

	print()
	print("Eigenvectors: {}".format(eigenvectors))
	print()
	print("Eigenwerte: {}".format(eigenvalues))

	angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians

	return angle

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


# Convert image to grayscale
#gray = cv2.cvtColor(pic_res, cv2.COLOR_BGR2GRAY)

# Convert image to binary
_, bw = cv2.threshold(pic_res, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
## [pre-process]

contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for i, c in enumerate(contours):
	# Calculate the area of each contour
	area = cv2.contourArea(c)
	# Ignore contours that are too small or too large
	if area < 1e2 or 1e5 < area:
	    continue
	# Draw each contour only for visualisation purposes
	# cv2.drawContours(pic_res, contours, i, (0, 255, 0), 1)
	# Find the orientation of each shape
	getOrientation(c, pic_res)

cv2.imshow('output', pic_res)

pic_flip = cv2.flip(pic_res, 1)

cv2.imshow('output', pic_flip)

#print(eigenVectors)

cv2.waitKey(0)
cv2.destroyAllWindows()
