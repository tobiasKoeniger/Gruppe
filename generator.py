#!/usr/bin/python3

# commit and push to git with: git add . ; git commit -m "Next commit3" ; git push origin master

from os import listdir
import numpy as np

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

pic0 = (255-img[0])
pic1 = (255-img[1])
pic_res = pic0 + pic1
pic_res = (255-pic_res)

cv2.imshow('Image 0', pic0)
cv2.imshow('Image 1', pic1)
cv2.imshow('Image res', pic_res)	

cv2.waitKey(0)
cv2.destroyAllWindows()	


