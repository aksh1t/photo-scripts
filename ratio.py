import os
import cv2
import math
import numpy
import imutils

# Recursively read jpeg files from the root directory and process them.
# Limit number of files to the given value. Values less than or equal to 0 processes all files.
def readFiles(limit):
	rootdir = "/Users/akshat/Desktop/Output/Gopikunj/Album1"

	mn = 1000
	mx = 0

	for root, dirs, files in os.walk(rootdir):
	    jpegs = filter(lambda x: x.endswith(".JPG"), files)
	    for jpeg in jpegs:
	    	image = cv2.imread(root+"/"+jpeg)
	    	l1 = image.shape[0]
	    	l2 = image.shape[1]
	    	ratio = max(l1, l2) / min(l1, l2)
	    	if ratio > mx:
	    		mx = ratio
	    	if ratio < mn:
	    		mn = ratio

	print (str(mx) + " min" + str(mn))

readFiles(0)