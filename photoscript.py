import os
import cv2
import math
import numpy
import imutils

def checkCorner(a, b, c):
	threshold = 5
	ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
	ang = ang + 360 if ang < 0 else ang
	ang = ang - 180 if ang > 180 else ang
	return abs(ang - 90) < threshold

def isAlmostRect(shape):
	if len(shape) != 4:
		return False

	cornersRightAngle = True
	for i in range(4):
		cornersRightAngle = cornersRightAngle and checkCorner(shape[i%4][0], shape[(i+1)%4][0], shape[(i+2)%4][0])
	return cornersRightAngle

def unwarp(image, pts):
	ysorted = sorted(pts, key=lambda x: x[0][1])
	tl = ysorted[0][0] if ysorted[0][0][0] < ysorted[1][0][0] else ysorted[1][0]
	tr = ysorted[1][0] if ysorted[0][0][0] < ysorted[1][0][0] else ysorted[0][0]
	br = ysorted[3][0] if ysorted[2][0][0] < ysorted[3][0][0] else ysorted[2][0]
	bl = ysorted[2][0] if ysorted[2][0][0] < ysorted[3][0][0] else ysorted[3][0]

	rect = numpy.float32((tl, tr, br, bl))

	widthA = numpy.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = numpy.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = numpy.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = numpy.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = numpy.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	unwarped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	
	return unwarped    

def checkRatio(image, pts):
	mn = 1.0
	mx = 3.0
	unwarped = unwarp(image, pts)
	ratio = max(unwarped.shape[0], unwarped.shape[1]) / min (unwarped.shape[0], unwarped.shape[1])
	return (ratio < mx and ratio > mn)

def auto_canny(image, sigma=0.33):
	v = numpy.median(image)
 
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	dilated = cv2.dilate(edged, numpy.ones((4,4), dtype=numpy.uint8))

	return dilated

def findAndShowRects(image, outputdir, filename):
	resizedsize = 1000
	resized = imutils.resize(image, height = resizedsize) if image.shape[0] > image.shape[1] else imutils.resize(image, width = resizedsize)
	ratio = image.shape[0] / float(resized.shape[0])

	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (11, 11), 0)
	edges = auto_canny(blurred)

	cv2.imshow("Image", edges)
	key = cv2.waitKey(0)	

	thresh_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]
	thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	thresh_mean = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 701, 10)
	thresh_gauss = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 701, 10)

	opening_bin = cv2.morphologyEx(thresh_bin, cv2.MORPH_OPEN, numpy.ones((2,2),numpy.uint8))
	opening_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, numpy.ones((2,2),numpy.uint8))
	opening_mean = cv2.morphologyEx(thresh_mean, cv2.MORPH_OPEN, numpy.ones((2,2),numpy.uint8))
	opening_gauss = cv2.morphologyEx(thresh_gauss, cv2.MORPH_OPEN, numpy.ones((2,2),numpy.uint8))
	
	cnts_find_bin = cv2.findContours(opening_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts_find_otsu = cv2.findContours(opening_otsu.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts_find_mean = cv2.findContours(opening_mean.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts_find_gauss = cv2.findContours(opening_gauss.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts_find_edges = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	cnts = imutils.grab_contours(cnts_find_edges)

	all_approx = []
	all_c = []

	for c in cnts:
		arclength = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * arclength, True)

		if len(approx) > 4:
			boxpoints = numpy.array(cv2.boxPoints(cv2.minAreaRect(approx)), dtype=int)
			approx = []
			for point in boxpoints:
				approx.append(numpy.array(point, dtype=int, ndmin=2))
			approx = numpy.array(approx)

		if len(approx) != 4:
			continue
		
		if not isAlmostRect(approx):
			continue

		if not checkRatio(resized, approx):
			continue

		if cv2.arcLength(approx, True) < (resized.shape[0] + resized.shape[1])/2:
			continue

		all_approx += [approx]
		all_c += [c]

	cv2.drawContours(resized, all_approx, -1, (0, 0, 255), 2)
	cv2.drawContours(resized, all_c, -1, (0, 255, 0), 1)		

	cv2.imshow("Image", resized)
	key = cv2.waitKey(0)

	write_file_success = False

	if key == 112:
		print("Processing and exporting image " + str(filename))
		for index, approx in enumerate(all_approx):
			approx = numpy.float32(approx) * ratio
			filepath = outputdir + "/" + filename + "_" + str(index) + ".JPG"
			unwarped = unwarp(image, approx)
			print("Writing file to: "+filepath)
			cv2.imwrite(filepath, unwarped)
			write_file_success = True
	else:
		print("Keeping image untouched.")

	return write_file_success

# Recursively read jpeg files from the root directory and process them.
# Limit number of files to the given value. Values less than or equal to 0 processes all files.
def readFiles(limit):
	counter = 0
	rootdir = "/Users/akshat/Desktop/Input/Gopikunj/Album1"
	outputdir = "/Users/akshat/Desktop/Output/Gopikunj/Album1"
	unprocessed_files = []
	for root, dirs, files in os.walk(rootdir):
	    jpegs = filter(lambda x: x.endswith(".JPG") and "DONE" not in x, files)
	    for jpeg in jpegs:
	    	if counter >= limit and limit > 0:
	    		return
	    	image = cv2.imread(root+"/"+jpeg)
	    	outputfolder = outputdir if (root == rootdir) else outputdir+"/"+root.replace(rootdir+"/", "")
	    	success = findAndShowRects(image, outputfolder, jpeg[:-4])
	    	if success:
	    		os.rename(root+"/"+jpeg, root+"/DONE_"+jpeg)
	    	else:
	    		unprocessed_files += [str(jpeg[:-4])]
	    	counter += 1

	print("\n\nUnprocessed files")
	print("*********************")
	print(unprocessed_files)
readFiles(0)