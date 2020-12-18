import numpy as np
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import sys
import argparse
import PIL
import pytesseract
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression

def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue
			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

def iterateC(contours, depth=0):
	s = "\n"
	for d in range(0,depth):
		s += "-"
	for c in contours:
		#print(type(c))
		if isinstance(c, np.int32):
			s += str(c) + ", "
		else:
			s += iterateC(c, depth + 1)
	return s

def resize(path, mode="cv2"):

	if mode == "PIL":
		PIL.Image.MAX_IMAGE_PIXELS = 2856110592

		im_pil = PIL.Image.open(path).convert('RGB')

		#Make the new image half the width and half the height of the original image
		im_pil = im_pil.resize((round(im_pil.size[0]/im_pil.size[1]*args["width"]), args["width"]))

		im_pil.save("temp.jpg")
		#im = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
		im = cv2.imread("temp.jpg")
		return im

	elif mode == "cv2":

		#os.system('export CV_IO_MAX_IMAGE_PIXELS=1099511627776')
		im = cv2.imread(args["image"])

		#im_ocr = im.copy()
		(origH, origW) = im.shape[:2]
		# set the new width and height and then determine the ratio in change
		# for both the width and height
		(newH, newW) = (int(origH/origW*args["width"]), args["width"])
		#rW = origW / float(newW)
		#rH = origH / float(newH)
		# resize the image and grab the new image dimensions
		im = cv2.resize(im, (newW, newH))
		#(H, W) = im_ocr.shape[:2]
		return im

def remove_section_map(im):
	im[:int(im.shape[0]*1 / 7),int(im.shape[1]*3 / 4):] = 255
	return im

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

	if brightness != 0:
		if brightness > 0:
			shadow = brightness
			highlight = 255
		else:
			shadow = 0
			highlight = 255 + brightness
		alpha_b = (highlight - shadow)/255
		gamma_b = shadow

		buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
	else:
		buf = input_img.copy()

	if contrast != 0:
		f = 131*(contrast + 127)/(127*(131-contrast))
		alpha_c = f
		gamma_c = 127*(1-f)

		buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

	return buf

def showImage(window, image, resize=True, waitKey=0):
	if resize:
		image = cv2.resize(image,(500,int(image.shape[0]/image.shape[1]*500)))
	cv2.imshow(window,image)
	cv2.waitKey(waitKey)

def detect_text(im, imshow="new"):
	im_orig = im.copy()
	im_proc = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	#(thresh, im_proc) = cv2.threshold(im_proc, 180, 255, cv2.THRESH_BINARY)
	im_proc = cv2.Canny(im_proc, 30, 200)


	#hsv = cv2.cvtColor(cv2.cvtColor(im,cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
	#hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	#lower = np.array([0, 0, 0])
	#upper = np.array([10, 50, 50])
	#mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([10,50,50])) + cv2.inRange(hsv, np.array([160,0,0]), np.array([360,50,50]))

	# Create horizontal kernel and dilate to connect text characters
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
	im_proc = cv2.dilate(im_proc, kernel, iterations=3)
	im_proc = cv2.bitwise_not(im_proc)
	showImage("proc",im_proc)


	# Find contours and filter using aspect ratio
	# Remove non-text contours by filling in the contour
	cnts, hierarchy = cv2.findContours(im_proc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_EXTERNAL
	#print(iterateC(cnts,0))
	#cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	#cnts_text = []
	#for c in cnts:
		#for c2 in c1:
		#x,y,w,h = cv2.boundingRect(c)
		#ar = w / float(h)
		#if ar < 5:
			#cv2.drawContours(im_orig, [c], -1, (0,255,0), 3)
			#cnts_text.append(c)
	# Bitwise dilated image with mask, invert, then OCR
	#result = 255 - cv2.bitwise_and(dilate, mask)

	res = []
	cnts_text = []
	cnts_notext = []
	for contour in cnts:#_final:
		[x, y, w, h] = cv2.boundingRect(contour)

		# Don't plot small false positives that aren't text
		if w < im.shape[1]/26.5 and h < im.shape[0]/26.5:
			cnts_notext.append(contour)
			continue
		if h > im.shape[1]/22.85:
			cnts_notext.append(contour)
			continue

		cnts_text.append(contour)

		pos = (x+w/2,y+h/2)
		rect = (x,y,w,h)

		seg = im[y:y+h,x:x+w]
		(origH, origW) = seg.shape[:2]
		(newH, newW) = (int(origH/origW*256), 256)
		seg = cv2.resize(seg, (newW, newH))
		seg = cv2.cvtColor(seg,cv2.COLOR_BGR2GRAY)
		#mean, std = cv2.meanStdDev(seg)
		#mean = mean[0][0]
		seg = cv2.GaussianBlur(seg,(5,5),0)
		(thresh, seg) = cv2.threshold(seg, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		seg = cv2.dilate(seg, np.ones((2,2),np.uint8),iterations = 6)

		#seg = cv2.adaptiveThreshold(seg,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,25,3)
		#(thresh, seg) = cv2.threshold(seg, 110, 255, cv2.THRESH_BINARY) #219=145, erode?

		#seg = cv2.medianBlur(seg,5)
		#seg = apply_brightness_contrast(seg, 0, 10)
		#print(mean)
		d = pytesseract.image_to_string(seg)

		print(d.strip())
		cv2.imshow("seg",seg)
		cv2.waitKey(0)

		res.append({"text":d.strip(), "pos":pos, "rect":rect, "contour":contour})

	for t in res:
		print(t["text"])

	# draw rectangle around contour on original image
	if imshow == "new":
		imshow = im.copy()
	for contour in cnts_text:
		[x, y, w, h] = cv2.boundingRect(contour)
		cv2.rectangle(imshow, (x, y), (x + w, y + h), (255, 0, 255), 2)

	return res, imshow, cnts_notext, hierarchy

def detect_contours(im, imshow="new", ignoreContours=None):
	if isinstance(imshow, str) and imshow == "new":
		imshow = im.copy()
	#if contours is None:
	imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	#ret,thresh = cv2.threshold(imgray,127,255,0)
	im_proc = cv2.Canny(imgray, 30, 200)

	im_proc = cv2.dilate(im_proc, np.ones((2,2),np.uint8), iterations=3)
	showImage("proc",im_proc)

	contours, hierarchy = cv2.findContours(im_proc,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


	print(iterateC(contours, 0))
	#print(im)
	ignored = []

	for ci in ignoreContours:
		#tmpshow = imshow.copy()
		for i in range(0, len(contours)):
			c = contours[i]
			#[x, y, w, h] = cv2.boundingRect(ci)
			#cv2.rectangle(tmpshow, (x, y), (x + w, y + h), (255, 0, 255), 2)
			#[x, y, w, h] = cv2.boundingRect(c)
			#cv2.rectangle(tmpshow, (x, y), (x + w, y + h), (255, 0, 0), 2)


			per = percentRectinRect(cv2.boundingRect(c),cv2.boundingRect(ci),6,2)
			#if per > 0.01:
			#	print(per)


			if per > 0.08:
				cv2.drawContours(imshow, c, -1, (0, 0, 255), 5)
				#cv2.drawContours(tmpshow, c, -1, (0, 0, 255), 5)

				ignored.append(i)
				#print(per)
				#break
		#showImage("calcing overlaps...",tmpshow,waitKey=0)


	if len(ignored) > 0:
		ignored.sort()
		ignored.reverse()
	for i in ignored:
		contours.pop(i)

	cv2.drawContours(imshow, contours, -1, (0, 255, 0), 3)
	return imshow

def getRectBoundary(rect):
	b = []
	[x, y, h, w] = rect
	for x in range(x,x+w):
		for y in range(y,y+h):
			b.append((x,y))
	return b

def pointInRect(point, rect, padding=0):
	(px, py) = point
	[x, y, h, w] = rect
	if px >= x-padding/2 and px <= x+w+padding/2 and py >= y-padding/2 and py <= y+h+padding/2:
		return True
	return False

def percentRectinRect(test, compare, padding=10, skip=1):
	tb = getRectBoundary(test)
	finds = 0
	for i in range(0,len(tb),skip):
		tp = tb[i]
		if pointInRect(tp,compare,padding):
			finds += 1
	return (finds*skip)/len(tb)

if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", type=str,
		help="path to input image")
	ap.add_argument("-w", "--width", type=int, default=1600,
		help="nearest multiple of 32 for resized width")
	ap.add_argument("-p", "--padding", type=float, default=0.0,
		help="amount of padding to add to each border of ROI")
	args = vars(ap.parse_args())

	im = resize(args["image"],mode="cv2")
	im = remove_section_map(im)

	im_orig = im.copy()

	res, imshow, cnts, hier = detect_text(im)
	textContours = []
	for d in res:
		textContours.append(d["contour"])

	imshow = detect_contours(im, imshow, ignoreContours=textContours) #cnts, hier)
	showImage("detection final",imshow)
