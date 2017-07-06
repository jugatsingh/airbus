import cv2
import numpy as np
from skimage import measure
from scipy import ndimage
from matplotlib import pyplot as plt

filename = 'image001.jpg'
im = cv2.imread(filename,0)
equ = cv2.equalizeHist(im)
blur = cv2.GaussianBlur(equ,(5,5),0)

#edge detection
laplacian = cv2.Laplacian(blur,cv2.CV_64F)
dest = np.zeros(laplacian.shape,dtype = np.uint8)
cv2.convertScaleAbs(laplacian,dest)
sharp_im = im + dest
print(sharp_im)

# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
	 
#filter by color
params.filterByColor = False
params.blobColor = 0

# Filter by Area.
params.filterByArea = True
params.minArea = 200
params.maxArea = 40000
	 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5
 
# Filter by Convexity
params.filterByConvexity = False
#params.minConvexity = 0.87
	 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8

# Distance Between Blobs
params.minDistBetweenBlobs = 2
	 
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs.
keypoints = detector.detect(sharp_im)
print(type(keypoints))
print('Total blobs:' + str(len(keypoints)))
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
# thresh_val,im_thresh = cv2.threshold(dest,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#disp edges
cv2.imshow('edge enhanced',sharp_im)
cv2.waitKey(0)
# print(dest)
# print('\n\n')
# print(im_thresh)
# cv2.imshow('edge enhanced corrected',dest)
# cv2.waitKey(0)
# cv2.imshow('edge enhanced thresh',im_thresh)
# cv2.waitKey(0)


