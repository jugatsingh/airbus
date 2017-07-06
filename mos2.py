import cv2
import numpy as np
from skimage import measure
from scipy import ndimage
from matplotlib import pyplot as plt

filename = 'image004.jpg'
im = cv2.imread(filename)
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# equ = cv2.equalizeHist(im_gray)
# blur = cv2.GaussianBlur(equ,(9,9),0)
thresh_val,im_thresh = cv2.threshold(im_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
	 
params.minThreshold = 0
params.maxThreshold = 220
# Filter by Area.
params.filterByArea = True
params.minArea = 1
params.maxArea = 40000
	 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = False
#params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1

# Distance Between Blobs
params.minDistBetweenBlobs = 3
	 
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs.
keypoints = detector.detect(im_thresh)
print(type(keypoints))
print('Total blobs:' + str(len(keypoints)))

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



cv2.imshow('original',im)
cv2.waitKey(0)
# cv2.imshow('equalised',equ)
# cv2.waitKey(0)
# cv2.imshow('blurred',blur)
cv2.imshow('thresh',im_thresh)
cv2.waitKey(0)
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()