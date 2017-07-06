import cv2
import numpy as np
from skimage import measure
from scipy import ndimage
from matplotlib import pyplot as plt

filename = 'image001.jpg'
im = cv2.imread(filename,0)
equ = cv2.equalizeHist(im)
blur = cv2.GaussianBlur(equ,(5,5),0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
unsharp = equ - blur

print(unsharp)
cv2.imshow('original',im)
cv2.waitKey(0)
cv2.imshow('blur',blur)
cv2.waitKey(0)
cv2.imshow('unsharp',unsharp)
cv2.waitKey(0)
dest = np.zeros(unsharp.shape,dtype = np.uint8)
unsharp = cv2.bitwise_not(unsharp)
erosion = cv2.erode(unsharp,kernel,iterations=1)
# erosion = cv2.bitwise_not(erosion)
# _,contour,hier = cv2.findContours(erosion,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

# for cnt in contour:
#     cv2.drawContours(dest,[cnt],0,255,2)
# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
	 
params.minThreshold = 0
params.maxThreshold = 178
# Filter by Area.
params.filterByArea = True
params.minArea = 10
params.maxArea = 40000
	 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.2
 
# Filter by Convexity
params.filterByConvexity = False
#params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.2
	 
# Distance Between Blobs
params.minDistBetweenBlobs = 1
	 
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs.
keypoints = detector.detect(erosion)
print(type(keypoints))
print('Total blobs:' + str(len(keypoints)))
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('contours',dest)
cv2.waitKey(0)

cv2.imshow('eroded',erosion)
cv2.waitKey(0)

#edge detection
# laplacian = cv2.Laplacian(blur,cv2.CV_64F)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()