import cv2
import numpy as np
from skimage import measure
from scipy import ndimage
from matplotlib import pyplot as plt

filename = 'image003.jpg'
im = cv2.imread(filename)
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(im_gray)
blur = cv2.GaussianBlur(equ,(9,9),0)

im_thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,7,2)
# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
	 
params.minThreshold = 0
params.maxThreshold = 220
# Filter by Area.
params.filterByArea = True
params.minArea = 60
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
params.minDistBetweenBlobs = 0.5
	 
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs.
keypoints = detector.detect(blur)
print(type(keypoints))
print('Total blobs:' + str(len(keypoints)))
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
blur_with_keypoints = cv2.drawKeypoints(blur, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



#image outputs
cv2.imshow('original',im)
cv2.waitKey(0)
cv2.imshow('equalised',equ)
cv2.waitKey(0)
cv2.imshow('blurred',blur)
cv2.waitKey(0)
cv2.imshow('thresh',im_thresh)
cv2.waitKey(0)
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()