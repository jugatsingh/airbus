import cv2
import numpy as np
from skimage import measure
from scipy import ndimage
from matplotlib import pyplot as plt

filename = 'image007.jpg'
im = cv2.imread(filename,0)

equ = cv2.equalizeHist(im)
blur = cv2.GaussianBlur(equ,(5,5),0)

thresh_val,im_thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
labels = measure.label(blur)
# print(labels.max())

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# laplacian = cv2.Laplacian(blur,cv2.CV_64F)
# drops = ndimage.binary_fill_holes(laplacian)
# opening = cv2.morphologyEx(equ, cv2.MORPH_OPEN, kernel)

# cv2.convertScaleAbs( laplacian, laplacian)
# des = 255 - laplacian
# cv2.imshow('inverted edge detected',des)
# cv2.waitKey(0)


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
# cv2.imshow('edge enhanced',laplacian)
# cv2.waitKey(0)
# cv2.imshow('holes filled',gray)
# cv2.waitKey(0)
# cv2.imshow('opening',opening)
# cv2.waitKey(0)
# cv2.imshow('threshold',im_thresh)
# cv2.waitKey(0)
# Show keypoints
cv2.imshow("Keypoints", blur_with_keypoints)
cv2.waitKey(0)
cv2.imshow("Keypoints on original", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()