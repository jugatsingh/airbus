import cv2
import numpy as np
from skimage import measure
from scipy import ndimage
from matplotlib import pyplot as plt

filename = 'image002.jpg'
im = cv2.imread(filename,0)

edges = cv2.Canny(im,100,200)
plt.subplot(121),plt.imshow(im,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

equ = cv2.equalizeHist(im)
blur = cv2.GaussianBlur(equ,(7,7),0)
laplacian = cv2.Laplacian(blur,cv2.CV_64F)
im_thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,7,2)

edge = blur + laplacian
# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
	 
params.minThreshold = 0
params.maxThreshold = 220
# Filter by Area.
params.filterByArea = True
params.minArea = 10
params.maxArea = 400
	 
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



#image outputs
cv2.imshow('original',im)
cv2.waitKey(0)
cv2.imshow('equalised',equ)
cv2.waitKey(0)
cv2.imshow('blurred',blur)
cv2.waitKey(0)
cv2.imshow('thresh',im_thresh)
cv2.waitKey(0)
cv2.imshow('edge',edge)
cv2.waitKey(0)
# Show keypoints
cv2.imshow("Keypoints", blur_with_keypoints)
cv2.waitKey(0)
# cv2.imshow("Keypoints on original", im_with_keypoints)
# cv2.waitKey(0)
cv2.destroyAllWindows()