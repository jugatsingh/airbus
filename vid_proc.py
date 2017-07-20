import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('/home/mbt/Dropbox/Airbus_Videos/Video 1_A350-0059-Flight-0009-150630-150830-LHWING2_1.mp4')
i = 0
while(i<300):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i +=1
im_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,7,2)
equ = cv2.equalizeHist(gray)
inv = cv2.bitwise_not(im_thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
erosion = cv2.erode(inv, kernel, iterations = 1)
erosion = cv2.bitwise_not(erosion)
# # Set up the detector with default parameters.
# params = cv2.SimpleBlobDetector_Params()
# params.minThreshold = 0
# params.maxThreshold = 220
# # Filter by Area.
# params.filterByArea = True
# params.minArea = 60
# params.maxArea = 40000
	 
# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1
 
# # Filter by Convexity
# params.filterByConvexity = False
# #params.minConvexity = 0.87

# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.1

# # Distance Between Blobs
# params.minDistBetweenBlobs = 0.5
	 
# # Create a detector with the parameters
# detector = cv2.SimpleBlobDetector_create(params)
# # Detect blobs.
# keypoints = detector.detect(equ)
# print(type(keypoints))
# print('Total blobs:' + str(len(keypoints)))
# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# equ_with_keypoints = cv2.drawKeypoints(equ, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



# gray = cv2.GaussianBlur(gray,(9,9),0)
cv2.imshow('original',gray)
cv2.waitKey(0)
cv2.imshow('equalised',equ)
cv2.waitKey(0)
cv2.imshow("thresh", im_thresh)
cv2.waitKey(0)
cv2.imshow("erosion", erosion)
cv2.waitKey(0)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


im = im_thresh
# unsharp = equ - blur


dft = cv2.dft(np.float32(im),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)


magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(im, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()




mean_filter = np.ones((3,3))

rows, cols = im.shape
crow,ccol = rows/2 , cols/2

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-100:crow+100, ccol-100:ccol+100] = 1
# apply mask and inverse DFT
fft_filters = np.fft.fft2(mean_filter)
fshift = np.fft.ifftshift(fft_filters)

# fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
magnitude_spectrum = 20*np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1]))
plt.subplot(121),plt.imshow(img_back, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
img_back = img_back.astype(np.uint8)

im_thresh_2 = cv2.adaptiveThreshold(img_back,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,7,2)
plt.imshow(im_thresh_2,cmap='gray')
plt.show()
cv2.imshow("thresh", im_thresh_2)
cv2.waitKey(0)
cv2.destroyAllWindows()