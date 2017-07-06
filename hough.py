import cv2
import numpy as np

filename = 'image002.jpg'
img = cv2.imread(filename,0)
equ = cv2.equalizeHist(img)
laplacian = cv2.Laplacian(equ,cv2.CV_64F)
sharp = img - laplacian
blur = cv2.GaussianBlur(sharp,(5,5),0)
blur = blur.astype(np.uint8)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,80,
                            param1=128,param2=30,minRadius=10,maxRadius=80)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('equ',equ)
cv2.waitKey(0)
cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()