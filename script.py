#!/usr/bin/python2
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('wing.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template1 = cv2.imread('shoelace1.png',0)
template2 = cv2.imread('shoelace2.png',0)
w1, h1 = template1.shape[::-1]
w2, h2 = template2.shape[::-1]

res1 = cv2.matchTemplate(img_gray,template1,cv2.TM_CCOEFF_NORMED)
res2 = cv2.matchTemplate(img_gray,template2,cv2.TM_CCOEFF_NORMED)

print(type(res2))
threshold = 0.95

loc1 = np.where( res1 >= threshold)

for pt in zip(*loc1[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w1, pt[1] + h1), (0,0,255), 2)

threshold = 0.8

loc2 = np.where( res2 >= threshold)

loc2 = [pt for pt in loc2 if pt not in loc1]

for pt in zip(*loc2[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w2, pt[1] + h2), (0,0,255), 2)


#cv2.imshow('res.png',img_rgb)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imwrite('res.png',img_rgb)
