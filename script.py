#!/usr/bin/python2
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def l2_dist_check(pt, loc, dist):
	for pt1 in zip(*loc[::-1]):
		if math.sqrt(((pt1[0]-pt[0])**2) + ((pt1[1]-pt[1])**2)) < dist:
			return False
	return True

fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1680,1050))
cap = cv2.VideoCapture('/home/mbt/Dropbox/Airbus_Videos/Video 1_A350-0059-Flight-0009-150630-150830-LHWING2_1.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX
# frame = cv2.imread('wing.png',1)
# gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
i = 0
while(True):
    ret, frame = cap.read()
    if ret == False:
    	break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break
    template1 = cv2.imread('shoelace1.png',0)
    template2 = cv2.imread('shoelace2.png',0)
    template3 = cv2.imread('shoelace3.png',0)
    w1, h1 = template1.shape[::-1]
    w2, h2 = template2.shape[::-1]
    w3, h3 = template3.shape[::-1]
    res1 = cv2.matchTemplate(gray,template1,cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(gray,template2,cv2.TM_CCOEFF_NORMED)
    res3 = cv2.matchTemplate(gray,template3,cv2.TM_CCOEFF_NORMED)

    # print(type(res2))
    threshold1 = 0.97
    threshold3 = 0.9
    threshold2 = 0.8

    loc1 = np.where( res1 >= threshold1)
    loc2 = np.where( res2 >= threshold2)
    loc3 = np.where( res3 >= threshold3)

    for pt in zip(*loc1[::-1]):
    	cv2.rectangle(frame, pt, (pt[0] + w1, pt[1] + h1), (0,255,0), 2)
    
    loc3 = [pt for pt in zip(*loc3[::-1]) if l2_dist_check(pt, loc2, 50) == True]
    loc3 = [pt for pt in loc3 if l2_dist_check(pt, loc1, 50) == True]
    loc2 = [pt for pt in zip(*loc2[::-1]) if l2_dist_check(pt,loc1,15) == True]

    for pt in loc2:
    	cv2.rectangle(frame, pt, (pt[0] + w2, pt[1] + h2), (0,255,0), 2)
    
    for pt in loc3:
    	cv2.rectangle(frame, pt, (pt[0] + w3, pt[1] + h3), (0,255,0), 2)

    if len(loc2)<150:

    	cv2.putText(frame,'EXTREME TURBULENCE!',(int(frame.shape[0]/2)-200,int(frame.shape[1]/2)-80), font, 4,(0,0,255),2,cv2.LINE_AA)
    elif len(loc2)<220:
    	cv2.putText(frame,'TURBULENCE!',(int(frame.shape[0]/2)-50,int(frame.shape[1]/2)-80), font, 4,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,str(len(loc2[0])),(int(frame.shape[0])-150, 200), font, 4,(255,255,255),2,cv2.LINE_AA)
    # print(len(loc1[0]),len(l))
    # cv2.imshow('frame',frame)
    out.write(frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #cv2.imwrite('res.png',frame)
cap.release()
out.release()
