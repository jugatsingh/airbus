#!/usr/bin/python2
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def l2_dist_check(pt, loc, dist):
    for pt1 in loc:
        if math.sqrt(((pt1[0]-pt[0])**2) + ((pt1[1]-pt[1])**2)) < dist:
            return (False, pt1)
    return True,1

width = 1680
height = 1050
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1680,1050))
cap = cv2.VideoCapture('/home/mbt/Dropbox/Airbus_Videos/Video 1_A350-0059-Flight-0009-150630-150830-LHWING2_1.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX
detected_markers = {}
lookup_table = []
# frame = cv2.imread('wing.png',1)
# gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
i = 0
n = -1.5
step = 0.0
#loop over the frames
while(True):
    count_red = 0
    count_green = 0

    i += 1
    if i%75 == 0:
        for pt in detected_markers:
            if detected_markers[pt] > 50:
                if l2_dist_check(pt, lookup_table, 15)[0] == True:
                    lookup_table.append(pt)
        detected_markers = {}
        # print(lookup_table)

    ret, frame = cap.read()
    if ret == False: #logic for end of video
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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


    threshold1 = 0.99
    threshold3 = 0.9
    threshold2 = 0.8

    loc1 = np.where( res1 >= threshold1)
    loc2 = np.where( res2 >= threshold2)
    loc3 = np.where( res3 >= threshold3)
    loc1 = [pt for pt in zip(*loc1[::-1])]
    loc2 = [pt for pt in zip(*loc2[::-1])]
    loc3 = [pt for pt in zip(*loc3[::-1])]


    temp = loc3[:]
    for x in range(len(loc3)):
        if l2_dist_check(loc3[x],loc3[x+1:], 10)[0] != True:
            temp.remove(loc3[x])
    loc3 = temp[:]

    # temp = lookup_table[:]
    # for x in range(len(lookup_table)):
    #     if l2_dist_check(lookup_table[x],lookup_table[x+1:], 3)[0] != True:
    #         temp.remove(lookup_table[x])
    # lookup_table = temp[:]


    # loc3 = [pt for pt in zip(*loc3[::-1]) if l2_dist_check(pt,loc3,3) == True]
    loc3 = [pt for pt in loc3 if l2_dist_check(pt, loc2, 50)[0] == True]
    loc3 = [pt for pt in loc3 if l2_dist_check(pt, loc1, 50)[0] == True]
    loc2 = [pt for pt in loc2 if l2_dist_check(pt, loc1, 15)[0] == True]

    #update detected_markers for every frame
    for pt in loc1:
        res = l2_dist_check(pt, list(detected_markers.keys()), 15)
        if res[0] == True:
            detected_markers[pt] = 1
        else:
            detected_markers[res[1]] += 1
    for pt in loc2:
        res = l2_dist_check(pt, list(detected_markers.keys()), 15)
        if res[0] == True:
            detected_markers[pt] = 1
        else:
            detected_markers[res[1]] += 1
    for pt in loc3:
        res = l2_dist_check(pt, list(detected_markers.keys()), 15)
        if res[0] == True:
            detected_markers[pt] = 1
        else:
            detected_markers[res[1]] += 1

    #update lookup table for every frame based on std. deviation
    # for pt in detected_markers:
    #     # print(detected_markers[pt])
    #     if detected_markers[pt] > (np.mean(list(detected_markers.values())) + n*np.std(list(detected_markers.values()))):
    #         lookup_table.append(pt)

    #snippet for drawing on every frame
    # print(len(lookup_table))
    for pt in lookup_table:
        res1 = l2_dist_check(pt, loc1, 12)
        res2 = l2_dist_check(pt, loc2, 12)
        res3 = l2_dist_check(pt, loc3, 12)
        if res1[0] == False:
            cv2.rectangle(frame, res1[1], (res1[1][0] + w1, res1[1][1] + h1), (0,255,0), 2)    #print
            count_green += 1
        elif res2[0] == False:
            cv2.rectangle(frame, res2[1], (res2[1][0] + w2, res2[1][1] + h2), (0,255,0), 2)    #green
            count_green += 1
        elif res3[0] ==False:
            cv2.rectangle(frame, res3[1], (res3[1][0] + w3, res3[1][1] + h3), (0,255,0), 2)    #markers
            count_green += 1
        else:
            cv2.rectangle(frame, pt, (pt[0] + w2, pt[1] + h2), (0,0,255), 2)    #print red markers
            count_red += 1

    # if len(loc2)<150:

    #     cv2.putText(frame,'EXTREME TURBULENCE!',(int(frame.shape[0]/2)-200,int(frame.shape[1]/2)-80), font, 4,(0,0,255),2,cv2.LINE_AA)
    # elif len(loc2)<220:
    #     cv2.putText(frame,'TURBULENCE!',(int(frame.shape[0]/2)-50,int(frame.shape[1]/2)-80), font, 4,(0,0,255),2,cv2.LINE_AA)
    cv2.rectangle(frame,(width - 500,110),(width - 200,380),(255,255,255),2)
    cv2.putText(frame,'Normal',(width - 480,150), font, 1,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(frame,'Deviated',(width - 480,250), font, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'Total',(width - 480,350), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame,str(count_green),(width - 300, 150), font, 1,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(frame,str(count_red),(width - 300, 250), font, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,str(count_green + count_red),(width - 300, 350), font, 1,(255,255,255),2,cv2.LINE_AA)
    # cv2.imshow('frame',frame)
    out.write(frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #cv2.imwrite('res.png',frame)
cap.release()
out.release()
