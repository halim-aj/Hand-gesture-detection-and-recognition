# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:49:02 2019

@author: Halim
"""

import cv2
import numpy as np
import time
import copy
import os


# parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
prev = 0 
frame_rate = 10

# variableslt
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyboard simulator works

# Create the directory structure
if not os.path.exists("dataset"):
    os.makedirs("dataset")
    os.makedirs("dataset/train")
    os.makedirs("dataset/test")
    os.makedirs("dataset/train/Palm")
    os.makedirs("dataset/train/Fist")
    os.makedirs("dataset/train/L")
    os.makedirs("dataset/train/rock")
    os.makedirs("dataset/train/swing")
    os.makedirs("dataset/train/okay")
    os.makedirs("dataset/train/unknown")
    os.makedirs("dataset/test/Palm")
    os.makedirs("dataset/test/Fist")
    os.makedirs("dataset/test/L")
    os.makedirs("dataset/test/rock")
    os.makedirs("dataset/test/swing")
    os.makedirs("dataset/test/okay")
    os.makedirs("dataset/test/unknown")
    
# Train or test 
mode = 'test'
directory = 'dataset/'+mode+'/'

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)#learning = learning zeyda
    kernel = np.ones((3, 3), np.uint8)#matrice 3*3 de 1
    fgmask = cv2.erode(fgmask, kernel, iterations=1)#tghaladh f taswira
    res = cv2.bitwise_and(frame, frame, mask=fgmask)#operation "and" (The operation of "And" will be performed only if mask[i] doesn't equal zero, else the the result of and operation will be zero. The mask should be either white or black image with single channel) 
    return res
# Camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)

while camera.isOpened():
    time_elapsed = time.time() - prev 
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    if time_elapsed > 2./frame_rate:
        prev = time.time()
        

    
    
        # Getting count of existing images
    count = {'Palm': len(os.listdir(directory+"/Palm")),
             'Fist': len(os.listdir(directory+"/Fist")),
             'L': len(os.listdir(directory+"/L")),
             'rock': len(os.listdir(directory+"/rock")),
             'swing': len(os.listdir(directory+"/swing")),
             'unknown': len(os.listdir(directory+"/unknown")),
             'okay': len(os.listdir(directory+"/okay"))}
               
        # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "Palm : "+str(count['Palm']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "Fist : "+str(count['Fist']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "L : "+str(count['L']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "rock : "+str(count['rock']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "swing : "+str(count['swing']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "okay : "+str(count['okay']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "unknown : "+str(count['unknown']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    
    cv2.imshow('original', frame)

    # Run once background is captured
    if isBgCaptured == 1:
        img = remove_background(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow('blur', blur)
                                                                
        # cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imshow('ori', thresh)

        # get the contours
       ######## zeyda tnajam tetna7a ama eli mba3dha fel findcontours lezem twali thresh1 ##thresh1 = copy.deepcopy(thresh)
        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(cnts)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = cnts[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = cnts[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

        cv2.imshow('output', drawing)

    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit all windows at any time
        break
    if k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
       #### b.set_light(6, on_command)
        time.sleep(2)
        isBgCaptured = 1
        print('Background captured')
    if k & 0xFF == ord('p'):
        cv2.imwrite(directory+'Palm/'+str(count['Palm'])+'.jpg', thresh)
    if k & 0xFF == ord('f'):
        cv2.imwrite(directory+'Fist/'+str(count['Fist'])+'.jpg', thresh)
    if k & 0xFF == ord('l'):
        cv2.imwrite(directory+'L/'+str(count['L'])+'.jpg', thresh)
    if k & 0xFF == ord('r'):
        cv2.imwrite(directory+'rock/'+str(count['rock'])+'.jpg', thresh)
    if k & 0xFF == ord('s'):
        cv2.imwrite(directory+'swing/'+str(count['swing'])+'.jpg', thresh)
    if k & 0xFF == ord('o'):
        cv2.imwrite(directory+'okay/'+str(count['okay'])+'.jpg', thresh) #cv2.imwrite(directory+'okay/'+str(count['five'])+'.jpg', thresh)
    if k & 0xFF == ord('u'):
        cv2.imwrite(directory+'unknown/'+str(count['unknown'])+'.jpg', thresh)
camera.release()
cv2.destroyAllWindows()

