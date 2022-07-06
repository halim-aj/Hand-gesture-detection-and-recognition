# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:29:03 2020

@author: Halim
"""


import cv2
import numpy as np
import time
import tensorflow as tf

from phue import Bridge             #for the use of Phue Lights 


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

model = tf.keras.models.load_model("Reference_model_05_202212052022.model")

#Method of the list of categories
def allCategories():
    return ['rock', 'Palm', 'Fist', 'swing', 'L', 'okay', 'unknown']

#
def getCategory(list_val):
    CATEGORIES = allCategories()
    x= np.argmax(list_val)# Returns the indices of the maximum values along an axis
    print (np.max(list_val) * 100) #Returns the maximum values along an axis then multipling it to get a purcentage
    return CATEGORIES[x]

#input path of image and return ready preprocessed image
def procImage(image):
    image = image/255.
    image = cv2.resize(image,(100,120))
    image = image.reshape(1,image.shape[0],image.shape[1],1)
    return image

#Method of background subtraction
def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)#Matrix of 3*3 of ones
    fgmask = cv2.erode(fgmask, kernel, iterations=1)#dilatate the image
    res = cv2.bitwise_and(frame, frame, mask=fgmask)#operation "and" (The operation of "And" will be performed only if mask[i] doesn't equal zero, else the the result of and operation will be zero. The mask should be either white or black image with single channel) 
    return res


# Camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)
category = ''
thresh = np.empty ([384, 320], dtype = float)

while camera.isOpened():
    
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
   
    
    cv2.putText(frame, "prediction :"+ str(category) , (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

    
    #cv2.putText(frame, "Press 'b' to capture the background " , (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.imshow('original', frame)

    # Run once background is captured in a way to preprocess the input image
    if isBgCaptured == 1:
        #cv2.putText(frame, "Background is captured " , (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        img = remove_background(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        cv2.imshow('mask', img)

        # convert the image into binary image
      
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)#otsu binarisation after gaussian filtering (A Gaussian filter is a linear filter. It's usually used to blur the image or to reduce noise. If you use two of them and subtract, you can use them for "unsharp masking" (edge detection). The Gaussian filter alone will blur edges and reduce contrast.
                                                                #The Median filter is a non-linear filter that is most commonly used as a simple way to reduce noise in an image. It's claim to fame (over Gaussian for noise reduction) is that it removes noise while keeping edges relatively sharp.
                                                                #I guess the one advantage a Gaussian filter has over a median filter is that it's faster because multiplying and adding is probably faster than sorting. )
        cv2.imshow('blur', blur)
        
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    

        cv2.imshow('ori', thresh)
        cv2.putText(frame, "Background is captured " , (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

        
        # get the contours
        #cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #length = len(cnts)
        #maxArea = -1
        #if length > 0:
         #   for i in range(length):  # find the biggest contour (according to area)
          #      temp = cnts[i]
           #     area = cv2.contourArea(temp)
            #    if area > maxArea:
             #       maxArea = area
              #      ci = i
        

           # res = cnts[ci]
            #hull = cv2.convexHull(res)
            #drawing = np.zeros(img.shape, np.uint8)
            #cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            #cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
            

        #cv2.imshow('output', drawing)
        

    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27 & 0xFF:  # press ESC to exit all windows at any time
        break
    if k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        time.sleep(2)
        isBgCaptured = 1

        print('Background captured')
        #cv2.putText(frame, "Background is captured " , (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

        
        
        
        
    elif k == ord('r'):  # press 'r' to reset the background
         time.sleep(1)
         bgModel = None
         triggerSwitch = False
         isBgCaptured = 0
         print('Reset background')
    
        
    

        
    X = procImage(thresh)
    prediction = model.predict(X)
    #if isBgCaptured == 1:
    if k == 32:
        category = getCategory(prediction[0])
        print(category)
        #time.sleep(1)
    
    #bridge_ip = '192.168.0.100'
    #b = Bridge(bridge_ip)
    #b.connect()
    #b.get_api()

    #on_command =  {'transitiontime' : 0, 'on' : True, 'bri' : 30}
    #off_command =  {'transitiontime' : 0, 'on' : False, 'bri' : 30}


    
    #if category == 'Palm':
        
     #       action = b.set_light(1, on_command)
      #      cv2.putText(frame, "lights_on" , (50, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
            
    #else :
        
     #       action = b.set_light(1, off_command)
      #      cv2.putText(frame, "lights_off" , (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
            
    
    
        

        
camera.release()
cv2.destroyAllWindows()
