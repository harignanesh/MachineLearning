import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2 as cv
import os
from time import time
from windowcapture import WindowCapture



mobile_cassade = cv.CascadeClassifier("C:/Users/Hari/source/repos/Face_recognition/Face_recognition/cascade.xml")
mobie_img_1 =  cv.imread("C:/Users/Hari/source/repos/Face_recognition/Face_recognition/IMG_20211201_071713.jpg")
mobie_img_2 =  cv.imread("C:/Users/Hari/Downloads/FaceRecognition/IMG_20211201_071804.jpg")
cv.namedWindow('img',cv.WINDOW_NORMAL)

cv.imshow('img',mobie_img_1)
gray_img_1 = cv.cvtColor(mobie_img_1,cv.COLOR_BGR2GRAY)

cascade = mobile_cassade.detectMultiScale(gray_img_1,1.01,7)

for(x,y,w,h) in cascade:
    mobie_img_1 = cv.rectangle(mobie_img_1,(0,0),(x+w,y+h),(255,0,0),2)

cv.imshow('img',mobie_img_1)
cv.waitKey(0)
cv.destroyAllWindows()