
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tarfile
import os
import sys

img1="C:/Users/Hari/Downloads/FaceRecognition/Images/Test/13.jpg"
img2 ="WIN_20211123_08_00_13_Pro.jpg"
file_name='lfw-funneled.tgz'
person=cv2.imread(img1)


#Displaying Image
def show_image(image):
    plt.figure(figsize=(8,5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()

show_image(person)    

def face_detection(image):
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray)
    print('Number of faces detected:', len(faces))
        
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv_rgb

face_detection(person)
img1=cv2.imread("WIN_20211123_07_45_35_Pro.jpg")
img2 =cv2.imread("WIN_20211123_08_00_13_Pro.jpg")

modelFile ="res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
#net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def Singleface_Dimesion_Extraction(image,coord=False):
    blob = cv2.dnn.blobFromImage(image, 1,(224,224), [104, 117, 123], False, False)
    conf_threshold=0.8
    frameWidth=image.shape[1] 
    frameHeight=image.shape[0] 
    max_confidence=0
    detections = net.forward()
    detection_index=0
    bboxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            
            if max_confidence < confidence: # only show maximum confidence face
                max_confidence = confidence
                detection_index = i
    i=detection_index        
    x1 = int(detections[0, 0, i, 3] * frameWidth)
    y1 = int(detections[0, 0, i, 4] * frameHeight)
    x2 = int(detections[0, 0, i, 5] * frameWidth)
    y2 = int(detections[0, 0, i, 6] * frameHeight)
    cv2.rectangle(image,(x1,y1),(x2,y2),(255,255,0),2)
    cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if coord==True:
        return x1, y1, x2, y2
    return cv_rgb

image_person=Singleface_Dimesion_Extraction(person)
plt.imshow(image_person)
plt.show() 


print('unzipping the Images')

def jpg_files(members): #only extract jpg files
    for tarinfo in members:
        if os.path.splitext(tarinfo.name)[1] == ".jpg":
            yield tarinfo
def untar(fname,path="LFW"): #untarring the archive
    tar = tarfile.open(fname)
    tar.extractall(path,members=jpg_files(tar))
    tar.close()
    if path is "":
        print("File Extracted in Current Directory")
    else:
        print("File Extracted in to",  path)

untar(file_name,"LFW")

total = sum([len(files) for  files in os.walk('LFW/lfw_funneled/')])
 
print("Total number of FOlder alvailabel is",total)