
import cv2 #for image processing
import easygui #to open the filebox
import numpy as np #to store image
import imageio #to read image stored at particular path
import sys
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import Simple_Face_Extracter as sfe
import Face_detection as fd

#"C:/Users/Hari/Downloads/FaceRecognition/Images/Test/6.jpg"

def upload():
    ImagePath=easygui.fileopenbox()
    cartoonify(ImagePath)

def cartoonify(ImagePath):
    #read the image
    if originalmage is None:
        return
    originalmage = cv2.imread(ImagePath)
    originalmage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2RGB)
    #cv2.imshow('cartoon', originalmage)
    #cv2.waitKey()
    
    if originalmage is None:
        print("Can not find any image. Choose appropriate file")
        sys.exit()

    ReSized1 = cv2.resize(originalmage, (960, 540))
    plt.imshow(ReSized1, cmap='gray')
    grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
    ReSized2 = cv2.resize(grayScaleImage, (960, 540))
    #cv2.imshow('cartoon', grayScaleImage)
    #cv2.waitKey()
    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
    ReSized3 = cv2.resize(smoothGrayScale, (960, 540))
    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, 
    cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, 9, 9)
    #cv2.imshow('cartoon', getEdge)
    #cv2.waitKey()
    ReSized4 = cv2.resize(getEdge, (960, 540))
    colorImage = cv2.bilateralFilter(originalmage, 9, 300, 300)
    #cv2.imshow('cartoon', colorImage)
    #cv2.waitKey()
    ReSized5 = cv2.resize(colorImage, (960, 540))
    cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
    #cv2.imshow('cartoon', cartoonImage)
    #cv2.waitKey()
    ReSized6 = cv2.resize(cartoonImage, (960, 540))
    images=[ReSized1, ReSized2, ReSized3, ReSized4, ReSized5, ReSized6]
    fig, axes = plt.subplots(3,2, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
    plt.show()

def save(ReSized6, ImagePath):
    #saving an image using imwrite()
    newName="cartoonified_Image"
    path1 = os.path.dirname(ImagePath)
    extension=os.path.splitext(ImagePath)[1]
    path = os.path.join(path1, newName+extension)
    cv2.imwrite(path, cv2.cvtColor(ReSized6, cv2.COLOR_RGB2BGR))
    I = "Image saved by name " + newName +" at "+ path
    tk.messagebox.showinfo(title=None, message=I)

def face_extracter():
  f=  sfe.Face_extracter() 
  f.Capture_face()
  
def face_detect():
  f=  fd.Facedetect() 
  f.detectface_fromimage()

top=tk.Tk()
top.geometry('800x800')
top.title('Face Recognition !')
top.configure(background='white')
label=Label(top,background='#CDCDCD', font=('calibri',20,'bold'))

upload=Button(top,text="Cartoonify an Image",command=upload,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('calibri',10,'bold'))
upload.pack(side=TOP,pady=50)

save=Button(top,text="Save cartoon image",command=lambda: save("C:/Users/Hari/Documents/Save", ReSized6),padx=20,pady=5)
save.configure(background='#364156', foreground='white',font=('calibri',10,'bold'))
save.pack(side=TOP,pady=50)

face_recognize=Button(top,text="Capture Image",command = face_extracter,padx=30,pady=5)
face_recognize.configure(background='#364156', foreground='white',font=('calibri',10,'bold'))
face_recognize.pack(side=TOP,pady=50)

Face_detect=Button(top,text="Detect Image",command = face_detect,padx=40,pady=5)
Face_detect.configure(background='#364156', foreground='white',font=('calibri',10,'bold'))
Face_detect.pack(side=TOP,pady=50)

top.mainloop()
upload()