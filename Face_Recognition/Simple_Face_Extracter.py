import numpy as np 
import pandas as pd 
import cv2

class Face_extracter:

    # Load functions

    def face_extractor(self,img):
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

        faces = face_classifier.detectMultiScale(img, 1.3, 5)
        if faces is ():
            return None
    
        for (x,y,w,h) in faces:
            x=x-10
            y=y-10
            cropped_face = img[y:y+h+50, x:x+w+50]

        return cropped_face

    

    def Capture_face(self):
      #Capturing Images
      cap = cv2.VideoCapture(0)
      count = 0
      while True:

            ret, frame = cap.read()
            if fe.face_extractor(frame) is not None:
                count += 1
                face = cv2.resize(fe.face_extractor(frame), (400, 400))
                file_name_path = 'C:/Users/Hari/Downloads/FaceRecognition/Images/' + str(count) + '.jpg'
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Cropper', face)
            else:
                print("Face not found")
                pass
            if cv2.waitKey(1) == 13 or count == 100:
                cap.release()
                cv2.destroyAllWindows()      
                print("Collecting Samples Complete")
                break
        


fe = Face_extracter()
