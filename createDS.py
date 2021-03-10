# Face Recognition 
import cv2
import numpy as np
# Source to default webcam
cap = cv2.VideoCapture(0)

# Calls haar cascade classifier
# *save the file in the same folder
fc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
id=input("Enter a id : ")
Sn=0;
while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    # Haar works on greyscale image so we need to convert the image to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Returns x,y position of the detected face and width and height as well 
    faces = fc.detectMultiScale(gray, 1.3, 5)
    # Display the rectangle on the face
    for (x,y,w,h) in faces:
         Sn=Sn+1;
         cv2.imwrite("data/User."+str(id)+"."+str(Sn)+".jpg",gray[y:y+h,x:x+w])
         cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
         cv2.waitKey(100);
    # Displays the captured image from camera with the face
    cv2.imshow('face detection',img);
    # Break when q is pressed
    cv2.waitKey(1);
    if(Sn>20):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
