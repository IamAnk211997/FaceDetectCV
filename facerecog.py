import cv2
import numpy as np
# Source to default webcam
cap = cv2.VideoCapture(0)
fc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\tranningData.yml")
id=0
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 2
fontcolor = (0,255,0)
while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    # Haar works on greyscale image so we need to convert the image to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Returns x,y position of the detected face and width and height as well 
    faces = fc.detectMultiScale(gray, 1.3, 5)
    # Display the rectangle on the face
    for (x,y,w,h) in faces:
         cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
         id,conf=rec.predict(gray[y:y+h,x:x+w])
         if(id==3):
                 id="Ankit"
         elif(id==2):
                 id="Sunny"
         elif (id == 4):
             id = "MR.Modi"
         elif (id == 5):
             id = "BB"
         cv2.putText(img, str(id), (x,y+h), fontface, fontscale, fontcolor) 
    # Displays the captured image from camera with the face
    cv2.imshow('face detection',img)
    # Break when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
