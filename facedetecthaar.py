# Face Detection 

import cv2
# Source to default webcam
cap = cv2.VideoCapture(0)


# Calls haar cascade classifier
# *save the file in the same folder
fc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
    # Displays the captured image from camera with the face
    cv2.imshow('face detection',img)
    # Break when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
