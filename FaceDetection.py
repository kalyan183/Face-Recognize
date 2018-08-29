import cv2
import sys

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

while True:

    # This Line Capture frame-by-frame
    retval, frame = video_capture.read()

    # This Line Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Here features stated in Haar Cascade wil be detected
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(35, 35)
    )

    # rectangle around recognized faces 
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 200), 2)

    # resulting frame
    cv2.imshow('Video', frame)

    # Exit the camera view
    if cv2.waitKey(1) & 0xFF == ord('q'):
       sys.exit() 
