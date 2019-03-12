import cv2
import numpy as np

cap = cv2.VideoCapture('manypeople.mp4')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load('C:/Users/moses.arthur/PycharmProjects/tensorflow/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while (cap.isOpened()):

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, 1.2, 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        print("Number of Faces Detected: " + str(faces.shape[0]))

    # Capture frame-by-frame
    # ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        # cv2.resizeWindow('img', 1024, 800)
        cv2.imshow('img', img)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

cap.release()
cv2.destroyAllWindows()




