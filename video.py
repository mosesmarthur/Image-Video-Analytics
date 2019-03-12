import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load('C:/Users/moses.arthur/PycharmProjects/tensorflow/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    # height, width = img.shape[0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 2)

#(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        print("Number of Faces Detected: " + str(faces.shape[0]))

    # define the screen resolution
    screen_res = 1024, 768
    scale_width = screen_res[0] / img .shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)

    # resized window width and height
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    # cv2.WINDOW_NORMAL makes the output window resize
    cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)

    # resize the window according to the screen resolution
    cv2.resizeWindow('Resized Window', window_width, window_height)

    # resized window width and height
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    cv2.imshow('Resized Window', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

