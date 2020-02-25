
#face recongition system to detect faces on image.

import cv2

#endre bildet til det du ønsker...
img = cv2.imread("C:/Users/Ismail/PycharmProjects/untitled/Image/face.jpeg")
#resize vis det trengs
img= cv2.resize(img, (800, 600))

#får bilde til grå farge
grayimg= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#algorithm to find the face

face_classifier = cv2.CascadeClassifier("image-recognition/haarcascade_frontalface_default.xml")
#detect the face on image
faces = face_classifier.detectMultiScale(grayimg, scaleFactor=1.3, minNeighbors=5)

#gir deg bare digits som er koordinater i bildet.
print(faces)

#for løkke for verdiene tallene er rgb farge rød
for(x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow("face Detected", img)
cv2.waitKey(0)

