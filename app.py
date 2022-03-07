from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
classifier = load_model('Model-2.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


cap = cv2.VideoCapture(0)

while True:
    _,Frame = cap.read()
    labels = []
    gray = cv2.cvtColor(Frame, cv2.COLOR_RGB2GRAY)
    face = face_classifier.detectMultiScale(gray)
    for (x,y,w,h) in face:
        cv2.rectangle(Frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(128,128),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray/255.0
            roi = image.img_to_array(roi)
            roi = np.expand_dims(roi,axis = 0)
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x,y-10)
            cv2.putText(Frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(Frame,"No faces",(30,80),cv2.FONT_HERSHEY_SIMPLEX1,(0,255,0),2)
        cv2.imshow('Emotion Detector',Frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break 
cap.release()
cv2.destroyAllWindows()





