"""
Serdar AKYOL
Youtube: https://bit.ly/Mask_Detection
LinkedIn: https://www.linkedin.com/in/serdarakyol55/
"""

from cv2 import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import gc

model = load_model("mask_detection3.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)     
labels = ["With mask", "Without mask"]
colors = [(0,255,0),(0,0,255)]
while 1:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert gray scala for get better result from haar cascade
    face = face_cascade.detectMultiScale(gray_frame, 1.1, 5) #get face coordinates
    
    if face in np.array(face):
        for (x, y, w, h) in face:
            try: # I used try-except because when there is no face in frame, then givin None result and Its stops projects. For fix this
                img_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert BGR image to RGB because I trained my model in RGB
                img_roi = img_RGB[y-25:y+h+25, x-25:x+w+25] # crop little bit large zone than face because some train images was bigger than only face 
                img_roi = cv2.resize(img_roi, (224, 224), interpolation= cv2.INTER_AREA) # reshaping cropped image because I trained on 224x244x3 shape
                img_roi = preprocess_input(img_roi) # normalize cropped data between -1 and 1
                img_roi = np.reshape(img_roi,(1,224,224,3)) # reshape for prediction
                pred = model.predict(img_roi) # get predictions
                pred_index = np.argmax(pred) # get higher result of prediction
                #print(pred)
                print(img_roi)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 1) #draw rectangle
                cv2.putText(frame, labels[pred_index], (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, colors[pred_index], 2) #write result
            except Exception as e:
                print(str(e))
    cv2.imshow("Frame", frame)
    gc.collect() # for collect memory
    k = cv2.waitKey(30) & 0xff
    
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
