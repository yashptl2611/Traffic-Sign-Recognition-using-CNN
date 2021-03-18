from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import os
import imutils

classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing veh > 3.5 tons'}

model = load_model('my_model.h5')

# vs = VideoStream(src=0).start()
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("video")

time.sleep(2.0)

(W, H) = (None, None)

while True:
    ret, frame = cap.read()
    if ret == True:

        # frame = imutils.resize(frame,width=700)
        image = cv2.resize(frame, (30, 30), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        pred = model.predict_classes([image])[0]
        sign = classes[pred + 1]
        print(sign)
        cv2.putText(frame, sign, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        # cv2.imshow("Frame",frame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(25) & 0xFF
        if key == ord("q"):
            break

cv2.destroyAllWindows()
#cap.stop()
cap.release()
