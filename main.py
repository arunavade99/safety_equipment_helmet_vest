import ultralytics
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy
import math
ultralytics.checks()
# model = YOLO('yolov8n.pt')
model = YOLO('best_with_labeled_bg.pt')

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('816.mp4')

my_file = open("safety_helmet_vest_mask.txt", "r")
data = my_file.read()
class_list = data.split("\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # count += 1
    # if count % 3 != 0:
    #     continue
    #frame = cv2.resize(frame, (600, 600))
    results = model.predict(frame,conf=0.6) #, classes=[0, 2, 1, 3, 4]
    a = results[0].boxes.data
    a = a.cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    list = []
    for index, row in px.iterrows():
        # print("row", row[0])

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        conf = (math.ceil(row[4] * 100))
        c = class_list[d]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, c+' '+str(conf), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)
    # frame = cv2.resize(frame, (1020, 500))
    frame = cv2.resize(frame, (640,600))
    cv2.imshow("Display", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
