import os
import sys
import cv2

video_path = 'C:/Users/USER/PycharmProjects/flask_study/uploads.mp4'
cap = cv2.VideoCapture(video_path)

cc = int(cap.get(cv2.CAP_PROP_FOURCC))
print(cc)
print(cc & 0xff, cc & 0xff00 >> 8)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(os.getcwd(), 'output.mp4'), fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        out.write(frame)

    else: break

cap.release()
out.release()
