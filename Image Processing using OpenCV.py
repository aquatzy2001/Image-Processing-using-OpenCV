import numpy as np
import cv2

capture = cv2.VideoCapture(0)
width = capture.get(3)
height = capture.get(4)
font = cv2.FONT_HERSHEY_COMPLEX


low_H = [0, 0]
low_S = [0, 0]
low_V = [0, 0]

high_H = [255, 255]
high_S = [255, 255]
high_V = [255, 255]

low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

window_detection_name = 'Trackbar'

def on_low_H_thresh_trackbar(va1):
    global low_H
    global high_H
    low_H[0] = va1
    low_H[0] = min(high_H[0]-1, low_H[0])
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H[0])
def on_high_H_thresh_trackbar(va1):
    global low_H
    global high_H
    high_H[0] = va1
    high_H[0] = max(high_H[0], low_H[0]+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H[0])
def on_low_S_thresh_trackbar(va1):
    global low_S
    global high_S
    low_S[0] = va1
    low_S[0] = min(high_S[0]-1, low_S[0])
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S[0])
def on_high_S_thresh_trackbar(va1):
    global low_S
    global high_S
    high_S[0] = va1
    high_S[0] = max(high_S[0], low_S[0]+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S[0])
def on_low_V_thresh_trackbar(va1):
    global low_V
    global high_V
    low_V[0] = va1
    low_V[0] = min(high_V[0]-1, low_V[0])
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V[0])
def on_high_V_thresh_trackbar(va1):
    global low_V
    global high_V
    high_V[0] = va1
    high_V[0] = max(high_V[0], low_V[0]+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V[0])

cv2.namedWindow(window_detection_name)

def Trackbar():
    cv2.createTrackbar(low_H_name, window_detection_name, low_H[0], 255, on_low_H_thresh_trackbar)
    cv2.createTrackbar(high_H_name, window_detection_name, high_H[0], 255, on_high_H_thresh_trackbar)
    cv2.createTrackbar(low_S_name, window_detection_name, low_S[0], 255, on_low_S_thresh_trackbar)
    cv2.createTrackbar(high_S_name, window_detection_name, high_S[0], 255, on_high_S_thresh_trackbar)
    cv2.createTrackbar(low_V_name, window_detection_name, low_V[0], 255, on_low_V_thresh_trackbar)
    cv2.createTrackbar(high_V_name, window_detection_name, high_V[0], 255, on_high_V_thresh_trackbar)

while(True):
    ret, gambar = capture.read()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    HSV = cv2.cvtColor(gambar, cv2.COLOR_BGR2HSV)
    face = face_cascade.detectMultiScale(HSV, 1.1, 4)
    for(x, y, w, h) in face:
        cv2.rectangle(gambar, (x,y), (x+w, y+h), (255, 0, 0), 3)
    Trackbar()
    Dunialain = cv2.inRange(HSV, (low_H[0], low_S[0], low_V[0]), (high_H[0], high_S[0], high_V[0]))
    cnts = cv2.findContours(Dunialain.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if len(cnts)>0:
        c = max(cnts, key = cv2.contourArea)
        cnt = cnts[0]
        M = cv2.moments(c)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        #cx = int(M['m10'] / M['m00'])
        #cy = int(M['m01'] / M['m00'])
        x0= width / 2
        y0= height / 2
        center = (x0 / y0)
        center = cv2.circle(gambar, (int(x0), int(y0)), 0, (0, 0, 255), 2)
        cv2.line(gambar, (int(x), int(y)), (int(x0), int(y0)), (0, 0, 255), 2)
        dist = np.sqrt(int(x0) - int(x) ^ 2 + int(y0) - int(y) ^ 2 )
        S = 'Distance Of Object: ' + str(dist)
        cv2.putText(gambar, S, (5, 50), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        if radius>10:
            cv2.circle(gambar, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(gambar, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.putText(gambar, "centroid", (int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 225), 1)
    cv2.imshow('Camera', gambar)
    cv2.imshow('Threshold', Dunialain)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()