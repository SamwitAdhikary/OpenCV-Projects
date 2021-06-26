import cv2
import time
import numpy as np
import handtrackingmodule as htm
import math
import pyautogui

cap = cv2.VideoCapture(0)
detector = htm.handDetect(detectionCon=0.8)

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    lmList = detector.findPositions(img, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 13, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 13, (255, 0, 255), cv2.FILLED)

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)

        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            print("JUMP")
            pyautogui.press('space')


    cv2.imshow('Image', img)
    cv2.waitKey(1)