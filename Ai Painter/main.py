import cv2
import numpy as np
import handtrackingmodule as htm
import time
import os

folderPath = "Header"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []

for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

# print(len(overlayList))

header = overlayList[0]
drawColor = (0, 0, 255)
brushThickness = 15
eraserThickness = 100
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetect(detectionCon=0.7, maxHands=1)

while True:
    #! 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #! 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPositions(img, draw=False)

    if len(lmList) != 0:

        # print(lmList)

        #! Tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #! 3. Check which fingures are up
        fingures = detector.finguresUp()
        # print(fingures)

        #! 4. If selection mode - 2 fingures are up
        if fingures[1] and fingures[2]:
            xp, yp = 0, 0
            # print('selection')
            #! Checking for click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            cv2.putText(img, "Selection", (150, 170), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        #! 5. If drawing mode - Index fingure is up
        if fingures[1] and fingures[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            cv2.putText(img, "Drawing", (150, 170), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            # print('Drawing')
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1
    
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    
    # ! Setting the header image
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.putText(img, "Mode:", (40, 170), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Image', img)
    # cv2.imshow('Canvas', imgCanvas)
    if cv2.waitKey(1) == ord('q'):
        break