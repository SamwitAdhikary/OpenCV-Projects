import cv2
import time
import numpy as np
import handtrackingmodule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#################
wcam, hcam = 640, 480
#########

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
pTime = 0

detector = htm.handDetect(detectionCon=0.8)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0



while True:
    success, img = cap.read()


    #! Find Hand
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
        # print(length)

        #! Hand Range 50 -> 250
        #! Volume Range = -65 -> 0

        vol = np.interp(length, [50, 250], [minVol, maxVol])
        volBar = np.interp(length, [50, 250], [400, 150])
        volPer = np.interp(length, [50, 250], [0, 100])
        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            # volume.GetMute()
    
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f"{int(volPer)}%", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)