import cv2 as cv
import numpy as np

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol = volume.GetVolumeRange()
minVol = vol[0]
maxVol = vol[1]

from Hand_Tracking_Module import HandDetector

detector = HandDetector()

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while cap.isOpened():
    _, frame = cap.read()

    res = detector.locate_landmarks(frame)
    if res:
        lmList = res[0]["landmarks"]

        p1 = lmList[4]
        p2 = lmList[8]
        z = (lmList[8][2])
        lenght = detector.calculate_distance(p1, p2, True)
        x = np.interp(lenght, [50, 225], [minVol, maxVol])

        print(z, "\t", x)
        volume.SetMasterVolumeLevel(x, None)

    cv.imshow("RESULT", frame)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()