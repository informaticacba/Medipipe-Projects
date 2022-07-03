from Hand_Tracking_Module import HandDetector
from threading import Thread
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(trackingConf=0.3)

colorList = [(255,255,255), (255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
colorNames = ["WHITE", "BLUE", "GREEN", "RED", "LIGHT BLUE", "YELLOW", "PURPLE"]
startIndex = 0

canvas = None
clear = False

x1,y1 = 0, 0
alpha = 0.3

cnt = 0

while cap.isOpened():
    _, frame = cap.read()

    frame = cv.flip(frame, 1)
    overlay = frame.copy()

    if clear: canvas = None

    if canvas is None:
        canvas = np.ones_like(frame)
        clear = False

    color = colorList[startIndex]
    colorName = colorNames[startIndex]

    results = detector.locate_landmarks(frame, True)
    if results:  # if not empty
        cv.putText(frame, colorName, (15,45), cv.FONT_HERSHEY_PLAIN, 3, color, 3)
        fingerPos, handPos = detector.hand_and_finger_positions(results[0])

        lmList = results[0]['landmarks']
        x, y = lmList[8][0], lmList[8][1]

        # DRAW ACTION
        if fingerPos == [0,1,0,0,0]:
             # drawing point
            if x1 == 0 and y1 == 0:
                x1, y1 = x, y
            else:
                cv.line(canvas, (x1,y1), (x, y), color, 10)
            x1,y1=x,y
        else:
            x1,y1 = 0,0

        # CLEAR ACTION
        # COLOR CHANGE ACTION
        p1, p2, p3 = lmList[4], lmList[8], lmList[12]
        lenght1 = detector.calculate_distance(p1, p2)
        lenght2 = detector.calculate_distance(p1, p3)

        if handPos["FIST"] == True:
            clear = True
        else:
            if lenght1 > 55:
                cnt= 0

            # Next Color --with index finger
            elif lenght1 < 40 and cnt == 0:
                cnt += 1
                if startIndex == len(colorList)-1:
                    startIndex = 0
                else:
                    startIndex += 1

    frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    frame = cv.add(canvas, frame)

    cv.imshow("RESULT", frame)
    if cv.waitKey(25) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()