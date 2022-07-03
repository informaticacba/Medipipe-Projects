import cv2 as cv
import pyautogui
from   Hand_Tracking_Module import  HandDetector

resolution = pyautogui.size()

detector = HandDetector(detectionConf=0.8, trackingConf=0.8)
cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while cap.isOpened():
    _, frame = cap.read()
    frame = cv.flip(frame, 1)
    res = detector.locate_landmarks(frame, flip=True)
    if res:
        x, y = res[0]['landmarks'][12][1], res[0]['landmarks'][12][2]
        x = int(x * 7)
        y = int(y * 7)
        fingerPos, specialPos = detector.hand_and_finger_positions(res[0], True)
        p1 = res[0]['landmarks'][12]
        p2 = res[0]['landmarks'][8]
        length = detector.calculate_distance(p1, p2)
        #print(length)
        if length > 25:
            pyautogui.moveTo(x,y)
        if length < 25:
            pyautogui.click(x,y)

    cv.imshow("Frame", frame)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()