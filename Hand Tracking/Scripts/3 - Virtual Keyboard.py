from Hand_Tracking_Module import HandDetector
import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector()
text = ""
keys = [["Q", "W", "E",  "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ".", ",", "/"]]

maxLenght = 40 # max distance for clicking action
x, y, w, h = 150, 400, 90, 90 # start point, height and width of button
alpha = 0.6
limit = 20

backGroundColor = (255,255,255)
textColor = (0,0,0)
cnt = 0

while cap.isOpened():
    _, frame = cap.read()

    frame = cv.flip(frame, 1)
    overlay = frame.copy()

    results = detector.locate_landmarks(frame, True)
    if results: # if it's not empty, if there is a hand in frame
        lmList = results[0]['landmarks']
        p1 = lmList[4] # index finger
        p2 = lmList[8] # thumb finger
        lenght = detector.calculate_distance(p1,p2)

        if lenght > 55:
            cnt = 0

        # OUTPUT SCREEN
        cv.rectangle(overlay, (150, 300), (990, 390), backGroundColor, -1)
        cv.putText(frame, text, (160, 370), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)

        # BACKSPACE
        cv.rectangle(overlay, (1010, 300), (1140, 390), backGroundColor, -1)
        cv.putText(frame, "<=", (1020, 370), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)

        # REST OF BUTTONS
        for cnt0, row in enumerate(keys):
            for cnt1, key in enumerate(row):
                rate1 = cnt1 * 100
                rate2 = cnt0 * 100
                cv.rectangle(overlay, (x + rate1, y + rate2), (x + rate1 + w, y + rate2 + h), backGroundColor, -1)
                cv.putText(frame, key, (x + rate1 + 30, y + rate2 + 70), cv.FONT_ITALIC, 2, textColor, 5)

                # CLIKING ACTION
                if (lenght < maxLenght and cnt == 0):
                    if (1020<p2[0]<1110 and 320<p2[1]<380 and cnt == 0): # Delete action
                        textList = []
                        for i in text:
                            textList.append(i)
                        text = ""
                        if textList:
                            textList.pop()
                            for j in textList:
                                text += j
                        cnt += 1

                    if (x + rate1 + 10 < p2[0] < x + rate1 + 70) and \
                    (y + rate2 + 10 < p2[1] < y + rate2 + 70 and len(text)<20) and cnt == 0: # Add action
                        text += key
                        cnt += 1

    frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv.imshow("RESULT", frame)

    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()