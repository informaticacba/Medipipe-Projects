from Hand_Tracking_Module import HandDetector
import cv2 as cv

detector = HandDetector(trackingConf=0.3) # initiliazae the hand detector

# Start the cap
cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Arranging the image
src = cv.imread("images/02.jpg", cv.IMREAD_UNCHANGED)
_h, _w, _ = src.shape

scalePercent = 20
scaleForHold = 30
scaleForSize = 10
scaleForStart = 5

width = round(int(_w* scalePercent / 100))
height =round(int(_h * scalePercent / 100))
dsize = (width, height)
img = cv.resize(src, dsize)

# Arranging top left and bot right coordinates from center of image
startPoint = (50,50)
cX, cY = int(width/2) + startPoint[0], int(height/2) + startPoint[1]
x1, y1 = (cX - int(width/2)), (cY - int(height/2))
x2, y2 = (cX + int(width/2)), (cY + int(height/2))

while cap.isOpened():
    _, frame = cap.read()
    frame = cv.flip(frame, 1)

    alpha = 0

    h, w = frame.shape[:2]
    overlay = frame.copy()

    res = detector.locate_landmarks(frame, True)
    if res: # if it's note empty
        lmList = res[0]['landmarks']
        # calculatinf distance between thub's and index's top joint
        p1, p2 = lmList[4], lmList[8]
        mid = (int(abs(p1[0] + p2[0]) / 2), int(abs(p1[1] + p2[1]) / 2))
        lenght = detector.calculate_distance(p1, p2)
        # drawing joints and the middle points of joints (for test)
        cv.circle(frame, (p1[0], p1[1]), 7, (0, 255, 255), 3)
        cv.circle(frame, (p2[0], p2[1]), 7, (0, 255, 255), 3)
        cv.circle(overlay, mid, 7, (0, 255, 255), 3)

        # if middle point is in the image area (or behind) turn image to transparent
        if(x1<mid[0]<x2 and y1<mid[1]<y2):
            alpha = 0.5
            # calculate the ratio for drawing rectangles
            rateX = round(int(width * scaleForHold / 100))
            rateY = round(int(height * scaleForHold / 100))
            # draw the areas for actions (move - increase area - decrease area)
            cv.rectangle(overlay, (x1 + rateX, y1 + rateY), (x2 - rateX, y2 - rateY), (0, 0, 255), 3)
            cv.rectangle(overlay, (x1 + 25, y1 + 25), (x1 - 25, y1 - 25), (0, 0, 255), 3)
            cv.rectangle(overlay, (x2 + 25, y2 + 25), (x2 - 25, y2 - 25), (0, 0, 255), 3)

            if lenght < 50: # if position = click (thumb and index fingers are making contact)
                # and the middle point is inside the action rectangle
                if (x1 + rateX < p1[0] < x2 - rateX) and (y1 + rateY < p1[1] < y2 - rateY):
                    cX, cY = mid[0], mid[1]   # change new center to middle point

        # if poisition = clik and middle point is in the increase rectangle rearrange the area
        if (x1 - 25 < mid[0] < x1 + 25) and (y1 - 25 < mid[1] < y1 + 25) and lenght<50:
            rate = int(width/height)
            width = round(width + 10)
            height = round(height + (10 * rate))
            dsize = (width, height)
            img = cv.resize(img, dsize)
            cX, cY =  x2 - int(width / 2), y2 - int(height / 2)

        # if poisition = clik and middle point is in the decrease rectangle rearrange the area
        if (x2 - 25 < mid[0] < x2 + 25) and (y2 - 25 < mid[1] < y2 + 25) and lenght<50:
            rate = int(width / height)
            width = round(width - 10)
            height = round(height - (10 * rate))
            dsize = (width, height)
            img = cv.resize(img, dsize)
            cX, cY = int(width / 2) + x1, int(height / 2) + y1

        # Rearranging coordinats
        x1, y1 = (cX - int(width / 2)), (cY - int(height / 2))
        x2, y2 = (cX + int(width / 2)), (cY + int(height / 2))

    try:
        frame[y1:y2, x1:x2] = img
    except ValueError:  # --> means that image out of bounds
        # RIGHT TOP
        if x1 < 0 and y1 < 0:
            x1, y1 = 0, 0
            x2, y2 = width, height
        # RIGHT BOT
        if x1 < 0 and y2 > h:
            x1, y1 = 0, h - height
            x2, y2 = width, h
        # LEFT TOP
        if x2 > w and y1 < 0:
            x1, y1 = w - width, 0
            x2, y2 = w, height
        # LEFT BOT
        if x2 > w and y2 > h:
            x1, y1 = w - width, h - height
            x2, y2 = w, h
        # TOP BAR
        if y1 < 0:
            x1, y1 = (cX - int(width / 2)), 0
            x2, y2 = (cX + int(width / 2)), (0 + int(height))
        # BOT BAR
        elif y2 > h:
            x1, y1 = (cX - int(width / 2)), h - int(height)
            x2, y2 = (cX + int(width / 2)), h
        # RIGHT BAR
        if x1 < 0:
            x1, y1 = 0, (cY - int(height / 2))
            x2, y2 = (0 + int(width)), (cY + int(height / 2))
        # LEFT BAR
        elif x2 > w:
            x1, y1 = w - int(width), (cY - int(height / 2))
            x2, y2 = w, (cY + int(height / 2))
    finally:
        frame[y1:y2, x1:x2] = img

    frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv.imshow("Res", frame)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()