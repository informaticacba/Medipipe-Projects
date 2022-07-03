import mediapipe as mp
import numpy as np
import cv2 as cv
import math

class HandDetector:
    def __init__(self, mode=False, maxHands=1, modelComplexity=1,
                 detectionConf=0.5, trackingConf=0.5):
        #
        self.__mode = mode
        self.__maxHands = maxHands
        self.__modelComplexity = modelComplexity
        self.__detectionConf = detectionConf
        self.__trackingConf = trackingConf
        #
        self.mpDraw = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.__mode, self.__maxHands, self.__modelComplexity,
                                        self.__detectionConf, self.__trackingConf)


    def __find_results(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img.flags.writeable = False
        self.results = self.hands.process(img)

    def draw_all_landmarks(self, img, jointColor=(0,0,0), lineColor=(255,255,255),
                           jointThickness=2, lineThickness=1):
        self.__find_results(img)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS,
                                           self.mpDraw.DrawingSpec(color=jointColor, thickness=jointThickness, circle_radius=4),
                                           self.mpDraw.DrawingSpec(color=lineColor, thickness=lineThickness, circle_radius=2))

    def locate_landmarks(self, img, flip=False, draw=False):
        self.__img = img    # created a instance variable for future uses
        self.__flip = flip  # created a instance variable for future uses

        h, w, _ = self.__img.shape
        final = []

        self.__find_results(img)

        if self.results.multi_hand_landmarks: # if it is not empty
            for handType, landmarks in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {} # will hold the landmarks and hand side (left or right)
                lmList = [] # will hold the coordinats(x,y,z)  of landmarks

                # Calculating x,y,z positions of all landmarks
                for lm in landmarks.landmark:
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w * -1)
                    lmList.append([px, py, pz])
                myHand["landmarks"] = lmList

                # Calculating the hand side (left or right)
                if flip:
                    if handType.classification[0].label == "Right":
                        myHand["label"] = "R"
                    else:
                        myHand["label"] = "L"
                else:
                    if handType.classification[0].label == "Right":
                        myHand["label"] = "L"
                    else:
                        myHand["label"] = "R"

                final.append(myHand)
            return final

    def calculate_distance(self, p1, p2, draw=False):
        x1, y1 = p1[0], p1[1] # assigning starting point variables
        x2, y2 = p2[0], p2[1] # assigning end point variables
        length = math.hypot(x2 - x1, y2 - y1) # calculating lenght between points

        if draw:
            cv.circle(self.__img, (x1, y1), 10, (255, 255, 255), -1)
            cv.circle(self.__img, (x2, y2), 10, (255, 255, 255), -1)
            cv.line(self.__img, (x1, y1), (x2, y2), (0, 0, 0), 3)

        return length

    def hand_and_finger_positions(self, hand, mode=2):
        lmList = hand["landmarks"]
        label = hand["label"]

        fingerPos = [None,None,None,None,None]
        handPos = {"FIST":None, "GRIP":None}

        # For Thumb
        if self.__flip:
            if label == "R":    sit0 = lmList[4][0] > lmList[2][0]
            elif label == "L":  sit0 = lmList[4][0] < lmList[2][0]
        elif not self.__flip:
            if label == "R":    sit0 = lmList[4][0] < lmList[2][0]
            elif label == "L":  sit0 = lmList[4][0] > lmList[2][0]

        # Level 1
        sit1 = lmList[8][1] > lmList[7][1]
        sit2 = lmList[12][1] > lmList[11][1]
        sit3 = lmList[16][1] > lmList[13][1]
        sit4 = lmList[20][1] > lmList[19][1]
        # Level 2
        sit5 = lmList[8][1] > lmList[6][1]
        sit6 = lmList[12][1] > lmList[10][1]
        sit7 = lmList[16][1] > lmList[14][1]
        sit8 = lmList[20][1] > lmList[18][1]
        # Level 3
        sit9 = lmList[8][1] > lmList[5][1]
        sit10 = lmList[12][1] > lmList[9][1]
        sit11 = lmList[16][1] > lmList[13][1]
        sit12 = lmList[20][1] > lmList[17][1]

        finger0_down = sit0
        finger1_down = sit1 and sit5
        finger2_down = sit2 and sit6
        finger3_down = sit3 and sit7
        finger4_down = sit4 and sit8

        finger0_up = not finger0_down
        finger1_up = not finger1_down
        finger2_up = not finger2_down
        finger3_up = not finger3_down
        finger4_up = not finger4_down

        if finger0_up:      fingerPos[0] = 1
        elif finger0_down:  fingerPos[0] = 0
        if finger1_up:      fingerPos[1] = 1
        elif finger1_down:  fingerPos[1] = 0
        if finger2_up:      fingerPos[2] = 1
        elif finger2_down:  fingerPos[2] = 0
        if finger3_up:      fingerPos[3] = 1
        elif finger3_down:  fingerPos[3] = 0
        if finger4_up:      fingerPos[4] = 1
        elif finger4_down:  fingerPos[4] = 0

        if sit0 and sit9 and sit10 and sit11 and sit12: handPos["FIST"] = True
        else: handPos["FIST"] = False

        return fingerPos, handPos


