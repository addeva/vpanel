import cv2
import cvzone
import numpy as np
import random
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=1)

class Rect():
    def __init__(self, posCenter, size=[100,100]):
        self.posCenter = posCenter
        self.size = size
    
    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        # index finger on rectangle detected 
        if cx-w/2 < cursor[0] < cx+w/2 and cy-h/2 < cursor[1] < cy+h/2:
            self.posCenter = cursor

    def changeSize(self, lmList):
        cx, cy = self.posCenter
        w, h = self.size
        #Weight of min, max, thumbIndexDist
        wmin, wmax, wd = 0.9, 1.1, 1.1
        # thumb and index finger inside the rectangle
        if (
            (cx-w/2)*wmin < lmList[4][0] < (cx+w/2)*wmax
            and (cx-w/2)*wmin < lmList[8][0] < (cx+w/2)*wmax
            and (cy-h/2)*wmin < lmList[4][1] < (cy+h/2)*wmax
            and (cy-h/2)*wmin < lmList[8][1] < (cy+h/2)*wmax
        ):
            self.size = [thumbIndexDist*wd, thumbIndexDist*wd]
 
rectList = []
for i in range(random.randint(2, 4)):
    rectList.append(Rect([i*170+70, 200]))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img)
    cx, cy, w, h = 200, 200, 100, 100

    if lmList:
        # change size
        thumbIndexDist, _, _ = detector.findDistance(4, 8, img)

        # drag and drop
        drag, _, _, = detector.findDistance(8, 12, img)
        if drag < 30:
            cursor = lmList[8] # index finger tip landmark            
            # call cursor update
            for rect in rectList:
                rect.update(cursor)
        # change size
        for rect in rectList:
            rect.changeSize(lmList)

    ## Draw
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size 
        cv2.rectangle(imgNew, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), (196, 209, 131), cv2.FILLED)
        cvzone.cornerRect(imgNew , (int(cx-w/2), int(cy-h/2), int(w), int(h)), 20, rt=0, colorC=(80, 121, 241))
    out = img.copy()
    alpha = 0.4
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow('Image', out)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()