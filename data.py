import os
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from time import time

####################################
classID = 0  # 0 is fake and 1 is real
outputFolderPath = 'D:\\dataset\\datacollect'
confidence = 0.8
save = True
blurThreshold = 35  # Larger is more focus
debug = False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6
####################################

# Setup camera
cap = cv2.VideoCapture(0)  # Change to 0 if you have only one webcam
cap.set(3, camWidth)
cap.set(4, camHeight)

# Initialize face detector
detector = FaceDetector()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []  # True/False values indicating if the faces are blurred or not
    listInfo = []  # The normalized values and the class name for the label text file

    if bboxs:
        # Iterate through each detected face
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]

            # ------  Check the score --------
            if score > confidence:
                # ------  Adding an offset to the face Detected --------
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)

                # ------  To avoid values below 0 --------
                x = max(x, 0)
                y = max(y, 0)
                w = max(w, 0)
                h = max(h, 0)

                # ------  Find Blurriness --------
                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # ------  Normalize Values  --------
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2

                xcn, ycn = round(
                    xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(
                    w / iw, floatingPoint), round(h / ih, floatingPoint)

                # ------  To avoid values above 1 --------
                xcn = min(xcn, 1)
                ycn = min(ycn, 1)
                wn = min(wn, 1)
                hn = min(hn, 1)

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # ------  Drawing --------
                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 10),
                                   scale=2, thickness=3)
                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 10),
                                       scale=2, thickness=3)

        # ------  To Save --------
        if save:
            if all(listBlur) and listBlur:  # Ensure all faces are not blurred
                # ------  Save Image  --------
                timeNow = time()
                timeNow = str(timeNow).replace('.', '')
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)

                # ------  Save Label Text File  --------
                with open(f"{outputFolderPath}/{timeNow}.txt", 'a') as f:
                    for info in listInfo:
                        f.write(info)

    cv2.imshow("Image", imgOut)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
