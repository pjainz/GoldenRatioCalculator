import cv2
import time
import math
import numpy as np

def removeOutliers(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    low = mean - std*2
    high = mean + std*2
    index = 0
    for x in arr:
        if x < low or x > high:
            arr.pop(index)
        index = index + 1
    return arr

def main():
    top2chin = []
    left2right = []
    top2pupil = []
    pupil2lip = []
    noseWidth = []
    nose2lips = []

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    righteye_cascade = cv2.CascadeClassifier('haarcascade_righteye.xml')
    # lefteye_cascade = cv2.CascadeClassifier('haarcascade_leftteye.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade_nose.xml')
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    startTime = time.time()

    while True:
        ret, img = cap.read() # img = cv2.imread("test_face.jpg")
        height, width, channels = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for(x, y, w, h) in faces:
            # print("found a face")
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+h]
            roi_color = img[y:y+h, x:x+h]
            eyes = eye_cascade.detectMultiScale(roi_gray, 2.5, 5)
            smiles = smile_cascade.detectMultiScale(roi_gray, 3.4, 5)
            noses = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
            right_eyes = righteye_cascade.detectMultiScale(roi_gray, 2.5, 5)

#            for (rex, rey, rew, reh) in right_eyes:
#                cv2.rectangle(roi_color, (rex, rey), (rex + rew, rey + reh), (0, 255, 255), 2)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 1)
            for (nx, ny, nw, nh) in noses:
                cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (255, 0, 255), 1)
            if time.time() > (startTime + 2) and eyes.__len__() == 2 and smiles.__len__() == 1 and faces.__len__() == 1 and noses.__len__() == 1:
                cv2.putText(img, 'Scanning Face...', (math.floor(width / 3), math.floor(height / 12)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(roi_color, (math.floor(w/2), sy+math.floor(sh/2.5)), 2, (255, 255, 255), 2) # lips
                cv2.circle(roi_color, (math.floor(w/3), ey+math.floor(eh/2)), 2, (255, 255, 255), 2) # eye
                cv2.circle(roi_color, (math.floor(nx*1.125), ny+math.floor(nh/2)), 2, (255, 255, 255), 2) # nose R
                cv2.circle(roi_color, (nx +math.floor(nw*.875), ny+math.floor(nh/2)), 2, (255, 255, 255), 2) # nose L
                cv2.circle(roi_color, (nx + math.floor(nw/2), ny+math.floor(nh/2)), 2, (255, 255, 255), 2) # nose
                top2pupil.append(ey+(eh/2))
                pupil2lip.append((sy+(sh/2.5)) - (ey+(eh/2)))
                noseWidth.append(.75*nw)
                nose2lips.append((sy+(sh/3)) - (ny+(nh/2)))

        cv2.imshow("Face Detector", img)
        k = cv2.waitKey(30) & 0xff
        if k == 27 or top2pupil.__len__() > 40:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("The golden ratio is 1.6")
    top2pupil = removeOutliers(top2pupil)
    pupil2lip = removeOutliers(pupil2lip)
    noseWidth = removeOutliers(noseWidth)
    nose2lips = removeOutliers(nose2lips)
    avg = (np.mean(top2pupil)/np.mean(pupil2lip) + np.mean(noseWidth)/np.mean(nose2lips))/2
    print("Your ratio is: " + str(avg))
main()
