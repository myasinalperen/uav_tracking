import cv2
import numpy as np
import math
from math import sqrt
import datetime
import imutils

cap = cv2.VideoCapture("x.mp4")
#cap = cv2.VideoCapture(0)

classesFile = "classes.txt"

a = 0
distance = 0
whT = 320
classNames = []
confThreshold = 0.5
nmsThreshold = 0.3

kumanda_aralık=10
#kardeşlerimi çok seviyorum :)

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov3-testing.cfg'
modelWeigts = 'sihax.weights'

net = cv2.dnn.readNet("sihax.weights", "yolov3_testing.cfg")


# net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeigts)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        cv2.rectangle(img, (x, y), (x + w, y + h), (25, 50, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 50, 255), 2)

        return x, y, w, h


while True:
    success, img = cap.read()
    gray_video = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    cv2.rectangle(img, (int(img.shape[1] / 4), int(img.shape[0] / 10)),
                  (int(3 * img.shape[1] / 4), int(9 * img.shape[0] / 10)), (204, 0, 102), 3)
    cv2.rectangle(img, (0, 0), (int(img.shape[1]), int(img.shape[0])), (0, 153, 76), 4)
    # cv2.circle(img,(int (img.shape[1]/2),int (img.shape[0]/2)))

    cv2.putText(img, "AK : Kamera Gorus Alani", (10, int(img.shape[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0),
                2, cv2.LINE_AA, False)
    cv2.putText(img, "Av : Hedef Vurus Alani", (int(img.shape[1] / 4) + 5, int(9 * img.shape[0] / 10) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2, cv2.LINE_AA, False)

    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    returner = findObjects(outputs, img)

    print(returner)

    if returner is not None:
        cv2.line(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)),
                 (int(returner[0] + returner[2] / 2), int(returner[1] + returner[3] / 2)), (0, 255, 0), 2)
        roi_img = img[returner[1]:returner[1] + returner[3], returner[0]:returner[0] + returner[2]]
        # cv2.imshow('Image2', roi_img)

        # img[0: returner[3],0: returner[2]] = roi_img
        # roi_img2 = img[returner[1]:returner[1] + 100, returner[0]:returner[0] + 100]

        # dronu daha küçük bir alanda bulmak
        # roi_drone = img[returner[1]+100 :returner[1] + returner[3] + 100, returner[0]-100 :returner[0] + returner[2]+ 100]
        # findObjects(outputs, roi_drone)

        # cv2.imshow('Image2', roi_img2)

        distance = ((returner[0] - int(img.shape[0] / 2)) ** 2 + (returner[1] - int(img.shape[0] / 2) ** 2))
        #print(math.sqrt(distance))
        #cv2.putText(img, "Distance:" + ('%d' % int(distance)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200),
                    #2, cv2.LINE_AA, False)

        # sağ yada sol taraftamı ona bakan bir algoritma yaz
        # üst taraftamı alt taraftamı ona bakan bir algoritma
        horizantal_difference = int(returner[0] + returner[2] / 2) - int(img.shape[1] / 2)
        if horizantal_difference  > 0:
            print("right")
            cv2.putText(img, "Right:"+ ('%.2f' % float((horizantal_difference)/kumanda_aralık)), (int(img.shape[1] / 4) + 5, int(9 * img.shape[0] / 10) - 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA, False)
        else:
            print("left")
            cv2.putText(img, "Left:"+ ('%.2f' % float((horizantal_difference*-1)/kumanda_aralık)), (int(img.shape[1] / 4) + 5, int(9 * img.shape[0] / 10) - 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA, False)

        vertical_difference = int(returner[1] + returner[3] / 2) - int(img.shape[0] / 2)

        if vertical_difference > 0:
            print("down")
            cv2.putText(img, "Down:"+ ('%.2f' % float((vertical_difference)/kumanda_aralık)), (int(img.shape[1] / 4) + 5, int(9 * img.shape[0] / 10) - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA, False)
        else:
            print("up")
            cv2.putText(img, "Up:"+ ('%.2f' % float((vertical_difference*-1)/kumanda_aralık)), (int(img.shape[1] / 4) + 5, int(9 * img.shape[0] / 10) - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA, False)

        if int(img.shape[1] / 4)< returner[0] < int(3 * img.shape[1] / 4) and int(img.shape[0] / 10) < returner[1] < int(9 * img.shape[0] / 10):
           print("içerdeee")
           cv2.putText(img, "Inside",
                       (int(img.shape[1] / 4) + 5, int(9 * img.shape[0] / 10) - 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 220, 0), 2, cv2.LINE_AA, False)
        else:
            print("dışarda")
            cv2.putText(img, "Outside",
                        (int(img.shape[1] / 4) + 5, int(9 * img.shape[0] / 10) - 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 200), 2, cv2.LINE_AA, False)
        print(distance)

        cv2.circle(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), 3, (0, 0, 0), 2)
        cv2.circle(img, (int(returner[0] + returner[2] / 2), int(returner[1] + returner[3] / 2)), 3, (0, 0, 255), 2)

    cv2.imshow('Image', img)

    cv2.waitKey(1)