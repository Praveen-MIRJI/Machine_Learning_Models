import numpy as np
from sympy.physics.units import current
from sympy.stats.sampling.sample_numpy import numpy
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


#This is for webcam
cap = cv2.VideoCapture(0) 
cap.set(3, 1280)
cap.set(4, 480)


#This is for videos
cap = cv2.VideoCapture("../Videos/cars.mp4")

model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Boat",
    "Traffic light", "Fire hydrant", "Stop sign", "Parking meter", "Bench", "Bird", "Cat",
    "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe", "Backpack",
    "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard", "Sports ball",
    "Kite", "Baseball bat", "Baseball glove", "Skateboard", "Surfboard", "Tennis racket",
    "Bottle", "Wine glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple",
    "Sandwich", "Orange", "Broccoli", "Carrot", "Hot dog", "Pizza", "Donut", "Cake",
    "Chair", "Couch", "Potted plant", "Bed", "Dining table", "Toilet", "TV", "Laptop",
    "Mouse", "Remote", "Keyboard", "Cell phone", "Microwave", "Oven", "Toaster", "Sink",
    "Refrigerator", "Book", "Clock", "Vase", "Scissors", "Teddy bear", "Hair drier",
    "Toothbrush"
    ]

mask = cv2.imread('mask.png')

# Tracking
traker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

#limits for boundry line
limits = [400,297,673,297]

totalCount = []


while True:
    scccess, img = cap.read()
    # select not black region and choose it from image
    # basically it overlay the mask on real video
    imageRegion = cv2.bitwise_and(img, mask)

    # READING THE IMAGE HERE IS TO KEEP THE QUALITY BETTER TO KEPP IT UNCHANGED
    #IT IS PNG IMAGE ITS TRANSPARENCY REMOVED
    imgGraphics = cv2.imread('UI_car.png',cv2.IMREAD_UNCHANGED)

    # overlay this image on the video
    img = cvzone.overlayPNG(img, imgGraphics, (0,0))

    results = model(imageRegion, stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            #bounding box'
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)


            w,h = x2-x1, y2-y1

            #confidence
            conf = math.ceil((box.conf[0]*1000))/100

            #className
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == 'Car' or currentClass == 'Bus' or currentClass == 'Truck' or currentClass == 'Motorcycle' and conf > 0.3:
                #cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=0.6, thickness=1,
                                   #offset=3)
                #this is for rectangle around obj
                #cvzone.cornerRect(img, (x1, y1, w, h), l=7,rt=5)
                # selecting to detection list
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))



    resultsTracker = traker.update(detections)
    # boundry line to count if vehicle cros it
    cv2.line(img, (limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)


    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=7,rt=2, colorR=(255,0,255))
        cvzone.putTextRect(img, f" {int(id)}", (max(0, x1), max(35, y1)), scale=2, thickness=3,
                           offset=10)
        # center of the object box
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        # if the obj in this limit and multiple counts of it, it will count it once and append to list
        if limits[0] <cx <limits[2] and limits[1]-15 < cy < limits[1]+15:
            if totalCount.count(id) ==0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # rectangular count box at corner
    #cvzone.putTextRect(img, f" {len(totalCount)}", (50, 50))
    cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(0,50,255),7)



    cv2.imshow("Image", img)
    #cv2.imshow("ImageRegion", imageRegion)
    cv2.waitKey(1)