from ultralytics import YOLO
import cv2
import cvzone
import math


#This is for webcam
cap = cv2.VideoCapture(0) 
cap.set(3, 1280)
cap.set(4, 480)


#This is for videos
cap = cv2.VideoCapture("../Videos/motorbikes-1.mp4")

model = YOLO("../Yolo-Weights/yolov8n.pt")

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

while True:
    scccess, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            #bounding box'
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)


            w,h = x2-x1, y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
            #confidence
            conf = math.ceil((box.conf[0]*1000))/100

            #className
            cls = int(box.cls[0])
            print()

            cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)),scale=1,thickness=1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)