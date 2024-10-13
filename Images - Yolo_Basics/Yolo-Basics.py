
from ultralytics import YOLO
import cv2
model = YOLO('../Yolo-Weights/yolov8n.pt')
result = model("images_for_ml/3.jpg",show = True)
cv2.waitKey(0)