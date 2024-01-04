from ultralytics import YOLO #Import the YOLOV8 library
import cv2#Import OpenCV
model=YOLO('../Yolo-Weights/yolov8n.pt')#Loads the YOLOV8 nano model
img=cv2.imread("Images/3.png")#Read the image
results=model(img, show=True)#Process the image and show it
cv2.waitKey(0)#Wait a ky be pressed to close the screenc