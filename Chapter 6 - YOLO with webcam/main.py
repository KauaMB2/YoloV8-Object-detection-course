from ultralytics import YOLO #Import the YOLOV8 library
import cv2#Import OpenCV
import cvzone#Import cvzone library

cap=cv2.VideoCapture(0) #Caputre the frame
cap.set(3,1280)#Resize the image's width
cap.set(4,720)#Resize the image's height

while True: 
    success, img=cap.read()#Read the content
    cv2.imshow("Image", img)#Show it
    cv2.waitKey(1)#Freez the image if some key is clicked