import numpy as np#Import numpy
from ultralytics import YOLO#Import Ultralytics
import cv2#Import OpenCV
import cvzone#Import CVzone library
import math#Import math library
from sort import *#Import short
import time #Import timer

cap = cv2.VideoCapture("../Videos/people.mp4")#Load the video file

model = YOLO("../Yolo-Weights/yolov8n.pt")#Load the model
 
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]#classes of the model
 
mask = cv2.imread("mask.png")#read the mask image

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
#max_age => It is the number max of frames that the object won't be forgot if he desapear by chance
 
limitsUp = [103, 161, 296, 161]#Detection line of up information (start_x_point, start_y_point, end_x_point, end_y_point)
limitsDown = [527, 489, 735, 489]#Detection line of up information (start_x_point, start_y_point, end_x_point, end_y_point)
totalUp = []#Array to count the people going up
totalDown = []#Array to count the people going down

while True:
    success, img = cap.read()#Read the image
    cTime=0
    pTime=0
    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv2.putText(img,f"FPS: {(fps)}",(5,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)#putText(frame,text,(positionX,positionY),font,tamanho,(B,G,R),espessura)
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))#Resize the mask
    imgRegion = cv2.bitwise_and(img, mask_resized)#Apply the betwise operator between the frame and the mask
 
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)#Load the counter image
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))
    results = model(imgRegion, stream=True)
 
    detections = np.empty((0, 5))#Create the variable for the detections
 
    for r in results:#For each detection
        boxes = r.boxes#Get the bounding box
        for box in boxes:#For each box
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1#Calculate the width and height
 
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100#Calculate the confidence
            # Class Name
            cls = int(box.cls[0])#Get the class's index
            currentClass = classNames[cls]#Get the class's name
 
            if currentClass=="person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])#Organize the detection information
                detections = np.vstack((detections, currentArray))#Create a detection stack
 
    resultsTracker = tracker.update(detections)#Try assign a ID to the car detected  
 
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)#Creates the up line
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)#Creates the down line

    for result in resultsTracker:#For each result in the tracker(ID assigner)
        x1, y1, x2, y2, id = result#Get the information
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)#Convert to int
        print(result)
        w, h = x2 - x1, y2 - y1#Calculate the width and height
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))#Draw the bouding box
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)#Put the ID in the object
 
        cx, cy = x1 + w // 2, y1 + h // 2 #Find the center point of the bouding box
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)#Create a circle for this center point
 
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:#If the object passed...
            if totalUp.count(id) == 0:#If it doesn't have a object with this ID inside the counter array
                totalUp.append(id) #Add this ID in the counter array
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)#Overlay the red line
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:#If the object passed...
            if totalDown.count(id) == 0:#If it doesn't have a object with this ID inside the counter array
                totalDown.append(id) #Add this ID in the counter array
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)#Overlay the red line

    cv2.putText(img,str(len(totalUp)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)#Put the text of the up counter
    cv2.putText(img,str(len(totalDown)),(1191,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)#Put the text of the down counter text
    key=cv2.waitKey(1)#ESC = 27
    if key==27:#Se apertou o ESC
        break
    cv2.imshow("Image", img)
    #cv2.imshow("ImageRegion", imgRegion)