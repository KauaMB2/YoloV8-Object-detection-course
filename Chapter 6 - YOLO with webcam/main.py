from ultralytics import YOLO #Import the YOLOV8 library
import cv2#Import OpenCV
import cvzone#Import cvzone library
import math #Import the math library

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
              ]

#RUNNING FROM WEBCAM
cap=cv2.VideoCapture(1) #Caputre the frame
cap.set(3,1280)#Resize the image's width
cap.set(4,720)#Resize the image's height

#RUNNING FROM VIDEO FILE
#cap=cv2.VideoCapture("../Videos/bikes.mp4")#Running video file

model=YOLO("../Yolo-Weights/yolov8n.pt")

while True: 
    success, img=cap.read()#Read the content
    results=model(img, stream=True)#stream = True will use generator, which make the identification process more eficient
    for result in results:#Pass in each result
        boxes=result.boxes#Get the bouding boxes's infomation
        for box in boxes:#Pass in each bounding box
            #x1, y1, w, h=box.xywh[0]#Get the x1, y1, width and height information
            x1, y1, x2, y2=box.xyxy[0]#Get the x1, y1, x2 and y2 information
            x1, y1, x2, y2=int(x1), int(y1), int(x2), int(y2)#Convert it to integer
            w, h=x2-x1,y2-y1#Calculate the width and height of the bounding box
            #SHOWING BOUNDING BOXES USING OPENCV LIBRARY
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),3)

            #SHOWING BOUNDING BOXES USING CVZONE LIBRARY
            cvzone.cornerRect(img,(x1,y1,w,h),50)#Draw the bouding box
            conf=math.ceil(box.conf[0]*100.0)#Calculate the confidence
            classIndex=int(box.cls[0])#Get the class index
            cvzone.putTextRect(img, f'{classNames[classIndex]} {conf} %', (max(x1, 0), max(y1 - 30, 0)), scale=2, thickness=2)
            #max(x, y) is a function responsable to simplesment return the max value between the values passed in the function
    cv2.imshow("Image", img)#Show it
    cv2.waitKey(1)#Freez the image if some key is clicked