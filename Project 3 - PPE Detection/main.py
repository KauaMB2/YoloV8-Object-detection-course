from ultralytics import YOLO #Import the YOLOV8 library
import cv2#Import OpenCV
import cvzone#Import cvzone library
import math #Import the math library
import time #Import timer

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

#RUNNING FROM WEBCAM
# cap=cv2.VideoCapture(1) #Caputre the frame
# cap.set(3,1280)#Resize the image's width
# cap.set(4,720)#Resize the image's height

#RUNNING FROM VIDEO FILE
cap=cv2.VideoCapture("../Videos/ppe-2.mp4")#Running video file

model=YOLO("../ppe.pt")

myColor=(0,0,255)

while True: 
    success, img=cap.read()#Read the content
    cTime=0
    pTime=0
    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv2.putText(img,f"FPS: {(fps)}",(5,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)#putText(frame,text,(positionX,positionY),font,tamanho,(B,G,R),espessura)
    results=model(img, stream=True)#stream = True will use generator, which make the identification process more eficient
    for result in results:#Pass in each result
        boxes=result.boxes#Get the bouding boxes's infomation
        for box in boxes:#Pass in each bounding box
            #x1, y1, w, h=box.xywh[0]#Get the x1, y1, width and height information
            x1, y1, x2, y2=box.xyxy[0]#Get the x1, y1, x2 and y2 information
            x1, y1, x2, y2=int(x1), int(y1), int(x2), int(y2)#Convert it to integer
            w, h=x2-x1,y2-y1#Calculate the width and height of the bounding box
            conf=math.ceil(box.conf[0]*100.0)#Calculate the confidence
            classIndex=int(box.cls[0])#Get the class index
            if conf >0.6:#If the confidence is biggest than 0.6
                if classNames[classIndex] == 'NO-Hardhat' or classNames[classIndex] == 'NO-Safety Vest' or classNames[classIndex] == 'NO-Mask':
                    myColor=(0,0,255)
                elif classNames[classIndex] == 'Hardhat' or classNames[classIndex] == 'Safety Vest' or classNames[classIndex] == 'Mask':
                    myColor=(0,255,0)
                else:
                    myColor=(255,0,255)
                #SHOWING BOUNDING BOXES USING OPENCV LIBRARY
                cv2.rectangle(img,(x1,y1),(x2,y2),myColor,3)
                cvzone.putTextRect(img, f'{classNames[classIndex]} {conf}%', (max(x1, 0), max(y1, 0)), scale=1, thickness=1, colorB=myColor, colorT=(255,255,255), offset=5, colorR=myColor)
    key=cv2.waitKey(1)#ESC = 27
    if key==27:#Se apertou o ESC
        break
    cv2.imshow("Image", img)#Show it
    