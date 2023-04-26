from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

#cap = cv2.VideoCapture(r"C:\Users\rabalasa\OneDrive - Capgemini\Documents\PPE Detection\source_files\source_files\indianworkers.mp4")

model = YOLO("best.pt")

classNames=['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']
myColour=(0,0,255)
while True:
    success,img = cap.read()
    results=model(img,stream= True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w,h = x2-x1, y2-y1
            #cvzone.cornerRect(img,(x1,y1,w,h))
            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])

            currentClass = classNames[cls]
            if conf>0.5:
                if currentClass == 'Hardhat' or currentClass=='Safety Vest' or myColour=='Mask':
                    myColour = (0,255,0)
                elif currentClass == 'NO-Hardhat' or currentClass=='NO-Safety Vest' or myColour=='NO-Mask':
                    myColour = (0, 0, 255)
                else:
                    myColour = (255, 0, 0)


                cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1), max(35,y1)), scale=1, thickness=1,colorB=myColour,colorT=(255,255,255),colorR=myColour,offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColour,3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)