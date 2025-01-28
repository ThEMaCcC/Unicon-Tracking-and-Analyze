import cv2
import pandas as pd
from ultralytics.models.yolo import YOLO
from tracker import*
import cvzone
import numpy as np

model = YOLO('yolov8n.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture("test101.mp4") # initialize the camera

# cap = cv2.VideoCapture(0) # initialize the camera

tracker=Tracker()

# area1=[(386,252),(396,252),(396,364),(386,364)]
# area3=[(406,251),(416,251),(416,364),(406,364)]

# area2=[(241,256),(231,256),(231,364),(241,364)]
# area4=[(221,256),(211,256),(211,364),(221,364)]

# area1=[(284,464),(264,467),(200,264),(219,260)]
# area3=[(235,455),(215,478),(144,268),(171,269)]

# area4=[(534,477),(563,479),(660,286),(637,284)]
# area2=[(499,475),(515,475),(603,276),(582,277)]

# area1=[(293,464),(279,467),(208,264),(222,260)]
# area3=[(250,455),(235,478),(171,268),(189,269)]

# area4=[(534,477),(563,479),(660,286),(637,284)]
# area2=[(499,475),(515,475),(603,276),(582,277)]

area1=[(281,439),(260,439),(291,221),(308,221)]
area3=[(234,423),(219,423),(239,221),(255,221)]

area4=[(600,418),(624,418),(595,238),(572,238)]
area2=[(555,437),(572,429),(555,238),(530,238)]

going_out={}
going_in={}
counter1=[]
counter2=[]


while True:
    ret, frame = cap.read()

    resize = cv2.resize(frame,(848,480))

    # frame=cv2.resize(frame,(1020,500))
    results = model.predict(resize)  

    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    
    # person=0
    list=[]
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        if d == 0:
            list.append([x1,y1,x2,y2])
            # person+=1
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:

        x3,y3,x4,y4,id=bbox

        cx = int(x3+x4)//2
        cy = int(y3+y4)//2

        cv2.circle(resize,(cx,cy),4,(255,0,255),-1)
        cv2.rectangle(resize,(x3,y3),(x4,y4),(255,255,255),2)
        cvzone.putTextRect(resize,f'{id}',(x3,y3),1,1) 
        result=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)
        if result >= 0:
            going_out[id]=(cx,cy)
        if id in going_out:
            result1=cv2.pointPolygonTest(np.array(area4,np.int32),((cx,cy)),False)
            if result1>=0:
                cv2.circle(resize,(cx,cy),4,(255,0,255),-1)
                cv2.rectangle(resize,(x3,y3),(x4,y4),(255,0,0),2)
                cvzone.putTextRect(resize,f'{id}',(x3,y3),1,1)
                if counter1.count(id)==0:
                    counter1.append(id)

        result2=cv2.pointPolygonTest(np.array(area4,np.int32),((cx,cy)),False)
        if result2 >= 0:
            going_in[id]=(cx,cy)
        if id in going_in:
            result3=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)
            if result3>=0:
                cv2.circle(resize,(cx,cy),4,(255,0,255),-1)
                cv2.rectangle(resize,(x3,y3),(x4,y4),(255,0,0),2)
                cvzone.putTextRect(resize,f'{id}',(x3,y3),1,1)
                if counter2.count(id)==0:
                    counter2.append(id)

        result4=cv2.pointPolygonTest(np.array(area3,np.int32),((cx,cy)),False)
        if result4 >= 0:
            going_in[id]=(cx,cy)
        if id in going_in:
            result5=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
            if result5>=0:
                cv2.circle(resize,(cx,cy),4,(255,0,255),-1)
                cv2.rectangle(resize,(x3,y3),(x4,y4),(255,0,0),2)
                cvzone.putTextRect(resize,f'{id}',(x3,y3),1,1)
                if counter2.count(id)==0:
                    counter2.append(id)

        result6=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
        if result6 >= 0:
            going_out[id]=(cx,cy)
        if id in going_out:
            result7=cv2.pointPolygonTest(np.array(area3,np.int32),((cx,cy)),False)
            if result7>=0:
                cv2.circle(resize,(cx,cy),4,(255,0,255),-1)
                cv2.rectangle(resize,(x3,y3),(x4,y4),(255,0,0),2)
                cvzone.putTextRect(resize,f'{id}',(x3,y3),1,1)
                if counter1.count(id)==0:
                    counter1.append(id)

    out_c=(len(counter1))
    in_c=(len(counter2))
    summing=(in_c-out_c)
    cvzone.putTextRect(resize,f'OUT{out_c}',(50,60),1,1)
    cvzone.putTextRect(resize,f'IN{in_c}',(50,90),1,1)
    cvzone.putTextRect(resize,f'Sum{summing}',(50,120),1,1)

    cv2.polylines(resize,[np.array(area1,np.int32)],True,(0,255,0),1)
    cv2.polylines(resize,[np.array(area2,np.int32)],True,(0,255,0),1)
    cv2.polylines(resize,[np.array(area3,np.int32)],True,(0,255,0),1)
    cv2.polylines(resize,[np.array(area4,np.int32)],True,(0,255,0),1)
    # print("count = " , person)
    # annotated_frame = results[0].plot()

    # cv2.imshow("Yolo Iference", annotated_frame)
    cv2.imshow("RGB", resize) 

    if cv2.waitKey(1) == 32: # check press spacebar
        break
    # elif cv2.waitKey(0) == 65: # check press spacebar
    #     continue

cap.release()
cv2.destroyAllWindows()