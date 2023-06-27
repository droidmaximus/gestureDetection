import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volrange = volume.GetVolumeRange()
minvol = volrange[0]
maxvol = volrange[1]

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)
  
Draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
      
    frame=cv2.flip(frame,1)
      
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      
    Process = hands.process(frameRGB)
      
    landmarkList = []
    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for _id,landmarks in enumerate(handlm.landmark):
                height,width,color_channels = frame.shape
                  
                x,y = int(landmarks.x*width),int(landmarks.y*height)              
                landmarkList.append([_id,x,y]) 
              
            Draw.draw_landmarks(frame,handlm,mpHands.HAND_CONNECTIONS)
      
    if landmarkList != []:
        x_1,y_1 = landmarkList[4][1],landmarkList[4][2]
          
        x_2,y_2 = landmarkList[8][1],landmarkList[8][2]
          
        cv2.circle(frame,(x_1,y_1),7,(0,255,0),cv2.FILLED)
        cv2.circle(frame,(x_2,y_2),7,(0,255,0),cv2.FILLED)
        
        cv2.line(frame,(x_1,y_1),(x_2,y_2),(0,255,0),3)
        L = hypot(x_2-x_1,y_2-y_1)
        b_level = np.interp(L,[15,220],[0,100])
        sbc.set_brightness(int(b_level))

        x_3,y_3 = landmarkList[12][1],landmarkList[12][2]
        x_4,y_4 = landmarkList[16][1],landmarkList[16][2]

        cv2.circle(frame,(x_3,y_3),7,(0,0,225),cv2.FILLED)
        cv2.circle(frame,(x_4,y_4),7,(0,0,225),cv2.FILLED)

        cv2.line(frame,(x_3,y_3),(x_4,y_4),(0,0,225),3)
        L2 = hypot(x_4-x_3,y_4-y_3)
        vol = np.interp(L2,[50,100],[minvol,maxvol])
        volume.SetMasterVolumeLevel(vol, None)

    

    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
