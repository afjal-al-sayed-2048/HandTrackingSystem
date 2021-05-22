import cv2
import time
import SayedHandTrackingModule as shtm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np


cam = cv2.VideoCapture(0)

preTime = 0
curTime = 0

detector = shtm.HandDetector(max_num_hands=1,min_detection_confidence=0.7,min_tracking_confidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

#print(volume.GetVolumeRange())
master_volume_range = volume.GetVolumeRange()
min_volume = master_volume_range[0]
max_volume = master_volume_range[1]

#print(min_volume,max_volume)


while cam.isOpened():
    success, frame = cam.read()
    frame = cv2.flip(frame,1)

    frame = detector.detectHands(frame)

    positions = detector.findPositions()
    if len(positions) != 0:
        ax, ay = positions[4][1], positions[4][2]
        bx, by = positions[8][1], positions[8][2]

        #print(ax, ay)
        cv2.circle(frame,(ax,ay),10,(0,0,255),cv2.FILLED)
        cv2.circle(frame,(bx,by),10,(0,0,255),cv2.FILLED)

        cv2.line(frame,(ax,ay),(bx,by),(0,0,255),2)

        line_length = int(math.hypot((ax-bx),(ay-by)))
        #print(line_length)

        # min: 50 high: 200
        if line_length>=50 and line_length<=200:
            converted_volume = np.interp(line_length,[50,180],[min_volume,max_volume])
            print(converted_volume)
            volume.SetMasterVolumeLevel(converted_volume,None)



    curTime = time.time()
    fps = 1 / (curTime - preTime)
    preTime = curTime
    cv2.putText(frame, str(int(fps)), (15, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Window", frame)
    if cv2.waitKey(1) == ord('q'):
        break
