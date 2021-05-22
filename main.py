import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(0)
mpHandsSolution = mp.solutions.hands
hands = mpHandsSolution.Hands()
mpDrawUtil = mp.solutions.drawing_utils

preTime = 0
curTime = 0

while cam.isOpened():
    success, frame = cam.read()
    img_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    frame_height, frame_width, frame_channel = frame.shape
    # print(frame_width,frame_height)

    if result.multi_hand_landmarks:
        for hand_land_mark in result.multi_hand_landmarks:
            mpDrawUtil.draw_landmarks(frame,hand_land_mark,mpHandsSolution.HAND_CONNECTIONS)
            for id,each_land_mark in enumerate(hand_land_mark.landmark):
                abs_x, abs_y = int(each_land_mark.x * frame_width), int(each_land_mark.y * frame_height)
                print(id,abs_x,abs_y)


    curTime = time.time()
    fps = 1/(curTime - preTime)
    preTime = curTime

    # print(fps)

    cv2.putText(frame,str(int(fps)),(15,45),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

    cv2.imshow("Window",frame)
    if cv2.waitKey(1) == ord('q'):
        break
