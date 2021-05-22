import cv2
import time
import SayedHandTrackingModule as shtm


cam = cv2.VideoCapture(0)

preTime = 0
curTime = 0

detector = shtm.HandDetector()


while cam.isOpened():
    success, frame = cam.read()
    frame = cv2.flip(frame,1)

    frame = detector.detectHands(frame)

    positions = detector.findPositions()
    if len(positions) != 0:
        print(positions[4])

    curTime = time.time()
    fps = 1 / (curTime - preTime)
    preTime = curTime

    # print(fps)

    cv2.putText(frame, str(int(fps)), (15, 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Window", frame)
    if cv2.waitKey(1) == ord('q'):
        break
