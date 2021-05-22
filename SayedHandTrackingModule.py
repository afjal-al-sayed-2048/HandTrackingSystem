import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self,isStillImage = False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mpHandsSolution = mp.solutions.hands
        self.hands = self.mpHandsSolution.Hands(isStillImage,max_num_hands,min_detection_confidence,min_tracking_confidence)
        self.mpDrawUtil = mp.solutions.drawing_utils

    def detectHands(self, frame, draw = True):
        img_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(img_rgb)
        self.frame = frame
        # print(frame_width,frame_height)

        if self.result.multi_hand_landmarks:
            for hand_land_mark in self.result.multi_hand_landmarks:
                if draw : self.mpDrawUtil.draw_landmarks(frame, hand_land_mark, self.mpHandsSolution.HAND_CONNECTIONS)
                # for id, each_land_mark in enumerate(hand_land_mark.landmark):
                #     abs_x, abs_y = int(each_land_mark.x * frame_width), int(each_land_mark.y * frame_height)
                #     print(id, abs_x, abs_y)

        return frame

    def findPositions(self,handNo=0):

        landmark_positions = []

        frame_height, frame_width, frame_channel = self.frame.shape

        if self.result.multi_hand_landmarks:
            if len(self.result.multi_hand_landmarks) > handNo:
                selected_hand = self.result.multi_hand_landmarks[handNo]
                for id, each_land_mark in enumerate(selected_hand.landmark):
                    abs_x, abs_y = int(each_land_mark.x * frame_width), int(each_land_mark.y * frame_height)
                    landmark_positions.append([id,abs_x,abs_y])

        return landmark_positions


def main():
    cam = cv2.VideoCapture(0)

    preTime = 0
    curTime = 0

    detector = HandDetector()


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

if __name__ == '__main__':
    main()