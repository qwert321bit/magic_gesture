import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # 拇指、食指、中指、无名指、小指的指尖ID

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw and id in self.tipIds:  # 用圆圈高亮指尖
                    cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        if len(self.lmList) != 0:
            # 拇指：比较x坐标
            if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # 其他四指：比较y坐标
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

def main():
    # 初始化
    print("[INFO] 正在初始化手势识别系统...")
    detector = HandDetector(maxHands=1)  # 先检测一只手
    # 尝试打开摄像头。如果失败，请尝试将 0 改为 1 或 2。
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头！请检查：")
        print("  1. 摄像头是否被其他软件占用（如微信、Zoom）。")
        print("  2. 尝试将代码中 cv2.VideoCapture(0) 的 0 改为 1 或 2。")
        return

    pTime = 0  # 用于计算帧率
    print("[INFO] 摄像头启动成功！开始实时识别。")
    print("[提示] 将你的手放在摄像头前，窗口会显示骨骼和手指计数。")
    print("[提示] 按 'Q' 键退出程序。")

    while True:
        # 1. 读取摄像头画面
        success, img = cap.read()
        if not success:
            print("[WARN] 无法读取画面，退出。")
            break

        # 2. 检测手部并找出关键点
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)  # 不在这个函数里画点，我们用圆圈高亮

        # 3. 如果有手，判断手势并在屏幕上显示
        if lmList:
            fingers = detector.fingersUp()
            totalFingers = fingers.count(1)
            # 在屏幕上画一个信息板
            cv2.rectangle(img, (10, 10), (300, 160), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, f"Fingers: {totalFingers}", (20, 60),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
            cv2.putText(img, str(fingers), (20, 120),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            # 在指尖画高亮圆圈
            detector.findPosition(img, draw=True)

        # 4. 计算并显示帧率 (FPS)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # 5. 显示画面
        cv2.imshow("Gesture Magic - Real-time Hand Tracking", img)

        # 6. 按Q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 用户主动退出。")
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 程序已正常退出。")

if __name__ == "__main__":
    main()