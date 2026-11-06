"""
Fixed Gesture Painter - Stable Version
修复问题：
1. 添加详细的调试信息
2. 改进手势识别逻辑
3. 添加错误处理和状态显示
4. 确保摄像头正确初始化
"""

import cv2
import numpy as np
import time
import sys

# ========== Hand Detector Class ==========
class HandDetector:
    """稳定的手势检测器"""
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.5):
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
                model_complexity=0
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.tip_ids = [4, 8, 12, 16, 20]  # 指尖索引
            self.mediapipe_available = True
            print("✅ Hand detector initialized successfully")
        except ImportError as e:
            print(f"❌ MediaPipe import error: {e}")
            self.mediapipe_available = False
        except Exception as e:
            print(f"❌ Hand detector initialization error: {e}")
            self.mediapipe_available = False
    
    def detect_hands(self, frame, draw=True):
        """检测手部并绘制关键点"""
        if not self.mediapipe_available or frame is None:
            return frame, []
        
        try:
            # 转换颜色空间
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            landmarks_list = []
            
            if results.multi_hand_landmarks and draw:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制手部关键点和连接线
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
                    )
            
            # 获取手部关键点坐标
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # 只取第一只手
                h, w, c = frame.shape
                
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks_list.append([id, cx, cy])
            
            return frame, landmarks_list
            
        except Exception as e:
            print(f"Hand detection error: {e}")
            return frame, []
    
    def get_finger_state(self, landmarks):
        """获取手指状态（伸直或弯曲）"""
        if not landmarks or len(landmarks) < 21:
            return []
        
        fingers = []
        
        # 拇指：比较指尖和指根关节的x坐标
        if landmarks[self.tip_ids[0]][1] < landmarks[self.tip_ids[0] - 1][1]:
            fingers.append(1)  # 伸直
        else:
            fingers.append(0)  # 弯曲
        
        # 其他四指：比较指尖和前一关节的y坐标
        for i in range(1, 5):
            if landmarks[self.tip_ids[i]][2] < landmarks[self.tip_ids[i] - 2][2]:
                fingers.append(1)  # 伸直
            else:
                fingers.append(0)  # 弯曲
        
        return fingers

# ========== Fixed Gesture Painter ==========
class FixedGesturePainter:
    def __init__(self):
        print("="*60)
        print("FIXED GESTURE PAINTER - DEBUG VERSION")
        print("="*60)
        
        # 初始化手势检测器
        print("Initializing hand detector...")
        self.detector = HandDetector()
        
        if not self.detector.mediapipe_available:
            print("❌ Cannot start without MediaPipe. Please install: pip install mediapipe")
            sys.exit(1)
        
        # 初始化摄像头
        print("Initializing camera...")
        self.camera = None
        self.camera_index = 0
        self.camera_available = False
        self._init_camera()
        
        if not self.camera_available:
            print("❌ No camera available. Please connect a camera and try again.")
            sys.exit(1)
        
        # 画布相关
        self.canvas = None
        self.prev_point = None
        self.brush_color = (255, 0, 0)  # 蓝色 (BGR)
        self.brush_size = 10
        self.eraser_size = 30
        
        # 颜色选择
        self.colors = [
            (255, 0, 0),    # 蓝色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 紫色
            (0, 255, 255)   # 黄色
        ]
        self.color_index = 0
        self.brush_color = self.colors[self.color_index]
        
        # 控制模式
        self.mode = "draw"  # draw, erase, idle
        self.last_mode_change = time.time()
        
        # 状态跟踪
        self.frame_count = 0
        self.hand_detected_count = 0
        self.last_status_update = time.time()
        
        # 显示参数
        self.show_camera = True
        self.show_debug = True
        
        print("\n🎨 GESTURE PAINTER IS READY!")
        print("="*60)
        print("CONTROLS:")
        print("  1. INDEX FINGER POINTING - Draw")
        print("  2. FIST - Erase")
        print("  3. OPEN PALM - Change color")
        print("  4. Keyboard controls:")
        print("     C - Clear canvas")
        print("     S - Save drawing")
        print("     D - Toggle debug info")
        print("     V - Toggle camera view")
        print("     + - Increase brush size")
        print("     - - Decrease brush size")
        print("     Q - Quit")
        print("="*60)
        print("Tip: Make sure your hand is well-lit and visible to the camera.")
        print("="*60)
    
    def _init_camera(self):
        """初始化摄像头"""
        # 尝试多个摄像头索引
        for index in [0, 1, 2]:
            print(f"  Trying camera index {index}...")
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            
            if cap.isOpened():
                # 测试读取一帧
                ret, frame = cap.read()
                if ret:
                    self.camera = cap
                    self.camera_index = index
                    self.camera_available = True
                    
                    # 设置摄像头参数
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    print(f"✅ Camera {index} initialized: {frame.shape[1]}x{frame.shape[0]}")
                    return
                else:
                    cap.release()
                    print(f"  Camera {index} opened but can't read frames")
            else:
                print(f"  Camera index {index} not available")
        
        print("❌ All camera attempts failed")
    
    def _classify_gesture(self, fingers):
        """根据手指状态分类手势"""
        if not fingers:
            return "no_hand"
        
        # 计算伸直的手指数量
        total_fingers = sum(fingers)
        
        # 手势判断
        if total_fingers == 5:
            return "palm"           # 手掌张开
        elif total_fingers == 0:
            return "fist"           # 握拳
        elif fingers[1] == 1 and total_fingers == 1:
            return "pointing"       # 食指指向
        elif fingers[1] == 1 and fingers[2] == 1 and total_fingers == 2:
            return "victory"        # 剪刀手
        else:
            return f"fingers_{total_fingers}"
    
    def _process_gesture(self, gesture, landmarks, frame):
        """处理手势并执行相应操作"""
        current_time = time.time()
        
        if gesture == "no_hand":
            self.mode = "idle"
            self.prev_point = None
            return
        
        # 食指指向 - 绘画模式
        if gesture == "pointing":
            self.mode = "draw"
            if landmarks and len(landmarks) > 8:
                # 获取食指尖坐标
                index_tip = (landmarks[8][1], landmarks[8][2])
                
                if self.prev_point is None:
                    self.prev_point = index_tip
                
                # 在画布上绘制线条
                cv2.line(self.canvas, self.prev_point, index_tip, 
                        self.brush_color, self.brush_size)
                self.prev_point = index_tip
            else:
                self.prev_point = None
        
        # 握拳 - 擦除模式
        elif gesture == "fist":
            self.mode = "erase"
            if landmarks and len(landmarks) > 8:
                # 获取食指尖坐标用于擦除
                index_tip = (landmarks[8][1], landmarks[8][2])
                
                if self.prev_point is None:
                    self.prev_point = index_tip
                
                # 在画布上擦除（绘制黑色圆圈）
                cv2.circle(self.canvas, index_tip, self.eraser_size, (0, 0, 0), -1)
                self.prev_point = index_tip
            else:
                self.prev_point = None
        
        # 手掌张开 - 切换颜色
        elif gesture == "palm":
            if current_time - self.last_mode_change > 0.5:  # 防抖
                self.color_index = (self.color_index + 1) % len(self.colors)
                self.brush_color = self.colors[self.color_index]
                self.last_mode_change = current_time
                print(f"🎨 Color changed to: {self.brush_color}")
                self.prev_point = None
        
        else:
            # 其他手势，重置绘制点
            self.prev_point = None
    
    def _draw_status_panel(self, frame, gesture, landmarks_count):
        """绘制状态面板"""
        frame_h, frame_w = frame.shape[:2]
        
        # 创建半透明面板
        panel_h = 180
        panel = np.zeros((panel_h, frame_w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 60)  # 深灰色背景
        
        # 将面板添加到顶部
        frame[0:panel_h, 0:frame_w] = cv2.addWeighted(
            frame[0:panel_h, 0:frame_w], 0.3, panel, 0.7, 0
        )
        
        # 绘制状态信息
        y_offset = 30
        line_height = 25
        
        # 标题
        cv2.putText(frame, "GESTURE PAINTER - STATUS", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y_offset += line_height * 2
        
        # 手势信息
        gesture_color = (0, 255, 0) if gesture != "no_hand" else (255, 255, 255)
        cv2.putText(frame, f"Gesture: {gesture.upper()}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, gesture_color, 2)
        y_offset += line_height
        
        # 模式信息
        mode_color = (0, 255, 255) if self.mode == "draw" else (255, 255, 0) if self.mode == "erase" else (200, 200, 200)
        cv2.putText(frame, f"Mode: {self.mode.upper()}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        y_offset += line_height
        
        # 颜色信息
        color_text = f"Color: BGR{self.brush_color}"
        cv2.putText(frame, color_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.brush_color, 2)
        y_offset += line_height
        
        # 关键点信息
        cv2.putText(frame, f"Landmarks: {landmarks_count}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        y_offset += line_height
        
        # 帧率信息（如果启用调试）
        if self.show_debug:
            fps_text = f"FPS: {self._calculate_fps()}"
            cv2.putText(frame, fps_text, (frame_w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
        
        # 绘制控制提示（底部）
        controls_y = frame_h - 30
        controls = [
            "Point:Draw | Fist:Erase | Palm:Color | C:Clear | S:Save | Q:Quit"
        ]
        
        for control in controls:
            cv2.putText(frame, control, (20, controls_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1)
            controls_y += 25
    
    def _calculate_fps(self):
        """计算帧率"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_status_update >= 1.0:
            fps = self.frame_count / (current_time - self.last_status_update)
            self.frame_count = 0
            self.last_status_update = current_time
            return f"{fps:.1f}"
        
        return "Calculating..."
    
    def run(self):
        """主运行循环"""
        print("Starting gesture painter...")
        print("Press 'Q' to quit")
        
        last_frame_time = time.time()
        gesture_display = "no_hand"
        
        while True:
            # 读取摄像头帧
            ret, frame = self.camera.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # 水平镜像，让操作更直观
            frame = cv2.flip(frame, 1)
            
            # 初始化画布（如果需要）
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)
            
            # 检测手部
            processed_frame, landmarks = self.detector.detect_hands(frame)
            
            # 获取手指状态和手势
            fingers = self.detector.get_finger_state(landmarks)
            gesture = self._classify_gesture(fingers)
            
            # 更新手势显示（减少抖动）
            if gesture != "no_hand":
                gesture_display = gesture
                self.hand_detected_count += 1
            
            # 处理手势
            self._process_gesture(gesture, landmarks, frame)
            
            # 合并画布和摄像头画面
            # 创建掩码：画布非黑色部分
            canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY)
            
            # 反掩码：画布黑色部分
            mask_inv = cv2.bitwise_not(mask)
            
            # 从摄像头画面中去除画布部分
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            
            # 只保留画布的非黑色部分
            canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
            
            # 合并
            combined_frame = cv2.add(frame_bg, canvas_fg)
            
            # 在合并后的画面上绘制手部关键点
            if landmarks:
                combined_frame, _ = self.detector.detect_hands(combined_frame)
            
            # 绘制状态面板
            self._draw_status_panel(combined_frame, gesture_display, len(landmarks) if landmarks else 0)
            
            # 显示最终画面
            if self.show_camera:
                cv2.imshow("Gesture Painter - Fixed Version", combined_frame)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('c'):
                self.canvas = np.zeros_like(frame)
                print("Canvas cleared")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"painting_{timestamp}.png"
                cv2.imwrite(filename, self.canvas)
                print(f"Drawing saved as: {filename}")
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                print(f"Debug info: {'ON' if self.show_debug else 'OFF'}")
            elif key == ord('v'):
                self.show_camera = not self.show_camera
                print(f"Camera view: {'ON' if self.show_camera else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(30, self.brush_size + 2)
                self.eraser_size = min(50, self.eraser_size + 5)
                print(f"Brush size: {self.brush_size}, Eraser size: {self.eraser_size}")
            elif key == ord('-'):
                self.brush_size = max(2, self.brush_size - 2)
                self.eraser_size = max(10, self.eraser_size - 5)
                print(f"Brush size: {self.brush_size}, Eraser size: {self.eraser_size}")
            
            # 限制帧率，避免CPU占用过高
            elapsed = time.time() - last_frame_time
            if elapsed < 0.03:  # ~30 FPS
                time.sleep(0.03 - elapsed)
            last_frame_time = time.time()
        
        # 清理资源
        self.camera.release()
        cv2.destroyAllWindows()
        
        # 显示统计信息
        print("\n" + "="*60)
        print("SESSION STATISTICS:")
        print(f"  - Hand detected in {self.hand_detected_count} frames")
        print(f"  - Final brush color: {self.brush_color}")
        print(f"  - Final brush size: {self.brush_size}")
        print("="*60)
        print("Thank you for using Fixed Gesture Painter!")

# ========== 测试函数 ==========
def test_camera_only():
    """测试摄像头是否能正常工作"""
    print("Testing camera only...")
    
    # 尝试打开摄像头
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"✅ Camera {index} is working")
            
            # 显示几秒钟的测试画面
            start_time = time.time()
            while time.time() - start_time < 3:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f"Camera Test - Index {index}", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Camera Test", frame)
                    cv2.waitKey(1)
                else:
                    print("  Can't read frame")
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            return True
        else:
            print(f"  Camera {index} not available")
    
    print("❌ No working camera found")
    return False

def test_mediapipe():
    """测试MediaPipe是否能正常工作"""
    print("Testing MediaPipe...")
    
    try:
        import mediapipe as mp
        print("✅ MediaPipe is installed")
        
        # 简单的功能测试
        mp_hands = mp.solutions.hands
        print(f"✅ MediaPipe version: {mp.__version__}")
        print("✅ MediaPipe hands module available")
        return True
    except ImportError as e:
        print(f"❌ MediaPipe import error: {e}")
        print("Please install MediaPipe: pip install mediapipe")
        return False
    except Exception as e:
        print(f"❌ MediaPipe test error: {e}")
        return False

# ========== 主程序 ==========
if __name__ == "__main__":
    print("="*60)
    print("FIXED GESTURE PAINTER - STARTUP CHECK")
    print("="*60)
    
    # 运行测试
    print("\nRunning system checks...")
    
    # 测试MediaPipe
    mediapipe_ok = test_mediapipe()
    
    # 测试摄像头
    camera_ok = test_camera_only()
    
    if not mediapipe_ok or not camera_ok:
        print("\n❌ System check failed. Please fix the issues above.")
        if not mediapipe_ok:
            print("   - Install MediaPipe: pip install mediapipe")
        if not camera_ok:
            print("   - Check camera connection and permissions")
        print("="*60)
        sys.exit(1)
    
    print("\n✅ All system checks passed!")
    print("="*60)
    
    # 询问是否继续
    response = input("\nStart Gesture Painter? (Y/N): ").strip().lower()
    if response != 'y':
        print("Exiting...")
        sys.exit(0)
    
    # 启动主程序
    try:
        painter = FixedGesturePainter()
        painter.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")