import cv2
import mediapipe as mp
import numpy as np
import random
import time

class FingerControlledSnake:
    def __init__(self):
        # 初始化游戏参数
        self.width = 800
        self.height = 600
        self.cell_size = 20
        self.grid_width = self.width // self.cell_size
        self.grid_height = self.height // self.cell_size
        
        # 初始化蛇
        self.snake = [(self.grid_width // 2, self.grid_height // 2)]
        self.snake_dir = (1, 0)  # 初始向右移动
        self.snake_length = 3
        
        # 初始化食物
        self.food = self.generate_food()
        
        # 游戏状态
        self.score = 0
        self.game_over = False
        self.game_speed = 0.15  # 初始游戏速度（秒）
        self.last_move_time = time.time()
        
        # 初始化MediaPipe手部检测
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 方向映射
        self.direction_map = {
            "right": (1, 0),
            "left": (-1, 0),
            "up": (0, -1),
            "down": (0, 1)
        }
        
        # 创建游戏窗口
        cv2.namedWindow("Finger Controlled Snake Game", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Finger Controlled Snake Game", self.width, self.height)
        
    def generate_food(self):
        """在随机位置生成食物"""
        while True:
            food = (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))
            if food not in self.snake:
                return food
    
    def get_finger_direction(self, hand_landmarks):
        """根据食指位置判断方向"""
        # 获取关键点坐标
        landmarks = hand_landmarks.landmark
        
        # 获取食指指尖和根部坐标
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        # 计算食指向量
        dx = index_tip.x - index_mcp.x
        dy = index_tip.y - index_mcp.y
        
        # 根据向量方向判断移动方向
        if abs(dx) > abs(dy):
            if dx > 0.1:  # 向右
                return "right"
            elif dx < -0.1:  # 向左
                return "left"
        else:
            if dy > 0.1:  # 向下（注意：摄像头坐标系中y轴向下）
                return "down"
            elif dy < -0.1:  # 向上
                return "up"
        
        return None
    
    def update_snake(self):
        """更新蛇的位置"""
        if time.time() - self.last_move_time < self.game_speed:
            return
        
        # 计算新的头部位置
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.snake_dir
        new_head = ((head_x + dir_x) % self.grid_width, (head_y + dir_y) % self.grid_height)
        
        # 检查是否撞到自己
        if new_head in self.snake:
            self.game_over = True
            return
        
        # 添加新头部
        self.snake.insert(0, new_head)
        
        # 检查是否吃到食物
        if new_head == self.food:
            self.score += 10
            self.food = self.generate_food()
            # 随着得分增加，游戏速度加快
            self.game_speed = max(0.05, 0.15 - (self.score // 50) * 0.01)
        else:
            # 如果没有吃到食物，移除尾部
            if len(self.snake) > self.snake_length:
                self.snake.pop()
        
        self.last_move_time = time.time()
    
    def draw_game(self, frame):
        """绘制游戏界面"""
        # 创建游戏画布
        game_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # 绘制网格背景
        for x in range(0, self.width, self.cell_size):
            cv2.line(game_canvas, (x, 0), (x, self.height), (40, 40, 40), 1)
        for y in range(0, self.height, self.cell_size):
            cv2.line(game_canvas, (0, y), (self.width, y), (40, 40, 40), 1)
        
        # 绘制蛇
        for i, (x, y) in enumerate(self.snake):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)  # 头部为亮绿色，身体为暗绿色
            cv2.rectangle(game_canvas, 
                         (x * self.cell_size, y * self.cell_size),
                         ((x + 1) * self.cell_size, (y + 1) * self.cell_size),
                         color, -1)
        
        # 绘制食物
        food_x, food_y = self.food
        cv2.rectangle(game_canvas,
                     (food_x * self.cell_size, food_y * self.cell_size),
                     ((food_x + 1) * self.cell_size, (food_y + 1) * self.cell_size),
                     (0, 0, 255), -1)
        
        # 绘制分数
        cv2.putText(game_canvas, f"Score: {self.score}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 绘制游戏速度
        speed_text = f"Speed: {int(1/self.game_speed)}"
        cv2.putText(game_canvas, speed_text, (self.width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 绘制控制提示
        cv2.putText(game_canvas, "Control: Point your index finger", (10, self.height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(game_canvas, "Press 'R' to restart, 'Q' to quit", (10, self.height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # 如果游戏结束，显示游戏结束信息
        if self.game_over:
            overlay = game_canvas.copy()
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, game_canvas, 0.3, 0, game_canvas)
            
            cv2.putText(game_canvas, "GAME OVER", (self.width // 2 - 150, self.height // 2 - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(game_canvas, f"Final Score: {self.score}", (self.width // 2 - 120, self.height // 2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(game_canvas, "Press 'R' to restart", (self.width // 2 - 130, self.height // 2 + 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # 显示摄像头画面和游戏画面
        if frame is not None:
            # 调整摄像头画面大小
            cam_height, cam_width = frame.shape[:2]
            cam_aspect = cam_width / cam_height
            cam_display_width = 300
            cam_display_height = int(cam_display_width / cam_aspect)
            frame_resized = cv2.resize(frame, (cam_display_width, cam_display_height))
            
            # 将摄像头画面叠加到游戏画面上
            game_canvas[10:10 + cam_display_height, 10:10 + cam_display_width] = frame_resized
            
            # 在摄像头画面上绘制方向指示框
            cv2.rectangle(game_canvas, (8, 8), (12 + cam_display_width, 12 + cam_display_height), (255, 255, 255), 2)
        
        return game_canvas
    
    def reset_game(self):
        """重置游戏"""
        self.snake = [(self.grid_width // 2, self.grid_height // 2)]
        self.snake_dir = (1, 0)
        self.snake_length = 3
        self.food = self.generate_food()
        self.score = 0
        self.game_over = False
        self.game_speed = 0.15
        self.last_move_time = time.time()
    
    def run(self):
        """运行游戏主循环"""
        print("手指控制的贪吃蛇游戏")
        print("=====================")
        print("控制方法：")
        print("  - 将食指指向不同方向来控制贪吃蛇移动")
        print("  - 食指向上指：蛇向上移动")
        print("  - 食指向下指：蛇向下移动")
        print("  - 食指向左指：蛇向左移动")
        print("  - 食指向右指：蛇向右移动")
        print("  - 按 'R' 键重新开始游戏")
        print("  - 按 'Q' 键退出游戏")
        
        while True:
            # 读取摄像头画面
            ret, frame = self.cap.read()
            if not ret:
                print("无法访问摄像头")
                break
            
            # 水平翻转画面，使操作更直观
            frame = cv2.flip(frame, 1)
            
            # 转换为RGB格式供MediaPipe处理
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测手部
            results = self.hands.process(frame_rgb)
            
            # 如果有检测到手部，处理手部关键点
            if results.multi_hand_landmarks and not self.game_over:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 获取手指方向
                    direction = self.get_finger_direction(hand_landmarks)
                    if direction:
                        new_dir = self.direction_map[direction]
                        # 防止直接反向移动（例如从右直接向左）
                        current_dir = self.snake_dir
                        if (new_dir[0] != -current_dir[0]) or (new_dir[1] != -current_dir[1]):
                            self.snake_dir = new_dir
                    
                    # 在手部画出手部关键点
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    )
            
            # 更新游戏状态
            if not self.game_over:
                self.update_snake()
            
            # 绘制游戏画面
            game_canvas = self.draw_game(frame)
            
            # 显示游戏画面
            cv2.imshow("Finger Controlled Snake Game", game_canvas)
            
            # 处理按键事件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                self.reset_game()
        
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 检查必要的库是否已安装
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("请使用以下命令安装所需库:")
        print("pip install opencv-python mediapipe numpy")
        exit(1)
    
    # 运行游戏
    game = FingerControlledSnake()
    game.run()