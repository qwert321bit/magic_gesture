"""
手势猜拳游戏 - 完整实现
功能：
1. 实时识别石头、剪刀、布手势
2. AI对手随机出拳
3. 胜负判定与得分系统
4. 多回合游戏机制
5. 丰富的视觉反馈
"""

import cv2
import pygame
import numpy as np
import random
import time
import sys
from collections import deque

# 游戏常量
WIDTH, HEIGHT = 1000, 700
FPS = 30

# 颜色定义
BACKGROUND = (25, 25, 40)
TEXT_COLOR = (240, 240, 255)
HIGHLIGHT = (255, 215, 0)
WIN_COLOR = (100, 255, 100)
LOSE_COLOR = (255, 100, 100)
DRAW_COLOR = (255, 255, 150)
ROCK_COLOR = (180, 180, 200)      # 石头 - 灰色
SCISSORS_COLOR = (100, 200, 255)  # 剪刀 - 蓝色
PAPER_COLOR = (255, 200, 100)     # 布 - 橙色

# 手势定义
GESTURES = {
    "rock": {"name": "石头", "color": ROCK_COLOR, "defeats": "scissors"},
    "scissors": {"name": "剪刀", "color": SCISSORS_COLOR, "defeats": "paper"},
    "paper": {"name": "布", "color": PAPER_COLOR, "defeats": "rock"},
    "unknown": {"name": "未知", "color": (100, 100, 100), "defeats": None}
}

# ========== 手势检测器（优化版） ==========
class RPSHandDetector:
    def __init__(self):
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]
        self.gesture_history = deque(maxlen=5)  # 手势历史，用于稳定识别
        print("🎮 猜拳手势检测器已就绪")
    
    def process_frame(self, frame):
        """处理一帧图像，返回识别到的手势和绘制了骨骼的画面"""
        if frame is None:
            return "unknown", frame
        
        # 预处理
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        # 绘制手部关键点
        display_frame = frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(121, 22, 76), thickness=3, circle_radius=4),
                    self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=3, circle_radius=3)
                )
        
        # 识别手势
        gesture = self._recognize_gesture(results)
        
        # 更新历史记录
        if gesture != "unknown":
            self.gesture_history.append(gesture)
        
        # 使用历史记录来稳定手势（取最近出现最多的手势）
        if len(self.gesture_history) >= 3:
            from collections import Counter
            most_common = Counter(self.gesture_history).most_common(1)
            if most_common:
                gesture = most_common[0][0]
        
        return gesture, display_frame
    
    def _recognize_gesture(self, results):
        """识别具体手势：石头、剪刀、布"""
        if not results.multi_hand_landmarks:
            return "unknown"
        
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 提取关键点坐标
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append((lm.x, lm.y, lm.z))
        
        # 计算手指状态
        fingers = self._get_finger_state(landmarks)
        
        # 手势判断逻辑
        # 石头: 所有手指弯曲 (全0)
        if fingers == [0, 0, 0, 0, 0]:
            return "rock"
        
        # 剪刀: 食指和中指伸直，其他弯曲
        elif fingers == [0, 1, 1, 0, 0]:
            return "scissors"
        
        # 布: 所有手指伸直 (全1)
        elif fingers == [1, 1, 1, 1, 1]:
            return "paper"
        
        # V字手势（也视为剪刀）
        elif fingers == [0, 1, 1, 0, 1] or fingers == [1, 1, 1, 0, 0]:
            return "scissors"
        
        return "unknown"
    
    def _get_finger_state(self, landmarks):
        """获取手指伸直/弯曲状态"""
        fingers = []
        
        # 拇指：比较指尖和指根的水平位置
        if landmarks[self.tip_ids[0]][0] < landmarks[self.tip_ids[0]-1][0]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # 其他四指：比较指尖和前一关节的垂直位置
        for id in range(1, 5):
            if landmarks[self.tip_ids[id]][1] < landmarks[self.tip_ids[id]-2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers

# ========== AI对手系统 ==========
class RPS_AI:
    def __init__(self, difficulty="normal"):
        """
        difficulty: easy(随机), normal(简单策略), hard(学习玩家模式)
        """
        self.difficulty = difficulty
        self.player_history = []
        self.ai_history = []
        self.win_streak = 0
        print(f"🤖 AI对手已创建 - 难度: {difficulty}")
    
    def make_choice(self, player_last_choice=None):
        """AI做出选择"""
        if self.difficulty == "easy" or not self.player_history:
            # 简单模式：完全随机
            return random.choice(["rock", "scissors", "paper"])
        
        elif self.difficulty == "normal":
            # 普通模式：简单策略
            if len(self.player_history) < 3:
                return random.choice(["rock", "scissors", "paper"])
            
            # 统计玩家最常用的手势
            from collections import Counter
            player_counter = Counter(self.player_history[-5:])  # 只看最近5次
            most_common = player_counter.most_common(1)
            
            if most_common:
                player_fav = most_common[0][0]
                # 选择能克制玩家最常用手势的选项
                for gesture, info in GESTURES.items():
                    if info["defeats"] == player_fav:
                        return gesture
            
            return random.choice(["rock", "scissors", "paper"])
        
        else:  # hard模式
            # 这里可以实现更复杂的策略
            return random.choice(["rock", "scissors", "paper"])
    
    def record_round(self, player_choice, ai_choice, result):
        """记录回合结果"""
        self.player_history.append(player_choice)
        self.ai_history.append(ai_choice)
        
        if result == "ai_win":
            self.win_streak += 1
        else:
            self.win_streak = 0

# ========== 猜拳游戏主类 ==========
class GestureRPSGame:
    def __init__(self):
        # 初始化PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("手势猜拳游戏 - 人机对战")
        self.clock = pygame.time.Clock()
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera_available = False
        
        # 尝试多个摄像头索引
        for index in [0, 1, 2]:
            cap_test = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap_test.isOpened():
                self.cap = cap_test
                self.camera_index = index
                self.camera_available = True
                print(f"✅ 摄像头 {index} 就绪")
                break
            cap_test.release()
        
        if not self.camera_available:
            print("❌ 摄像头不可用，游戏将使用模拟模式")
        
        # 设置摄像头参数
        if self.camera_available:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 初始化手势检测器
        self.detector = RPSHandDetector()
        
        # 初始化AI
        self.ai = RPS_AI(difficulty="normal")
        
        # 游戏状态
        self.game_state = "ready"  # ready, counting, show, result
        self.player_gesture = "unknown"
        self.ai_gesture = "unknown"
        self.result = "pending"  # player_win, ai_win, draw
        self.countdown = 3
        self.countdown_start = 0
        self.show_time = 0
        
        # 得分
        self.player_score = 0
        self.ai_score = 0
        self.draw_count = 0
        self.round_count = 0
        
        # 历史记录
        self.history = []
        
        # 字体
        self.title_font = pygame.font.SysFont("simhei", 48, bold=True)
        self.normal_font = pygame.font.SysFont("simhei", 32)
        self.small_font = pygame.font.SysFont("simhei", 24)
        self.big_font = pygame.font.SysFont("simhei", 72, bold=True)
        
        # 加载图标（如果没有，使用几何图形代替）
        self.icons = self._create_icons()
        
        # 最后的手势帧
        self.last_gesture_frame = None
        
        print("\n" + "="*60)
        print("手势猜拳游戏 - 操作说明")
        print("="*60)
        print("游戏流程:")
        print("  1. 准备阶段: 将手放在摄像头前")
        print("  2. 倒计时阶段: 保持手势稳定")
        print("  3. 展示阶段: 显示双方出拳")
        print("  4. 结果阶段: 显示胜负和得分")
        print("\n手势识别:")
        print("  👊 握拳 = 石头")
        print("  ✌️ 食指中指伸直 = 剪刀")
        print("  ✋ 手掌张开 = 布")
        print("\n按键:")
        print("  SPACE - 开始新回合")
        print("  R - 重置游戏")
        print("  1/2/3 - 切换AI难度")
        print("  C - 显示/隐藏摄像头")
        print("  Q - 退出游戏")
        print("="*60)
    
    def _create_icons(self):
        """创建手势图标（几何图形）"""
        icons = {}
        icon_size = 120
        
        # 石头图标（圆形）
        rock_surface = pygame.Surface((icon_size, icon_size), pygame.SRCALPHA)
        pygame.draw.circle(rock_surface, ROCK_COLOR, (icon_size//2, icon_size//2), icon_size//2 - 10)
        pygame.draw.circle(rock_surface, (255, 255, 255), (icon_size//2, icon_size//2), icon_size//2 - 10, 3)
        icons["rock"] = rock_surface
        
        # 剪刀图标（两个交叉三角形）
        scissors_surface = pygame.Surface((icon_size, icon_size), pygame.SRCALPHA)
        # 左三角形
        points1 = [(icon_size//4, icon_size//2), 
                  (icon_size//2 - 20, icon_size//4),
                  (icon_size//2 - 20, icon_size*3//4)]
        pygame.draw.polygon(scissors_surface, SCISSORS_COLOR, points1)
        # 右三角形
        points2 = [(icon_size*3//4, icon_size//2),
                  (icon_size//2 + 20, icon_size//4),
                  (icon_size//2 + 20, icon_size*3//4)]
        pygame.draw.polygon(scissors_surface, SCISSORS_COLOR, points2)
        pygame.draw.polygon(scissors_surface, (255, 255, 255), points1, 3)
        pygame.draw.polygon(scissors_surface, (255, 255, 255), points2, 3)
        icons["scissors"] = scissors_surface
        
        # 布图标（方形）
        paper_surface = pygame.Surface((icon_size, icon_size), pygame.SRCALPHA)
        pygame.draw.rect(paper_surface, PAPER_COLOR, 
                        (10, 10, icon_size-20, icon_size-20), border_radius=15)
        pygame.draw.rect(paper_surface, (255, 255, 255), 
                        (10, 10, icon_size-20, icon_size-20), 3, border_radius=15)
        icons["paper"] = paper_surface
        
        # 未知图标（问号）
        unknown_surface = pygame.Surface((icon_size, icon_size), pygame.SRCALPHA)
        pygame.draw.circle(unknown_surface, (100, 100, 100), 
                          (icon_size//2, icon_size//2), icon_size//2 - 10)
        question_mark = self.small_font.render("?", True, (255, 255, 255))
        unknown_surface.blit(question_mark, 
                            (icon_size//2 - question_mark.get_width()//2, 
                             icon_size//2 - question_mark.get_height()//2))
        icons["unknown"] = unknown_surface
        
        return icons
    
    def determine_winner(self, player_gesture, ai_gesture):
        """判定胜负"""
        if player_gesture == "unknown" or ai_gesture == "unknown":
            return "draw"
        
        if player_gesture == ai_gesture:
            return "draw"
        
        # 检查玩家是否击败AI
        if GESTURES[player_gesture]["defeats"] == ai_gesture:
            return "player_win"
        else:
            return "ai_win"
    
    def update_scores(self, result):
        """更新分数"""
        if result == "player_win":
            self.player_score += 1
        elif result == "ai_win":
            self.ai_score += 1
        else:
            self.draw_count += 1
        
        self.round_count += 1
        
        # 记录历史
        self.history.append({
            "round": self.round_count,
            "player": self.player_gesture,
            "ai": self.ai_gesture,
            "result": result
        })
        
        # 记录到AI
        self.ai.record_round(self.player_gesture, self.ai_gesture, result)
    
    def draw_background(self):
        """绘制背景"""
        self.screen.fill(BACKGROUND)
        
        # 绘制渐变效果
        for i in range(HEIGHT):
            color_value = 25 + int(15 * (i / HEIGHT))
            pygame.draw.line(self.screen, (color_value, color_value, color_value + 15), 
                            (0, i), (WIDTH, i), 1)
        
        # 绘制标题
        title = self.title_font.render("手势猜拳游戏 - 人机对战", True, HIGHLIGHT)
        self.screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))
        
        # 绘制分隔线
        pygame.draw.line(self.screen, (80, 80, 120), (WIDTH//2, 80), (WIDTH//2, HEIGHT-150), 2)
    
    def draw_scoreboard(self):
        """绘制得分板"""
        score_y = 100
        
        # 玩家得分
        player_text = self.normal_font.render("玩家", True, (180, 220, 255))
        self.screen.blit(player_text, (WIDTH//4 - player_text.get_width()//2, score_y))
        
        player_score_text = self.big_font.render(str(self.player_score), True, WIN_COLOR)
        self.screen.blit(player_score_text, (WIDTH//4 - player_score_text.get_width()//2, score_y + 40))
        
        # AI得分
        ai_text = self.normal_font.render("AI对手", True, (255, 180, 180))
        self.screen.blit(ai_text, (WIDTH*3//4 - ai_text.get_width()//2, score_y))
        
        ai_score_text = self.big_font.render(str(self.ai_score), True, LOSE_COLOR)
        self.screen.blit(ai_score_text, (WIDTH*3//4 - ai_score_text.get_width()//2, score_y + 40))
        
        # 平局和回合数
        stats_y = score_y + 120
        draw_text = self.small_font.render(f"平局: {self.draw_count}", True, DRAW_COLOR)
        self.screen.blit(draw_text, (WIDTH//2 - draw_text.get_width()//2, stats_y))
        
        round_text = self.small_font.render(f"回合: {self.round_count}", True, TEXT_COLOR)
        self.screen.blit(round_text, (WIDTH//2 - round_text.get_width()//2, stats_y + 30))
        
        # 胜率
        if self.round_count > 0:
            win_rate = self.player_score / self.round_count * 100
            rate_text = self.small_font.render(f"胜率: {win_rate:.1f}%", True, (200, 200, 255))
            self.screen.blit(rate_text, (WIDTH//2 - rate_text.get_width()//2, stats_y + 60))
    
    def draw_gesture_display(self):
        """绘制手势显示区域"""
        display_y = 250
        
        # 玩家手势区域
        player_rect = pygame.Rect(WIDTH//4 - 150, display_y, 300, 300)
        pygame.draw.rect(self.screen, (40, 40, 60), player_rect, border_radius=20)
        pygame.draw.rect(self.screen, (80, 80, 120), player_rect, 3, border_radius=20)
        
        player_label = self.normal_font.render("你的手势", True, (180, 220, 255))
        self.screen.blit(player_label, (WIDTH//4 - player_label.get_width()//2, display_y - 40))
        
        # 玩家手势图标
        player_icon = self.icons[self.player_gesture]
        self.screen.blit(player_icon, 
                        (WIDTH//4 - player_icon.get_width()//2, 
                         display_y + 150 - player_icon.get_height()//2))
        
        # 玩家手势名称
        player_name = self.normal_font.render(GESTURES[self.player_gesture]["name"], True, TEXT_COLOR)
        self.screen.blit(player_name, (WIDTH//4 - player_name.get_width()//2, display_y + 280))
        
        # AI手势区域
        ai_rect = pygame.Rect(WIDTH*3//4 - 150, display_y, 300, 300)
        pygame.draw.rect(self.screen, (60, 40, 40), ai_rect, border_radius=20)
        pygame.draw.rect(self.screen, (120, 80, 80), ai_rect, 3, border_radius=20)
        
        ai_label = self.normal_font.render("AI的手势", True, (255, 180, 180))
        self.screen.blit(ai_label, (WIDTH*3//4 - ai_label.get_width()//2, display_y - 40))
        
        # AI手势图标（在展示阶段前显示问号）
        if self.game_state == "show" or self.game_state == "result":
            ai_icon = self.icons[self.ai_gesture]
        else:
            ai_icon = self.icons["unknown"]
        
        self.screen.blit(ai_icon, 
                        (WIDTH*3//4 - ai_icon.get_width()//2, 
                         display_y + 150 - ai_icon.get_height()//2))
        
        # AI手势名称
        if self.game_state == "show" or self.game_state == "result":
            ai_name = self.normal_font.render(GESTURES[self.ai_gesture]["name"], True, TEXT_COLOR)
        else:
            ai_name = self.normal_font.render("???", True, (150, 150, 150))
        self.screen.blit(ai_name, (WIDTH*3//4 - ai_name.get_width()//2, display_y + 280))
        
        # 在展示阶段绘制VS动画
        if self.game_state == "show":
            vs_size = 50 + int(20 * abs(np.sin(time.time() * 5)))
            vs_text = self.big_font.render("VS", True, HIGHLIGHT)
            self.screen.blit(vs_text, (WIDTH//2 - vs_text.get_width()//2, display_y + 150 - vs_text.get_height()//2))
    
    def draw_result_display(self):
        """绘制结果展示"""
        if self.game_state != "result":
            return
        
        result_y = 600
        
        if self.result == "player_win":
            result_text = self.big_font.render("你赢了！", True, WIN_COLOR)
            message = "恭喜获胜！"
        elif self.result == "ai_win":
            result_text = self.big_font.render("AI赢了", True, LOSE_COLOR)
            message = "再接再厉！"
        else:
            result_text = self.big_font.render("平局！", True, DRAW_COLOR)
            message = "势均力敌！"
        
        self.screen.blit(result_text, (WIDTH//2 - result_text.get_width()//2, result_y))
        
        message_text = self.normal_font.render(message, True, TEXT_COLOR)
        self.screen.blit(message_text, (WIDTH//2 - message_text.get_width()//2, result_y + 70))
        
        # 显示克制关系
        if self.result != "draw":
            if self.result == "player_win":
                relation = f"{GESTURES[self.player_gesture]['name']} 克制 {GESTURES[self.ai_gesture]['name']}"
            else:
                relation = f"{GESTURES[self.ai_gesture]['name']} 克制 {GESTURES[self.player_gesture]['name']}"
            
            relation_text = self.small_font.render(relation, True, (200, 200, 255))
            self.screen.blit(relation_text, (WIDTH//2 - relation_text.get_width()//2, result_y + 110))
    
    def draw_countdown(self):
        """绘制倒计时"""
        if self.game_state != "counting":
            return
        
        countdown_y = HEIGHT//2 - 50
        
        # 计算倒计时
        elapsed = time.time() - self.countdown_start
        current_count = max(0, self.countdown - int(elapsed))
        
        if current_count > 0:
            # 放大动画效果
            size_factor = 1.0 + 0.5 * (1 - (elapsed % 1))
            count_text = self.big_font.render(str(current_count), True, HIGHLIGHT)
            
            # 创建放大的表面
            scaled_width = int(count_text.get_width() * size_factor)
            scaled_height = int(count_text.get_height() * size_factor)
            scaled_surface = pygame.transform.scale(count_text, (scaled_width, scaled_height))
            
            self.screen.blit(scaled_surface, 
                            (WIDTH//2 - scaled_width//2, 
                             countdown_y - scaled_height//2))
            
            # 提示文字
            hint_text = self.normal_font.render("保持手势稳定！", True, (255, 200, 100))
            self.screen.blit(hint_text, (WIDTH//2 - hint_text.get_width()//2, countdown_y + 80))
        else:
            # 倒计时结束，切换到展示阶段
            self.game_state = "show"
            self.show_time = time.time()
            
            # AI出拳
            self.ai_gesture = self.ai.make_choice(
                self.player_gesture if self.history else None
            )
            
            # 判定胜负
            self.result = self.determine_winner(self.player_gesture, self.ai_gesture)
    
    def draw_camera_feed(self):
        """绘制摄像头画面（画中画）"""
        if not self.camera_available or self.last_gesture_frame is None:
            return
        
        # 调整大小
        small_frame = cv2.resize(self.last_gesture_frame, (200, 150))
        
        # 转换为PyGame表面
        small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        small_frame_surface = pygame.surfarray.make_surface(
            np.rot90(small_frame_rgb)
        )
        
        # 绘制到左下角
        self.screen.blit(small_frame_surface, (20, HEIGHT - 170))
        
        # 边框
        pygame.draw.rect(self.screen, (255, 255, 255), 
                        (18, HEIGHT - 172, 204, 154), 2)
        
        # 标签
        cam_label = self.small_font.render("摄像头画面", True, TEXT_COLOR)
        self.screen.blit(cam_label, (20, HEIGHT - 190))
    
    def draw_controls(self):
        """绘制控制提示"""
        controls_y = HEIGHT - 40
        
        controls = [
            ("空格键", "开始新回合"),
            ("R", "重置游戏"),
            ("1/2/3", "AI难度"),
            ("C", "显示/隐藏摄像头"),
            ("Q", "退出游戏")
        ]
        
        x_pos = 20
        for key, desc in controls:
            control_text = self.small_font.render(f"{key}: {desc}", True, (180, 180, 220))
            self.screen.blit(control_text, (x_pos, controls_y))
            x_pos += control_text.get_width() + 30
    
    def draw_game_state(self):
        """根据游戏状态绘制不同界面"""
        self.draw_background()
        self.draw_scoreboard()
        self.draw_gesture_display()
        self.draw_camera_feed()
        self.draw_controls()
        
        if self.game_state == "ready":
            # 准备阶段：显示提示
            ready_text = self.normal_font.render("按下空格键开始猜拳", True, HIGHLIGHT)
            self.screen.blit(ready_text, (WIDTH//2 - ready_text.get_width()//2, HEIGHT//2 + 50))
            
            # 显示当前检测到的手势（如果摄像头可用）
            if self.player_gesture != "unknown":
                detected_text = self.small_font.render(
                    f"检测到: {GESTURES[self.player_gesture]['name']}", 
                    True, (150, 255, 150)
                )
                self.screen.blit(detected_text, (WIDTH//2 - detected_text.get_width()//2, HEIGHT//2 + 100))
        
        elif self.game_state == "counting":
            self.draw_countdown()
        
        elif self.game_state == "show":
            # 展示阶段：显示双方出拳
            if time.time() - self.show_time > 2.0:  # 展示2秒后切换到结果
                self.game_state = "result"
                self.update_scores(self.result)
        
        elif self.game_state == "result":
            self.draw_result_display()
            
            # 显示继续提示
            continue_text = self.normal_font.render("按下空格键继续", True, (200, 220, 255))
            self.screen.blit(continue_text, (WIDTH//2 - continue_text.get_width()//2, HEIGHT - 100))
    
    def process_camera(self):
        """处理摄像头输入"""
        if not self.camera_available:
            return
        
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # 识别手势
            gesture, processed_frame = self.detector.process_frame(frame)
            
            # 只在准备和倒计时阶段更新玩家手势
            if self.game_state in ["ready", "counting"]:
                self.player_gesture = gesture
            
            self.last_gesture_frame = processed_frame
    
    def start_new_round(self):
        """开始新回合"""
        if self.game_state == "ready" or self.game_state == "result":
            self.game_state = "counting"
            self.countdown_start = time.time()
            print(f"🎮 第 {self.round_count + 1} 回合开始！")
    
    def reset_game(self):
        """重置游戏"""
        self.player_score = 0
        self.ai_score = 0
        self.draw_count = 0
        self.round_count = 0
        self.history = []
        self.game_state = "ready"
        print("🔄 游戏已重置")
    
    def change_ai_difficulty(self, level):
        """改变AI难度"""
        if level == 1:
            self.ai = RPS_AI(difficulty="easy")
            print("🤖 AI难度: 简单 (随机出拳)")
        elif level == 2:
            self.ai = RPS_AI(difficulty="normal")
            print("🤖 AI难度: 普通 (简单策略)")
        elif level == 3:
            self.ai = RPS_AI(difficulty="hard")
            print("🤖 AI难度: 困难 (学习模式)")
    
    def run(self):
        """主游戏循环"""
        print("\n游戏开始！")
        
        running = True
        show_camera = True
        
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    
                    elif event.key == pygame.K_SPACE:
                        self.start_new_round()
                    
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    
                    elif event.key == pygame.K_1:
                        self.change_ai_difficulty(1)
                    
                    elif event.key == pygame.K_2:
                        self.change_ai_difficulty(2)
                    
                    elif event.key == pygame.K_3:
                        self.change_ai_difficulty(3)
                    
                    elif event.key == pygame.K_c:
                        show_camera = not show_camera
                        print(f"摄像头显示: {'开启' if show_camera else '关闭'}")
            
            # 处理摄像头输入
            if show_camera:
                self.process_camera()
            
            # 绘制游戏界面
            self.draw_game_state()
            
            # 更新显示
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # 清理资源
        if self.camera_available:
            self.cap.release()
        pygame.quit()
        cv2.destroyAllWindows()
        print("\n游戏结束！")
        print(f"最终比分: 玩家 {self.player_score} - {self.ai_score} AI")
        print(f"平局: {self.draw_count}，总回合: {self.round_count}")

# ========== 主程序入口 ==========
def main():
    print("正在启动手势猜拳游戏...")
    print("-" * 60)
    
    try:
        game = GestureRPSGame()
        game.run()
    except Exception as e:
        print(f"程序出错: {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")

if __name__ == "__main__":
    main()