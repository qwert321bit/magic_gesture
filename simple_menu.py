# simple_menu.py
import cv2
import numpy as np
import sys

class SimpleMenu:
    def __init__(self):
        self.width = 600
        self.height = 400
        self.selected = 0
        self.games = [
            ("Snake", "snake"),
            ("Painter", "painter"),
            ("RPS", "rps"),
            ("Exit", "exit")
        ]
        
        cv2.namedWindow("Simple Menu", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Simple Menu", self.width, self.height)
    
    def draw(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (40, 40, 60)
        
        # 标题
        cv2.putText(img, "GAME SELECTION", (150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 100), 2)
        
        # 游戏列表
        for i, (name, _) in enumerate(self.games):
            y = 120 + i * 60
            
            if i == self.selected:
                cv2.rectangle(img, (150, y-25), (450, y+25), (80, 80, 120), -1)
                cv2.rectangle(img, (150, y-25), (450, y+25), (100, 200, 255), 2)
                color = (100, 200, 255)
            else:
                color = (200, 200, 200)
            
            cv2.putText(img, f"{i+1}. {name}", (170, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 说明
        cv2.putText(img, "UP/DOWN: Navigate", (50, 350),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
        cv2.putText(img, "ENTER: Select", (350, 350),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
        cv2.putText(img, "Q: Quit", (500, 350),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
        
        return img
    
    def run_game(self, index):
        cv2.destroyAllWindows()
        
        if index == 0:  # Snake
            from snake import FingerControlledSnake
            game = FingerControlledSnake()
            game.run()
        elif index == 1:  # Painter
            from gesture_painter_fixed import FixedGesturePainter
            game = FixedGesturePainter()
            game.run()
        elif index == 2:  # RPS
            from gesture_rps import GestureRPSGame
            game = GestureRPSGame()
            game.run()
        elif index == 3:  # Exit
            return False
        
        cv2.namedWindow("Simple Menu", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Simple Menu", self.width, self.height)
        return True
    
    def run(self):
        while True:
            img = self.draw()
            cv2.imshow("Simple Menu", img)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == 13:  # Enter
                if not self.run_game(self.selected):
                    break
            elif key == 82:  # Up
                self.selected = (self.selected - 1) % len(self.games)
            elif key == 84:  # Down
                self.selected = (self.selected + 1) % len(self.games)
            elif ord('1') <= key <= ord('4'):
                self.selected = key - ord('1')
                if not self.run_game(self.selected):
                    break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    menu = SimpleMenu()
    menu.run()