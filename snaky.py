import pygame
import os
import random
import cv2
import mediapipe as mp
import time
import threading
import queue

WIDTH = 1024
HEIGHT = 512
FPS = 15

CELL_WIDTH = 16
ROWS = HEIGHT // CELL_WIDTH
COLS = WIDTH // CELL_WIDTH

LEVELS = 4
MAX_SCORE_PER_LVL = [90, 150, 210, 1000000]
DOUBLE_SPEED_TIME = 80

# Speed levels for menu selection (L1-L7)
SPEED_LEVELS = {
    1: 8,  # L1 - Very Slow
    2: 10,  # L2 - Slow
    3: 12,  # L3 - Medium Slow
    4: 15,  # L4 - Normal (default)
    5: 18,  # L5 - Medium Fast
    6: 22,  # L6 - Fast
    7: 28  # L7 - Very Fast
}

ARENAS = {
    1: "GRASS",
    2: "FIRE",
    3: "WATER",
    4: "DESERT"
}

# initialize pygame and create window
pygame.init()
pygame.mixer.quit()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake - Hand Controlled")

# fonts
FONT_LARGE = pygame.font.SysFont('comic-sans', 50)
FONT_MEDIUM = pygame.font.SysFont('comic-sans', 30)
FONT_SMALL = pygame.font.SysFont('comic-sans', 25)
FONT_TITLE = pygame.font.SysFont('comic-sans', 70)

# GRID CONSTANTS
EMPTY = 0
SNAKE = 1
WALL = 2
RABBIT = 3
BONUS_INFINITE_RABBITS = 4
BONUS_DOUBLE_SPEED = 5

# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GREEN = (0, 128, 0)
ORANGE = (255, 165, 0)

# Game states
STATE_MENU = 0
STATE_GAME = 1
STATE_CONTROLS = 2
STATE_GAME_OVER = 3
STATE_CALIBRATION = 4


class HandController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Calibration data
        self.calibration_data = []
        self.calibrated = False
        self.calibration_duration = 3  # seconds
        self.calibration_start_time = None
        self.center_x, self.center_y = 0, 0
        self.threshold = 50  # pixels

        # Direction tracking
        self.current_direction = "Center"
        self.last_direction_time = time.time()
        self.direction_cooldown = 0.2  # seconds between direction changes

        # Thread-safe communication
        self.direction_queue = queue.Queue()
        self.camera_active = True

    def get_mean_position(self, hand_landmarks):
        x_total = y_total = 0
        for lm in hand_landmarks.landmark:
            x_total += lm.x
            y_total += lm.y
        return x_total / len(hand_landmarks.landmark), y_total / len(hand_landmarks.landmark)

    def start_calibration(self):
        self.calibration_data = []
        self.calibrated = False
        self.calibration_start_time = time.time()
        print("🎯 Starting hand calibration...")

    def update(self):
        if not self.camera_active:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        current_time = time.time()
        direction_changed = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mean_x, mean_y = self.get_mean_position(hand_landmarks)
            px, py = int(mean_x * w), int(mean_y * h)

            # Draw hand landmarks
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            if not self.calibrated and self.calibration_start_time:
                # Calibration phase
                elapsed = current_time - self.calibration_start_time
                if elapsed <= self.calibration_duration:
                    self.calibration_data.append((px, py))
                    remaining = int(self.calibration_duration - elapsed)
                    cv2.putText(frame, f"Calibrating... {remaining}s",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, "Keep hand steady in center",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # Complete calibration
                    if self.calibration_data:
                        xs, ys = zip(*self.calibration_data)
                        self.center_x = sum(xs) // len(xs)
                        self.center_y = sum(ys) // len(ys)
                        self.calibrated = True
                        print(f"✅ Calibration complete at: ({self.center_x}, {self.center_y})")

            elif self.calibrated:
                # Direction detection
                dx = px - self.center_x
                dy = py - self.center_y
                direction = "Center"

                # Add deadzone for more stable control
                deadzone = self.threshold * 0.5

                if abs(dx) > abs(dy) and abs(dx) > deadzone:
                    if dx > self.threshold:
                        direction = "Right"
                    elif dx < -self.threshold:
                        direction = "Left"
                elif abs(dy) > deadzone:
                    if dy > self.threshold:
                        direction = "Down"
                    elif dy < -self.threshold:
                        direction = "Up"

                # Only update direction if enough time has passed (prevents jittery controls)
                if (direction != self.current_direction and
                        current_time - self.last_direction_time > self.direction_cooldown):
                    self.current_direction = direction
                    self.last_direction_time = current_time
                    direction_changed = direction

                # Visual feedback
                cv2.circle(frame, (self.center_x, self.center_y), 5, (0, 255, 0), -1)  # Center point
                cv2.circle(frame, (px, py), 8, (255, 0, 0), -1)  # Current position
                cv2.putText(frame, f"Direction: {self.current_direction}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Draw threshold boundaries
                cv2.rectangle(frame,
                              (self.center_x - self.threshold, self.center_y - self.threshold),
                              (self.center_x + self.threshold, self.center_y + self.threshold),
                              (0, 255, 255), 2)

        else:
            if not self.calibrated:
                cv2.putText(frame, "Show hand to calibrate",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No hand detected",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show calibration status
        status_text = "✅ Calibrated" if self.calibrated else "❌ Not Calibrated"
        cv2.putText(frame, status_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if self.calibrated else (0, 0, 255), 2)

        cv2.imshow("Hand Control", frame)
        cv2.waitKey(1)

        return direction_changed

    def cleanup(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


# Menu button class
class Button:
    def __init__(self, x, y, width, height, text, font, color=WHITE, hover_color=LIGHT_GRAY):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False

    def draw(self, screen):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)

        text_surface = self.font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


# Load sprites (with error handling for missing files)
def load_image(path, default_color=GREEN):
    try:
        return pygame.image.load(path)
    except:
        # Create a simple colored rectangle if image not found
        surface = pygame.Surface((CELL_WIDTH, CELL_WIDTH))
        surface.fill(default_color)
        return surface


# Try to load images, fall back to colored rectangles
snake_body_horiz = load_image(os.path.join('images', 'HORIZ.png'), GREEN)
snake_body_vert = load_image(os.path.join('images', 'VERT.png'), GREEN)
rabbit = load_image(os.path.join('images', 'rabbit.png'), RED)
grass = load_image(os.path.join('images', 'grass.png'), DARK_GREEN)
fire =  load_image(os.path.join('images', 'fire.png'), RED)
water = load_image(os.path.join('images', 'water.png'), BLUE)
border = load_image(os.path.join('images', 'border-grey.png'), GRAY)
star = load_image(os.path.join('images', 'star.png'), YELLOW)

snake_head = []
snake_tail = []
snake_corners = []

for i in range(1, 5):
    snake_head.append(load_image(os.path.join('images', f'H{i}.png'), GREEN))
    snake_tail.append(load_image(os.path.join('images', f'T{i}.png'), GREEN))
    snake_corners.append(load_image(os.path.join('images', f'C{i}.png'), GREEN))


# Sound (with error handling)
def load_sound(path):
    try:
        return pygame.mixer.Sound(path)
    except:
        return None


eat_sound = load_sound(os.path.join('sound', 'eat_sound.ogg'))
die_sound = load_sound(os.path.join('sound', 'die_sound.ogg'))


class Snake:
    def __init__(self):
        self.head = [ROWS // 2, COLS // 2]
        self.body = []
        self.body.append([self.head[0], self.head[1] - 1])
        self.dir = 1
        self.body_dir = [1]
        self.pending_direction = None  # For immediate direction changes

    def dir_up(self):
        if self.dir != 2:
            self.pending_direction = 0

    def dir_down(self):
        if self.dir != 0:
            self.pending_direction = 2

    def dir_left(self):
        if self.dir != 1:
            self.pending_direction = 3

    def dir_right(self):
        if self.dir != 3:
            self.pending_direction = 1

    def update(self):
        # Apply pending direction change immediately
        if self.pending_direction is not None:
            self.dir = self.pending_direction
            self.pending_direction = None

        # Rest of the update logic remains the same...
        self.body.append(self.head)
        self.body_dir.append(self.dir)
        self.head = self.head.copy()

        if self.dir == 0:
            self.head[0] -= 1
        elif self.dir == 1:
            self.head[1] += 1
        elif self.dir == 2:
            self.head[0] += 1
        else:
            self.head[1] -= 1

        if self.head[0] < 0:
            self.head[0] = ROWS - 1
        if self.head[0] >= ROWS:
            self.head[0] = 0
        if self.head[1] < 0:
            self.head[1] = COLS - 1
        if self.head[1] >= COLS:
            self.head[1] = 0

    def reset(self):
        self.head = [ROWS // 2, COLS // 2]
        self.body = []
        self.body.append([self.head[0], self.head[1] - 1])
        self.body_dir = [1]
        self.dir = 1
        self.pending_direction = None





    def draw(self, win):
        # draw head and tail
        head_x = self.head[1] * CELL_WIDTH
        head_y = self.head[0] * CELL_WIDTH

        tail_x = self.body[0][1] * CELL_WIDTH
        tail_y = self.body[0][0] * CELL_WIDTH
        tail_dir = self.body_dir[0]

        win.blit(snake_head[self.dir], (head_x, head_y))
        win.blit(snake_tail[tail_dir], (tail_x, tail_y))

        # draw body
        for i, (row, col) in enumerate(self.body[1:]):
            x = col * CELL_WIDTH
            y = row * CELL_WIDTH

            dir = self.body_dir[i]
            next_dir = self.dir if (i + 1 >= len(self.body)) else self.body_dir[i + 1]

            # select appropriate sprite for current body part
            if dir == next_dir:
                if dir == 0 or dir == 2:
                    sprite = snake_body_vert
                else:
                    sprite = snake_body_horiz
            elif dir == 0 and next_dir == 1 or dir == 3 and next_dir == 2:
                sprite = snake_corners[3]
            elif dir == 3 and next_dir == 0 or dir == 2 and next_dir == 1:
                sprite = snake_corners[2]
            elif dir == 1 and next_dir == 0 or dir == 2 and next_dir == 3:
                sprite = snake_corners[1]
            else:
                sprite = snake_corners[0]

            # draw body part
            win.blit(sprite, (x, y))


class Grid():
    def __init__(self, lvl=0):
        self.grid = [[EMPTY for j in range(COLS)] for i in range(ROWS)]
        self.lvl = lvl
        self.set_random_rabbit()

    def set_rabbit(self, pos):
        row, col = pos
        self.grid[row][col] = RABBIT

    def set_random_rabbit(self):
        pos = self.random_empty_pos()
        self.set_rabbit(pos)

    def cell(self, pos):
        row, col = pos
        return self.grid[row][col]

    def set_cell(self, pos, content):
        row, col = pos
        self.grid[row][col] = content

    def advance_level(self):
        self.lvl += 1
        self.clear_rabbits()

        # for each level draw add additional wall cells
        if self.lvl == 1:
            for row in range(ROWS):
                if row != ROWS // 2:
                    self.grid[row][0] = self.grid[row][COLS - 1] = WALL

            for col in range(COLS):
                self.grid[0][col] = self.grid[ROWS - 1][col] = WALL
        elif self.lvl == 2:
            row1, col1 = ROWS // 4, COLS // 4
            row2 = ROWS - row1
            col2 = COLS - col1

            for col in range(col1, col2):
                self.grid[row1][col] = WALL
                self.grid[row2][col] = WALL
        elif self.lvl == 3:
            row1, col1 = ROWS // 4, COLS // 4
            row2 = ROWS - row1
            row3 = (row1 + row2) // 2
            col2 = COLS - col1

            for row in range(row1, row2 + 1):
                if row != row3:
                    self.grid[row][col1] = WALL
                    self.grid[row][col2] = WALL

        self.set_random_rabbit()

    def clear_rabbits(self):
        for row in range(ROWS):
            for col in range(COLS):
                if self.grid[row][col] == RABBIT:
                    self.grid[row][col] = EMPTY

    def random_empty_pos(self):
        positions = []

        for row in range(ROWS):
            for col in range(COLS):
                if self.grid[row][col] == EMPTY:
                    positions.append([row, col])

        return random.choice(positions) if positions else [ROWS // 2, COLS // 2]

    def next_empty_pos(self, pos):
        row, col = pos
        for i in range(row, ROWS):
            if row != i:
                col = 0

            for j in range(col, COLS):
                if self.grid[i][j] == EMPTY:
                    return (i, j)

        return False

    def draw(self, screen):
        # draw grass on all cells
        for row in range(ROWS):
            for col in range(COLS):
                x = col * CELL_WIDTH
                y = row * CELL_WIDTH
                screen.blit(grass, (x, y))

        for row in range(ROWS):
            for col in range(COLS):
                if self.grid[row][col] == WALL:
                    x = col * CELL_WIDTH
                    y = row * CELL_WIDTH
                    screen.blit(border, (x, y))
                elif self.grid[row][col] == RABBIT:
                    x = col * CELL_WIDTH
                    y = row * CELL_WIDTH
                    screen.blit(rabbit, (x, y))
                elif self.grid[row][col] == BONUS_DOUBLE_SPEED or self.grid[row][col] == BONUS_INFINITE_RABBITS:
                    x = col * CELL_WIDTH - 8
                    y = row * CELL_WIDTH - 8
                    screen.blit(star, (x, y))


def draw_menu(screen, buttons, selected_level, hand_controller):
    screen.fill(DARK_GREEN)

    # Draw title
    title_text = FONT_TITLE.render("SNAKE GAME", True, WHITE)
    title_rect = title_text.get_rect(center=(WIDTH // 2, 60))
    screen.blit(title_text, title_rect)

    # Hand control status
    status_color = GREEN if hand_controller.calibrated else RED
    status_text = "Hand Control: ON" if hand_controller.calibrated else "Hand Control: CALIBRATING"
    status_surface = FONT_SMALL.render(status_text, True, status_color)
    status_rect = status_surface.get_rect(center=(WIDTH // 2, 100))
    screen.blit(status_surface, status_rect)

    # Draw level info
    level_text = FONT_MEDIUM.render(f"Speed Level: L{selected_level}", True, WHITE)
    level_rect = level_text.get_rect(center=(WIDTH // 2, 130))
    screen.blit(level_text, level_rect)

    # Draw

    # Draw buttons
    for button in buttons:
        button.draw(screen)

    pygame.display.flip()


def draw_controls(screen):
    screen.fill(DARK_GREEN)

    title_text = FONT_LARGE.render("CONTROLS", True, WHITE)
    title_rect = title_text.get_rect(center=(WIDTH // 2, 80))
    screen.blit(title_text, title_rect)

    controls = [
        "HAND CONTROL:",
        "Move hand up/down/left/right to control snake",
        "Keep hand steady during 3-second calibration",
        "",
        "KEYBOARD (Alternative):",
        "↑ ↓ ← → - Arrow Keys to Move Snake",
        "",
        "GAME RULES:",
        "• Collect rabbits to grow and score points",
        "• Avoid walls and your own body",
        "• Look for bonus items (stars) for special effects",
        "",
        "Press ESC to return to menu"
    ]

    y_offset = 130
    for control in controls:
        if control:
            color = ORANGE if control.endswith(":") else WHITE
            text = FONT_SMALL.render(control, True, color)
            text_rect = text.get_rect(center=(WIDTH // 2, y_offset))
            screen.blit(text, text_rect)
        y_offset += 25

    pygame.display.flip()


def draw_calibration(screen, hand_controller):
    screen.fill(DARK_GREEN)

    title_text = FONT_LARGE.render("HAND CALIBRATION", True, WHITE)
    title_rect = title_text.get_rect(center=(WIDTH // 2, 100))
    screen.blit(title_text, title_rect)

    if hand_controller.calibration_start_time:
        elapsed = time.time() - hand_controller.calibration_start_time
        remaining = max(0, hand_controller.calibration_duration - elapsed)

        if remaining > 0:
            instruction_text = FONT_MEDIUM.render("Keep your hand steady in the center", True, WHITE)
            instruction_rect = instruction_text.get_rect(center=(WIDTH // 2, 200))
            screen.blit(instruction_text, instruction_rect)

            countdown_text = FONT_LARGE.render(f"{int(remaining) + 1}", True, YELLOW)
            countdown_rect = countdown_text.get_rect(center=(WIDTH // 2, 280))
            screen.blit(countdown_text, countdown_rect)
        else:
            success_text = FONT_MEDIUM.render("Calibration Complete!", True, GREEN)
            success_rect = success_text.get_rect(center=(WIDTH // 2, 240))
            screen.blit(success_text, success_rect)

    pygame.display.flip()


def best_score(score):
    try:
        with open('scores.txt', 'r+') as file:
            file_content = file.readlines()
            best = 0

            if file_content:
                best = int(file_content[0].strip())

            # sees if the current score is greater than the previous best
            if best < score:
                file.truncate(0)
                file.seek(0)
                file.write(str(score))
                best = score

            return best
    except:
        # If file doesn't exist or error, create it
        try:
            with open('scores.txt', 'w') as file:
                file.write(str(score))
            return score
        except:
            return score


def draw_game_over(screen, score):
    screen.fill(DARK_GREEN)

    # Display score
    best = best_score(score)
    text_message = FONT_LARGE.render(f'Game Over!', 1, WHITE)
    text_score = FONT_MEDIUM.render(f'Score: {score}    Best: {best}', 1, WHITE)
    text_continue = FONT_SMALL.render('Press any key to return to menu...', 1, WHITE)

    screen.blit(text_message, ((WIDTH - text_message.get_width()) // 2, HEIGHT // 2 - 100))
    screen.blit(text_score, ((WIDTH - text_score.get_width()) // 2, HEIGHT // 2 - 50))
    screen.blit(text_continue, ((WIDTH - text_continue.get_width()) // 2, HEIGHT // 2 + 20))

    pygame.display.flip()


def draw_game(screen, snake, grid, score, hand_controller):
    screen.fill(WHITE)
    grid.draw(screen)
    snake.draw(screen)

    # show score
    text = FONT_MEDIUM.render(f'Score: {score}', False, WHITE)
    screen.blit(text, (WIDTH - text.get_width() - 25, 20))

    # Show hand control status in game
    if hand_controller.calibrated:
        status_text = FONT_SMALL.render(f"Hand: {hand_controller.current_direction}", False, GREEN)
        screen.blit(status_text, (25, 20))
    else:
        status_text = FONT_SMALL.render("Hand: Not Calibrated", False, RED)
        screen.blit(status_text, (25, 20))

    pygame.display.flip()


# Main game function
def main():
    clock = pygame.time.Clock()
    hand_controller = HandController()

    # Game state variables
    game_state = STATE_CALIBRATION
    selected_level = 4  # Default to L4 (normal speed)
    current_fps = SPEED_LEVELS[selected_level]

    # Create menu buttons
    button_width = 200
    button_height = 50
    button_x = WIDTH // 2 - button_width // 2

    start_button = Button(button_x, 180, button_width, button_height, "START", FONT_MEDIUM)
    level_button = Button(button_x, 240, button_width, button_height, f"LEVEL: L{selected_level}", FONT_MEDIUM)
    calibrate_button = Button(button_x, 300, button_width, button_height, "RECALIBRATE", FONT_MEDIUM)
    controls_button = Button(button_x, 360, button_width, button_height, "CONTROLS", FONT_MEDIUM)
    quit_button = Button(button_x, 420, button_width, button_height, "QUIT", FONT_MEDIUM)

    menu_buttons = [start_button, level_button, calibrate_button, controls_button, quit_button]

    # Game variables
    snake = None
    grid = None
    score = 0
    infinite_rabbits = False
    double_speed_timer = 0
    score_lvl = [0] * LEVELS
    rabbit_pos = (0, 0)

    # Start initial calibration
    hand_controller.start_calibration()

    running = True
    while running:
        # Update hand controller
        direction_changed = hand_controller.update()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif game_state == STATE_MENU:
                # Handle menu button clicks
                if start_button.handle_event(event):
                    if hand_controller.calibrated:
                        game_state = STATE_GAME
                        snake = Snake()
                        grid = Grid()
                        score = 0
                        infinite_rabbits = False
                        double_speed_timer = 0
                        score_lvl = [0] * LEVELS
                        current_fps = SPEED_LEVELS[selected_level]

                elif level_button.handle_event(event):
                    selected_level = (selected_level % 7) + 1
                    level_button.text = f"LEVEL: L{selected_level}"
                    current_fps = SPEED_LEVELS[selected_level]

                elif calibrate_button.handle_event(event):
                    game_state = STATE_CALIBRATION
                    hand_controller.start_calibration()

                elif controls_button.handle_event(event):
                    game_state = STATE_CONTROLS

                elif quit_button.handle_event(event):
                    running = False

                # Handle button hover effects
                for button in menu_buttons:
                    button.handle_event(event)

            elif game_state == STATE_CALIBRATION:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        game_state = STATE_MENU
                # Auto-transition when calibration is complete
                if hand_controller.calibrated:
                    game_state = STATE_MENU

            elif game_state == STATE_CONTROLS:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        game_state = STATE_MENU

            elif game_state == STATE_GAME:
                if event.type == pygame.KEYDOWN:
                    # Keyboard controls as backup
                    if event.key == pygame.K_DOWN:
                        snake.dir_down()
                    elif event.key == pygame.K_UP:
                        snake.dir_up()
                    elif event.key == pygame.K_RIGHT:
                        snake.dir_right()
                    elif event.key == pygame.K_LEFT:
                        snake.dir_left()
                    elif event.key == pygame.K_ESCAPE:
                        game_state = STATE_MENU

            elif game_state == STATE_GAME_OVER:
                if event.type == pygame.KEYDOWN:
                    game_state = STATE_MENU

        # Handle hand control direction changes
        if game_state == STATE_GAME and snake and direction_changed:
            if direction_changed == "Up":
                snake.dir_up()
            elif direction_changed == "Down":
                snake.dir_down()
            elif direction_changed == "Left":
                snake.dir_left()
            elif direction_changed == "Right":
                snake.dir_right()

        # Update game logic
        if game_state == STATE_GAME:
            # Snake head on RABBIT
            if grid.cell(snake.head) == RABBIT:
                if eat_sound:
                    eat_sound.play()
                score += 10 if not infinite_rabbits else 1
                grid.set_cell(snake.head, EMPTY)

                if not infinite_rabbits:
                    grid.set_random_rabbit()
            else:
                snake.body.pop(0)
                snake.body_dir.pop(0)

            # Snake head on Bonus element
            if grid.cell(snake.head) == BONUS_DOUBLE_SPEED:
                double_speed_timer = DOUBLE_SPEED_TIME
                grid.set_cell(snake.head, EMPTY)
            elif grid.cell(snake.head) == BONUS_INFINITE_RABBITS:
                infinite_rabbits = True
                grid.set_cell(snake.head, EMPTY)

            # Make move according to snake's direction
            snake.update()

            # Game Over
            if snake.head in snake.body or grid.cell(snake.head) == WALL:
                if die_sound:
                    die_sound.play()
                game_state = STATE_GAME_OVER

            if infinite_rabbits:
                rabbit_pos = grid.next_empty_pos(rabbit_pos)

                if rabbit_pos:
                    grid.set_rabbit(rabbit_pos)
                else:
                    infinite_rabbits = False
                    grid.clear_rabbits()
                    grid.set_random_rabbit()
            else:
                # advance to next level if accumulated score per level > MAX_SCORE_PER_LVL
                if grid.lvl < len(MAX_SCORE_PER_LVL) and score - (score_lvl[grid.lvl - 1] if grid.lvl > 0 else 0) > \
                        MAX_SCORE_PER_LVL[grid.lvl]:
                    if grid.lvl < len(score_lvl):
                        score_lvl[grid.lvl] = score
                    snake.reset()
                    grid.advance_level()

            # instantiate bonus drop item with given probability
            if not infinite_rabbits and not double_speed_timer:
                r = random.uniform(0, 1)

                if r < 0.0025:
                    row, col = grid.random_empty_pos()

                    if r < 0.0015:
                        grid.grid[row][col] = BONUS_DOUBLE_SPEED
                    else:
                        grid.grid[row][col] = BONUS_INFINITE_RABBITS
                        rabbit_pos = (0, 0)

        # Render
        if game_state == STATE_MENU:
            draw_menu(screen, menu_buttons, selected_level, hand_controller)
        elif game_state == STATE_CALIBRATION:
            draw_calibration(screen, hand_controller)
        elif game_state == STATE_CONTROLS:
            draw_controls(screen)
        elif game_state == STATE_GAME:
            draw_game(screen, snake, grid, score, hand_controller)
        elif game_state == STATE_GAME_OVER:
            draw_game_over(screen, score)

        # Control frame rate
        if game_state == STATE_GAME:
            if double_speed_timer > 0:
                double_speed_timer -= 1
                clock.tick(int(2 * current_fps))
            else:
                clock.tick(current_fps)
        else:
            clock.tick(60)  # Menu runs at 60 FPS for smooth interaction

    # Cleanup
    hand_controller.cleanup()
    pygame.quit()


if __name__ == "__main__":
    main()