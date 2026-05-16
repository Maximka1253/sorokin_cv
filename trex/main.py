import argparse
import json
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui

REGION_FILE = Path(__file__).with_name("dino_region.json")
ZONE = (0.12, 0.48, 0.85, 0.92)
ACTION_X = 0.12
LEAD_TIME = 0.12
START_SPEED = 6.0
MAX_SPEED = 13.0
FPS = 60.0
ACCELERATION = 0.0017
DARK_LIMIT = 125
JUMP_AREA = 260
DUCK_AREA = 130
MIN_FACTOR = 0.45
MIN_AREA = 25
MIN_HEIGHT = 12
GROUND = 0.72
BIRD_MIN = 0.34
MEDIUM_WIDTH = 35
LARGE_WIDTH = 60
LOOK_TIME = 0.055
LOOK_MIN = 30
LOOK_MAX = 55
KEY_TIME = 0.015
RESTART_DELAY = 1.0
DEBUG = False

JUMP_COOLDOWN = 0.30
BIRD_HOLD = (0.34, 0.54)
DUCK_TIMES = {
    "single": (0.20, 0.18, 0.16),
    "medium": (0.28, 0.26, 0.24),
    "large": (0.3, 0.28, 0.26),
}

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False
is_ducking = False


def load_region():
    if not REGION_FILE.exists():
        return None

    with REGION_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return {
        "top": int(data["top"]),
        "left": int(data["left"]),
        "width": int(data["width"]),
        "height": int(data["height"]),
    }


def save_region(region):
    with REGION_FILE.open("w", encoding="utf-8") as file:
        json.dump(region, file, indent=2)


def select_region(sct):
    monitor = sct.monitors[0]
    screen = np.array(sct.grab(monitor), dtype=np.uint8)
    screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
    window = "Выберите область игры"

    print("Выделите область игры мышью, "
          "затем нажмите Enter или Space.")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1400, 800)
    x, y, width, height = cv2.selectROI(window, screen, False, False)
    cv2.destroyWindow(window)

    if width <= 0 or height <= 0:
        raise SystemExit("Область не выбрана.")

    return {
        "top": int(monitor["top"] + y),
        "left": int(monitor["left"] + x),
        "width": int(width),
        "height": int(height),
    }


def get_region(sct, reselect):
    region = None if reselect else load_region()

    if region is None:
        region = select_region(sct)
        save_region(region)
        print("Область сохранена в", REGION_FILE.name)

    return region


def press(key, delay=KEY_TIME):
    pyautogui.keyDown(key)
    time.sleep(delay)
    pyautogui.keyUp(key)


def grab_frame(sct, region):
    frame = np.array(sct.grab(region), dtype=np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def prepare_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, DARK_LIMIT)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return gray, mask


def zone_pixels(shape):
    height, width = shape[:2]
    x1 = int(width * ZONE[0])
    y1 = int(height * ZONE[1])
    x2 = int(width * ZONE[2])
    y2 = int(height * ZONE[3])
    return x1, y1, x2, y2


def contours(mask):
    result = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return result[-2]


def group_by_width(roi, x_offset, trigger_x, look_x):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    merged = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    group = "single"

    for contour in contours(merged):
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        real_x = x_offset + x
        bottom = (y + height) / roi.shape[0]

        if real_x > look_x or real_x > trigger_x:
            continue
        if height < MIN_HEIGHT or bottom < GROUND:
            continue
        if width >= LARGE_WIDTH:
            return "large"
        if width >= MEDIUM_WIDTH:
            group = "medium"

    return group


def detect_obstacle(mask, jump_area, duck_area, action_x, look_ahead):
    x1, y1, x2, y2 = zone_pixels(mask.shape)
    roi = mask[y1:y2, x1:x2]

    if roi.size == 0:
        return False, False, "single"

    trigger_x = int(mask.shape[1] * action_x)
    trigger_x = max(x1 + 1, min(x2, trigger_x))
    look_x = min(x2, trigger_x + look_ahead)
    cactus_found = False

    for contour in contours(roi):
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        real_x = x1 + x
        bottom = (y + height) / roi.shape[0]

        if height < MIN_HEIGHT:
            continue

        is_bird = BIRD_MIN <= bottom < GROUND
        is_cactus = bottom >= GROUND
        is_wide = width >= MEDIUM_WIDTH

        if is_bird and real_x <= trigger_x and area >= duck_area:
            return True, True, "single"
        if is_cactus and real_x <= trigger_x:
            if area >= jump_area or is_wide:
                cactus_found = True

    if not cactus_found:
        return False, False, "single"

    group = group_by_width(roi, x1, trigger_x, look_x)
    return True, False, group


def do_action(action):
    global is_ducking

    if action == "duck":
        if not is_ducking:
            pyautogui.keyDown("down")
            is_ducking = True
        return

    if is_ducking:
        pyautogui.keyUp("down")
        is_ducking = False

    if action == "jump":
        press("space")


def speed_value(speed, start, end):
    progress = (speed - START_SPEED) / (MAX_SPEED - START_SPEED)
    progress = max(0.0, min(1.0, progress))
    return start + (end - start) * progress


def current_speed(seconds):
    return min(START_SPEED + seconds * FPS * ACCELERATION, MAX_SPEED)


def min_delay(delay):
    return delay * START_SPEED / MAX_SPEED


def get_thresholds(speed):
    factor = speed_value(speed, 1.0, MIN_FACTOR)
    return max(1, int(JUMP_AREA * factor)), max(1, int(DUCK_AREA * factor))


def get_action_x(speed, width):
    return ACTION_X + speed * FPS * LEAD_TIME / max(1, width)


def get_look_ahead(speed):
    pixels = int(speed * FPS * LOOK_TIME)
    return max(LOOK_MIN, min(LOOK_MAX, pixels))


def get_jump_cooldown(speed):
    return speed_value(speed, JUMP_COOLDOWN, min_delay(JUMP_COOLDOWN))


def get_duck_after_jump(speed, group):
    delay, hold, max_hold = DUCK_TIMES[group]
    delay = speed_value(speed, delay, min_delay(delay))
    hold = speed_value(speed, hold, max_hold)
    return delay, delay + hold


def get_bird_hold(speed):
    return speed_value(speed, BIRD_HOLD[0], BIRD_HOLD[1])


def game_over(gray):
    height, width = gray.shape[:2]
    center = gray[
        int(height * 0.28):int(height * 0.58),
        int(width * 0.28):int(width * 0.72),
    ]
    mask = cv2.inRange(center, 40, 215)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    for contour in contours(mask):
        _, _, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        is_wide = width > center.shape[1] * 0.22
        is_high = height > center.shape[0] * 0.08

        if area > 120 and is_wide and is_high:
            return True

    return False


def draw_debug(frame, mask, action_x, action, fps):
    image = frame.copy()
    x1, y1, x2, y2 = zone_pixels(image.shape)
    trigger_x = max(x1 + 1, min(x2, int(image.shape[1] * action_x)))

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(image, (trigger_x, y1), (trigger_x, y2), (255, 180, 0), 2)

    for contour in contours(mask[y1:y2, x1:x2]):
        if cv2.contourArea(contour) < MIN_AREA:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(
            image,
            (x1 + x, y1 + y),
            (x1 + x + width, y1 + y + height),
            (0, 0, 255),
            1,
        )

    text = "ACTION: " + (action or "run").upper()
    cv2.putText(image, f"FPS: {fps:.1f}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2)
    cv2.putText(image, text, (12, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (30, 30, 30), 2)
    return image


def choose_action(found, is_bird, can_jump):
    if not found:
        return None
    if is_bird:
        return "duck"
    return "jump" if can_jump else None


def parse_args():
    parser = argparse.ArgumentParser(description="Бот для Chrome Dino")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--reselect", action="store_true")
    return parser.parse_args()


def restart_game(now, speed):
    do_action(None)
    press("space")
    return now + get_jump_cooldown(speed)


def main():
    args = parse_args()
    debug = DEBUG or args.debug

    with mss.mss() as sct:
        region = get_region(sct, args.reselect)
        print("Откройте браузер с игрой. "
              "Старт через 3 секунды...")
        time.sleep(3)
        press("space")

        start = time.perf_counter()
        last_frame = start
        last_restart = 0.0
        game_over_hits = 0
        next_jump = 0.0
        duck_start = 0.0
        duck_until = 0.0
        bird_until = 0.0

        try:
            while True:
                now = time.perf_counter()
                fps = 1.0 / max(now - last_frame, 0.000001)
                last_frame = now
                speed = current_speed(now - start)

                frame = grab_frame(sct, region)
                gray, mask = prepare_frame(frame)
                game_over_hits = game_over_hits + 1 if game_over(gray) else 0

                if game_over_hits >= 6:
                    if now - last_restart > RESTART_DELAY:
                        next_jump = restart_game(now, START_SPEED)
                        start = last_restart = now
                        game_over_hits = 0
                        duck_start = duck_until = bird_until = 0.0
                    continue

                jump_area, duck_area = get_thresholds(speed)
                found, is_bird, group = detect_obstacle(
                    mask,
                    jump_area,
                    duck_area,
                    get_action_x(speed, region["width"]),
                    get_look_ahead(speed),
                )
                action = choose_action(found, is_bird, now >= next_jump)

                if action == "duck" and is_bird:
                    bird_until = now + get_bird_hold(speed)
                if now < bird_until:
                    action = "duck"
                elif action is None and duck_start <= now < duck_until:
                    action = "duck"

                do_action(action)

                if action == "jump":
                    next_jump = now + get_jump_cooldown(speed)
                    delay, finish = get_duck_after_jump(speed, group)
                    duck_start = now + delay
                    duck_until = now + finish

                if debug:
                    action_x = get_action_x(speed, region["width"])
                    image = draw_debug(frame, mask, action_x, action, fps)
                    cv2.imshow("Dino Bot Debug", image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            do_action(None)
            if debug:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
