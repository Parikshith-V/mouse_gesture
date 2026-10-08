"""
Hand Gesture Mouse Controller
==============================
Controls your mouse using hand gestures via webcam.
Compatible with mediapipe 0.10+

Requirements:
    pip install mediapipe opencv-python pyautogui numpy

Gestures:
    - Index finger up only     → Move cursor
    - Index + Middle up        → Freeze cursor (neutral)
    - Pinch (thumb + index)    → Left click
    - Pinch (thumb + middle)   → Right click
    - All fingers up (open palm) → Scroll mode (move hand up/down)
    - Fist (all fingers closed) → Drag (hold left click)

Run:
    python gesture_mouse.py
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
import pyautogui
import numpy as np
import time
import urllib.request
import os

# ─── Config ───────────────────────────────────────────────────────────────────
CAM_INDEX        = 0
SMOOTH_FACTOR    = 5
CLICK_THRESHOLD  = 0.04
SCROLL_SPEED     = 15
CLICK_COOLDOWN   = 0.4
FRAME_REDUCTION  = 100
MODEL_PATH       = "hand_landmarker.task"
MODEL_URL        = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

# ─── Download model if needed ─────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"[INFO] Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[INFO] Model downloaded to {MODEL_PATH}")

# ─── Mediapipe Setup (Tasks API) ──────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.75,
    min_hand_presence_confidence=0.75,
    min_tracking_confidence=0.75,
    running_mode=mp_vision.RunningMode.VIDEO,
)
detector = HandLandmarker.create_from_options(options)

# Hand connections for drawing skeleton manually
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ─── Helpers ──────────────────────────────────────────────────────────────────
screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0

def smooth(curr, prev, factor):
    return prev + (curr - prev) / factor

def distance(p1, p2):
    return np.hypot(p1.x - p2.x, p1.y - p2.y)

def fingers_up(lm):
    tips   = [4, 8, 12, 16, 20]
    joints = [3, 6, 10, 14, 18]
    result = []
    result.append(1 if lm[4].x < lm[3].x else 0)
    for tip, joint in zip(tips[1:], joints[1:]):
        result.append(1 if lm[tip].y < lm[joint].y else 0)
    return result

def draw_landmarks_on_frame(frame, landmarks):
    """Draw hand skeleton using OpenCV directly."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 180), -1)

def draw_status(frame, gesture, fps):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 55), (15, 15, 15), -1)
    cv2.putText(frame, f"Gesture: {gesture}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 180), 2)
    cv2.putText(frame, f"FPS: {fps}", (w - 110, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
    cv2.rectangle(frame, (FRAME_REDUCTION, FRAME_REDUCTION),
                  (w - FRAME_REDUCTION, h - FRAME_REDUCTION),
                  (0, 255, 180), 1)

def draw_legend(frame):
    legend = [
        ("INDEX only",     "Move cursor"),
        ("INDEX + MIDDLE", "Freeze / Neutral"),
        ("THUMB + INDEX",  "Left Click"),
        ("THUMB + MIDDLE", "Right Click"),
        ("OPEN PALM",      "Scroll (move up/down)"),
        ("FIST",           "Drag (hold LMB)"),
    ]
    h, w = frame.shape[:2]
    x_start = w - 310
    y_start = 70
    cv2.rectangle(frame, (x_start - 8, y_start - 18),
                  (w - 5, y_start + len(legend) * 22 + 5),
                  (20, 20, 20), -1)
    for i, (g, action) in enumerate(legend):
        y = y_start + i * 22
        cv2.putText(frame, f"{g}: {action}", (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)

# ─── State ────────────────────────────────────────────────────────────────────
last_click_time = 0
is_dragging     = False
scroll_ref_y    = None

# ─── Main Loop ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time  = time.time()
frame_ts   = 0  # millisecond timestamp for Tasks API

print("[INFO] Hand Gesture Mouse Controller started.")
print("[INFO] Press 'Q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("[ERROR] Camera read failed.")
        break

    frame    = cv2.flip(frame, 1)
    h, w     = frame.shape[:2]
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tasks API requires a mediapipe Image + timestamp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_ts += 1  # increment by 1ms each frame (monotonic)
    result   = detector.detect_for_video(mp_image, frame_ts)

    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time + 1e-6))
    prev_time = curr_time

    gesture = "No Hand Detected"

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]

        draw_landmarks_on_frame(frame, lm)

        f = fingers_up(lm)

        index_tip  = lm[8]
        middle_tip = lm[12]
        thumb_tip  = lm[4]

        now = time.time()

        # ── FIST → Drag ──────────────────────────────────────────
        if f == [0, 0, 0, 0, 0]:
            gesture = "FIST  →  Drag"
            if not is_dragging:
                pyautogui.mouseDown()
                is_dragging = True
            cx = np.interp(lm[5].x * w, [FRAME_REDUCTION, w - FRAME_REDUCTION], [0, screen_w])
            cy = np.interp(lm[5].y * h, [FRAME_REDUCTION, h - FRAME_REDUCTION], [0, screen_h])
            prev_x = smooth(cx, prev_x, SMOOTH_FACTOR)
            prev_y = smooth(cy, prev_y, SMOOTH_FACTOR)
            pyautogui.moveTo(prev_x, prev_y)

        # ── OPEN PALM → Scroll ────────────────────────────────────
        elif f == [1, 1, 1, 1, 1]:
            gesture = "OPEN PALM  →  Scroll"
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
            if scroll_ref_y is None:
                scroll_ref_y = index_tip.y
            else:
                delta = scroll_ref_y - index_tip.y
                if abs(delta) > 0.01:
                    pyautogui.scroll(int(delta * SCROLL_SPEED * 10))
            scroll_ref_y = index_tip.y

        else:
            scroll_ref_y = None

            d_left  = distance(thumb_tip, index_tip)
            d_right = distance(thumb_tip, middle_tip)

            # ── THUMB + INDEX pinch → Left Click ──────────────────
            if d_left < CLICK_THRESHOLD and (now - last_click_time) > CLICK_COOLDOWN:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                pyautogui.click()
                last_click_time = now
                gesture = "THUMB+INDEX  →  LEFT CLICK ✓"
                cv2.circle(frame, (int(index_tip.x * w), int(index_tip.y * h)), 18, (0, 255, 0), -1)

            # ── THUMB + MIDDLE pinch → Right Click ────────────────
            elif d_right < CLICK_THRESHOLD and (now - last_click_time) > CLICK_COOLDOWN:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                pyautogui.rightClick()
                last_click_time = now
                gesture = "THUMB+MIDDLE  →  RIGHT CLICK ✓"
                cv2.circle(frame, (int(middle_tip.x * w), int(middle_tip.y * h)), 18, (0, 100, 255), -1)

            # ── INDEX only → Move ─────────────────────────────────
            elif f[1] == 1 and f[2] == 0:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                gesture = "INDEX  →  Move Cursor"
                cx = np.interp(index_tip.x * w, [FRAME_REDUCTION, w - FRAME_REDUCTION], [0, screen_w])
                cy = np.interp(index_tip.y * h, [FRAME_REDUCTION, h - FRAME_REDUCTION], [0, screen_h])
                prev_x = smooth(cx, prev_x, SMOOTH_FACTOR)
                prev_y = smooth(cy, prev_y, SMOOTH_FACTOR)
                pyautogui.moveTo(prev_x, prev_y)

            # ── INDEX + MIDDLE → Neutral / Freeze ─────────────────
            elif f[1] == 1 and f[2] == 1:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                gesture = "INDEX+MIDDLE  →  Freeze"

            else:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                gesture = "Unknown gesture"

    draw_status(frame, gesture, fps)
    draw_legend(frame)

    cv2.imshow("Hand Gesture Mouse Controller  |  Press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ─── Cleanup ──────────────────────────────────────────────────────────────────
if is_dragging:
    pyautogui.mouseUp()

cap.release()
cv2.destroyAllWindows()
detector.close()
print("[INFO] Exited cleanly.")