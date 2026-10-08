# Hand Gesture Mouse Controller

Control your mouse cursor using hand gestures via webcam, built with MediaPipe's Hand Landmarker (Tasks API) and OpenCV.

## Features

| Gesture | Action |
|---|---|
| Index finger up only | Move cursor |
| Index + Middle up | Freeze cursor (neutral) |
| Thumb + Index pinch | Left click |
| Thumb + Middle pinch | Right click |
| Open palm (all fingers up) | Scroll mode (move hand up/down) |
| Fist (all fingers closed) | Drag (hold left click) |

## How it works

- **Hand tracking**: Uses MediaPipe's pre-trained Hand Landmarker model to detect 21 3D hand keypoints per frame in real time.
- **Gesture classification**: A rule-based system interprets finger states (up/down) and fingertip distances (for pinch detection) to map hand poses to mouse actions.
- **Cursor smoothing**: An exponential moving average smooths raw landmark coordinates before mapping them to screen space, reducing jitter.

## Requirements

```
pip install -r requirements.txt
```

Tested with `mediapipe>=0.10`, which uses the newer Tasks API (`HandLandmarker`) rather than the legacy `mp.solutions.hands`.

## Usage

```
python gesture_mouse.py
```

- The hand landmark model (`hand_landmarker.task`) is downloaded automatically on first run.
- Press `Q` to quit.

## Notes

- Works best with a single hand in frame (`num_hands=1`) and consistent lighting.
- `FRAME_REDUCTION`, `SMOOTH_FACTOR`, and `CLICK_THRESHOLD` in the config section can be tuned for sensitivity.
