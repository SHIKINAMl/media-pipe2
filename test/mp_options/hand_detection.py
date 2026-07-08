import shutil
import tempfile
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision


HAND_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "hand" / "hand_landmarker.task"


def runtime_model_path(path: Path) -> Path:
    if all(ord(ch) < 128 for ch in str(path)):
        return path
    cache = Path(tempfile.gettempdir()) / "mediapipe_models" / path.name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists() or cache.stat().st_size != path.stat().st_size:
        shutil.copy2(path, cache)
    return cache


def draw_hands(frame, hands, connections):
    h, w = frame.shape[:2]
    for hand in hands:
        for c in connections:
            p1 = hand[c.start]
            p2 = hand[c.end]
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)

        for lm in hand:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)


def run_hand_detection(
    window_title="MediaPipe Hand Parts Runner",
    hand_filter=None,
    overlay_callback=None,
    num_hands=2,
    display_width=1280,
    display_height=720,
):
    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found: {HAND_MODEL_PATH}")

    model_path = runtime_model_path(HAND_MODEL_PATH)
    latest_hands = []
    latest_handedness_labels = []

    def on_result(result, _output_image, _timestamp_ms):
        nonlocal latest_hands, latest_handedness_labels
        latest_hands = result.hand_landmarks or []
        latest_handedness_labels = []
        for handedness_list in result.handedness or []:
            if handedness_list and len(handedness_list) > 0:
                latest_handedness_labels.append(handedness_list[0].category_name)
            else:
                latest_handedness_labels.append("Unknown")

    base_options = tasks.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=num_hands,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open a web camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, display_width, display_height)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (640, 360))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.perf_counter() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            if latest_hands:
                hands_to_draw = hand_filter(latest_hands) if hand_filter is not None else latest_hands
                draw_hands(
                    frame,
                    hands_to_draw,
                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                )
                if overlay_callback is not None:
                    overlay_callback(frame, hands_to_draw, latest_handedness_labels)

            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
