import shutil
import tempfile
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision

from mp_options.pose_avatar import PoseAvatarWindow


POSE_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "pose" / "pose_landmarker_heavy.task"


def runtime_model_path(path: Path) -> Path:
    if all(ord(ch) < 128 for ch in str(path)):
        return path
    cache = Path(tempfile.gettempdir()) / "mediapipe_models" / path.name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists() or cache.stat().st_size != path.stat().st_size:
        shutil.copy2(path, cache)
    return cache


def draw_pose(frame, landmarks, connections):
    h, w = frame.shape[:2]

    for connection in connections:
        p1 = landmarks[connection.start]
        p2 = landmarks[connection.end]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)

    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)


def run_pose_detection(
    window_title="MediaPipe Pose Runner",
    pose_filter=None,
    overlay_callback=None,
    display_width=1280,
    display_height=720,
    enable_avatar_window=False,
    avatar_window_title="Pose Avatar",
    avatar_width=640,
    avatar_height=360,
):
    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found: {POSE_MODEL_PATH}")

    model_path = runtime_model_path(POSE_MODEL_PATH)
    latest_poses = []

    def on_result(result, _output_image, _timestamp_ms):
        nonlocal latest_poses
        latest_poses = result.pose_landmarks or []

    base_options = tasks.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=1,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open a web camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, display_width, display_height)
    avatar_window = None
    if enable_avatar_window:
        avatar_window = PoseAvatarWindow(
            window_title=avatar_window_title,
            width=avatar_width,
            height=avatar_height,
        )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (640, 360))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.perf_counter() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            if latest_poses:
                poses_to_draw = pose_filter(latest_poses) if pose_filter is not None else latest_poses
                draw_pose(
                    frame,
                    poses_to_draw[0],
                    vision.PoseLandmarksConnections.POSE_LANDMARKS,
                )
                if avatar_window is not None:
                    avatar_window.render(
                        poses_to_draw[0],
                        vision.PoseLandmarksConnections.POSE_LANDMARKS,
                    )
                if overlay_callback is not None:
                    overlay_callback(frame, poses_to_draw)

            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
