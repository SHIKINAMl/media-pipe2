import shutil
import tempfile
import time
from pathlib import Path

import cv2

import mediapipe as mp
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose" / "pose_landmarker_heavy.task"


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

    for c in connections:
        p1 = landmarks[c.start]
        p2 = landmarks[c.end]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)

    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {MODEL_PATH}")

    latest_poses = []

    def on_result(result, _output_image, _timestamp_ms):
        nonlocal latest_poses
        latest_poses = result.pose_landmarks or []

    model_path = runtime_model_path(MODEL_PATH)

    base_options = tasks.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=1,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webカメラを開けませんでした。")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

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
                draw_pose(
                    frame,
                    latest_poses[0],
                    vision.PoseLandmarksConnections.POSE_LANDMARKS,
                )

            cv2.imshow("MediaPipe Pose Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
