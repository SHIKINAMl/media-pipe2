from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import cv2
import mediapipe as mp
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision

from tracking.hand_gesture_classifier import assign_hand_sides


ROOT_DIR = Path(__file__).resolve().parent.parent
HAND_MODEL_PATH = ROOT_DIR / "models" / "hand" / "hand_landmarker.task"
POSE_MODEL_CANDIDATES = (
    ROOT_DIR / "models" / "pose" / "pose_landmarker_lite.task",
    ROOT_DIR / "models" / "pose" / "pose_landmarker_full.task",
    ROOT_DIR / "models" / "pose" / "pose_landmarker_heavy.task",
)

# Detector states. Camera / model problems are system errors and are shown to the
# operator; "no person in view" is not an error and stays STATUS_READY.
STATUS_STARTING = "starting"
STATUS_READY = "ready"
STATUS_MODEL_MISSING = "model_missing"
STATUS_CAMERA_UNAVAILABLE = "camera_unavailable"
STATUS_CAMERA_LOST = "camera_lost"
STATUS_ERROR = "error"
STATUS_STOPPED = "stopped"
ERROR_STATUSES = (STATUS_MODEL_MISSING, STATUS_CAMERA_UNAVAILABLE, STATUS_CAMERA_LOST, STATUS_ERROR)

CAMERA_LOST_FRAMES = 45


@dataclass
class DetectionSnapshot:
    hands: list = field(default_factory=list)
    poses: list = field(default_factory=list)
    handedness: list[str] = field(default_factory=list)
    hand_sides: list[str] = field(default_factory=list)
    timestamp_ms: int = 0

    def hand_for_side(self, side: str):
        for index, hand_side in enumerate(self.hand_sides):
            if hand_side == side and index < len(self.hands):
                return self.hands[index]
        return None


def _runtime_model_path(path: Path) -> Path:
    if all(ord(character) < 128 for character in str(path)):
        return path
    cache = Path(tempfile.gettempdir()) / "mediapipe_models" / path.name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists() or cache.stat().st_size != path.stat().st_size:
        shutil.copy2(path, cache)
    return cache


def _pose_model_path() -> Path:
    for path in POSE_MODEL_CANDIDATES:
        if path.exists():
            return path
    return POSE_MODEL_CANDIDATES[0]


class HandPoseDetector:
    """Non-blocking MediaPipe hand and pose detector for GUI event loops.

    The camera and models are opened once and shared by every screen.
    """

    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 640,
        frame_height: int = 360,
        num_hands: int = 2,
    ) -> None:
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.num_hands = num_hands
        self.status = STATUS_STARTING
        self.error_detail = ""
        self._capture = None
        self._hand_landmarker = None
        self._pose_landmarker = None
        self._snapshot = DetectionSnapshot()
        self._lock = Lock()
        self._last_timestamp_ms = 0
        self._failed_reads = 0

    @property
    def model_error(self) -> str | None:
        missing = []
        if not HAND_MODEL_PATH.exists():
            missing.append(str(HAND_MODEL_PATH))
        pose_path = _pose_model_path()
        if not pose_path.exists():
            missing.append(str(pose_path))
        if missing:
            return "Model file not found: " + ", ".join(missing)
        return None

    @property
    def has_error(self) -> bool:
        return self.status in ERROR_STATUSES

    def start(self) -> None:
        """Open the camera and models. Failures are recorded in ``status`` instead of raised."""
        self.close()
        self._failed_reads = 0
        error = self.model_error
        if error is not None:
            self._fail(STATUS_MODEL_MISSING, error)
            return
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            self._fail(STATUS_CAMERA_UNAVAILABLE, f"Could not open web camera #{self.camera_index}.")
            return
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self._capture = capture

        try:
            hand_options = vision.HandLandmarkerOptions(
                base_options=tasks.BaseOptions(model_asset_path=str(_runtime_model_path(HAND_MODEL_PATH))),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_hands=self.num_hands,
                result_callback=self._on_hand_result,
            )
            pose_options = vision.PoseLandmarkerOptions(
                base_options=tasks.BaseOptions(model_asset_path=str(_runtime_model_path(_pose_model_path()))),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_poses=1,
                result_callback=self._on_pose_result,
            )
            self._hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            self._pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        except Exception as error:  # MediaPipe raises plain RuntimeError / ValueError
            self.close()
            self._fail(STATUS_ERROR, f"MediaPipe initialisation failed: {error}")
            return
        self.status = STATUS_READY
        self.error_detail = ""

    def update(self) -> DetectionSnapshot:
        if self._capture is None or self._hand_landmarker is None or self._pose_landmarker is None:
            return self.snapshot()
        ok, frame = self._capture.read()
        if not ok:
            self._failed_reads += 1
            if self._failed_reads >= CAMERA_LOST_FRAMES:
                self._fail(STATUS_CAMERA_LOST, "Camera stopped delivering frames.")
                self._clear_snapshot()
            return self.snapshot()
        self._failed_reads = 0
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(time.perf_counter() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        try:
            self._hand_landmarker.detect_async(image, timestamp_ms)
            self._pose_landmarker.detect_async(image, timestamp_ms)
        except Exception as error:
            self._fail(STATUS_ERROR, f"MediaPipe inference failed: {error}")
        return self.snapshot()

    def snapshot(self) -> DetectionSnapshot:
        with self._lock:
            hands = list(self._snapshot.hands)
            poses = list(self._snapshot.poses)
            handedness = list(self._snapshot.handedness)
            timestamp_ms = self._snapshot.timestamp_ms
        return DetectionSnapshot(
            hands=hands,
            poses=poses,
            handedness=handedness,
            hand_sides=assign_hand_sides(hands, handedness, poses),
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        if self._hand_landmarker is not None:
            self._hand_landmarker.close()
            self._hand_landmarker = None
        if self._pose_landmarker is not None:
            self._pose_landmarker.close()
            self._pose_landmarker = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if not self.has_error:
            self.status = STATUS_STOPPED

    def _fail(self, status: str, detail: str) -> None:
        self.status = status
        self.error_detail = detail

    def _clear_snapshot(self) -> None:
        with self._lock:
            self._snapshot = DetectionSnapshot()

    def _on_hand_result(self, result, _output_image, timestamp_ms) -> None:
        handedness = [values[0].category_name if values else "Unknown" for values in result.handedness or []]
        with self._lock:
            self._snapshot.hands = result.hand_landmarks or []
            self._snapshot.handedness = handedness
            self._snapshot.timestamp_ms = timestamp_ms

    def _on_pose_result(self, result, _output_image, timestamp_ms) -> None:
        with self._lock:
            self._snapshot.poses = result.pose_landmarks or []
            self._snapshot.timestamp_ms = timestamp_ms

    def __enter__(self) -> "HandPoseDetector":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()
