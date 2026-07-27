import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision

from mp_options.hand_detection import HAND_MODEL_PATH, draw_hands, runtime_model_path
from mp_options.pose_avatar import PoseAvatarWindow
from mp_options.pose_detection import POSE_MODEL_PATH, draw_pose


def run_hand_pose_detection(
    window_title="MediaPipe Hand + Pose Runner",
    hand_filter=None,
    pose_filter=None,
    hand_overlay_callback=None,
    pose_overlay_callback=None,
    num_hands=2,
    display_width=1280,
    display_height=720,
    enable_avatar_window=False,
    avatar_window_title="Pose Avatar",
    avatar_width=640,
    avatar_height=360,
):
    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found: {HAND_MODEL_PATH}")
    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found: {POSE_MODEL_PATH}")

    hand_model_path = runtime_model_path(HAND_MODEL_PATH)
    pose_model_path = runtime_model_path(POSE_MODEL_PATH)

    latest_hands = []
    latest_handedness_labels = []
    latest_poses = []

    def on_hand_result(result, _output_image, _timestamp_ms):
        nonlocal latest_hands, latest_handedness_labels
        latest_hands = result.hand_landmarks or []
        latest_handedness_labels = []
        for handedness_list in result.handedness or []:
            if handedness_list and len(handedness_list) > 0:
                latest_handedness_labels.append(handedness_list[0].category_name)
            else:
                latest_handedness_labels.append("Unknown")

    def on_pose_result(result, _output_image, _timestamp_ms):
        nonlocal latest_poses
        latest_poses = result.pose_landmarks or []

    hand_base_options = tasks.BaseOptions(model_asset_path=str(hand_model_path))
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=num_hands,
        result_callback=on_hand_result,
    )

    pose_base_options = tasks.BaseOptions(model_asset_path=str(pose_model_path))
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=1,
        result_callback=on_pose_result,
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

    with (
        vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker,
        vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
    ):
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (640, 360))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.perf_counter() * 1000)

            hand_landmarker.detect_async(mp_image, timestamp_ms)
            pose_landmarker.detect_async(mp_image, timestamp_ms)

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
                if pose_overlay_callback is not None:
                    pose_overlay_callback(frame, poses_to_draw)

            if latest_hands:
                if hand_filter is not None:
                    try:
                        hands_to_draw = hand_filter(latest_hands, latest_handedness_labels)
                    except TypeError:
                        hands_to_draw = hand_filter(latest_hands)
                else:
                    hands_to_draw = latest_hands

                draw_hands(
                    frame,
                    hands_to_draw,
                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                )
                if hand_overlay_callback is not None:
                    hand_overlay_callback(frame, hands_to_draw, latest_handedness_labels)

            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()