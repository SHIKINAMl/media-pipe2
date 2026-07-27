import argparse

from mp_options.hand_gestures import (
    classify_gesture,
    classify_open_close,
    create_hand_label_overlay,
)
from mp_options.hand_pointer import create_hand_pointer_overlay
from mp_options.hand_pose_detection import run_hand_pose_detection
from mp_options.smoothing import (
    create_multi_hand_landmark_smoother,
    create_single_pose_landmark_smoother,
)


def build_hand_pipeline(enable_position_smoothing, position_window, gesture_mode, label_window):
    hand_filter = None
    overlay = None
    title_options = []

    if enable_position_smoothing:
        hand_filter = create_multi_hand_landmark_smoother(window_size=position_window)
        title_options.append(f"HandPosSmooth{position_window}")
    else:
        title_options.append("HandDetectOnly")

    if gesture_mode == "gesture":
        overlay = create_hand_label_overlay(
            classifier=classify_gesture,
            window_size=label_window,
        )
        title_options.append(f"Gesture{label_window}")
    elif gesture_mode == "open-close":
        overlay = create_hand_label_overlay(
            classifier=classify_open_close,
            window_size=label_window,
            vote_labels=("OPEN", "CLOSE"),
        )
        title_options.append(f"OpenClose{label_window}")
    elif gesture_mode == "pointer":
        overlay = create_hand_pointer_overlay(window_size=label_window)
        title_options.append(f"Pointer{label_window}")

    return hand_filter, overlay, title_options


def build_pose_pipeline(enable_position_smoothing, position_window):
    pose_filter = None
    title_options = []

    if enable_position_smoothing:
        pose_filter = create_single_pose_landmark_smoother(window_size=position_window)
        title_options.append(f"PosePosSmooth{position_window}")
    else:
        title_options.append("PoseDetectOnly")

    return pose_filter, title_options


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MediaPipe hand and pose pipeline in one camera loop."
    )
    parser.add_argument(
        "-hp",
        "--hand-position-smoothing",
        nargs="?",
        const=5,
        type=int,
        default=None,
        metavar="WINDOW",
        help="Enable hand landmark position smoothing with optional window size.",
    )
    parser.add_argument(
        "-pp",
        "--pose-position-smoothing",
        nargs="?",
        const=5,
        type=int,
        default=None,
        metavar="WINDOW",
        help="Enable pose landmark position smoothing with optional window size.",
    )
    parser.add_argument(
        "-g",
        "--gesture",
        nargs="+",
        type=int,
        default=None,
        metavar=("MODE", "WINDOW"),
        help="Enable hand overlay mode with mode and optional label smoothing window: 0=open/close, 1=gesture, 2=pointer.",
    )
    parser.add_argument(
        "-a",
        "--avatar",
        action="store_true",
        help="Launch avatar window that draws pose points and bones.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    hand_smoothing_enabled = args.hand_position_smoothing is not None
    hand_window = args.hand_position_smoothing if args.hand_position_smoothing is not None else 0

    pose_smoothing_enabled = args.pose_position_smoothing is not None
    pose_window = args.pose_position_smoothing if args.pose_position_smoothing is not None else 0

    gesture_mode = "none"
    label_window = 5
    if args.gesture is not None:
        if len(args.gesture) not in (1, 2):
            raise SystemExit("-g expects one or two integers: MODE [WINDOW]")

        if args.gesture[0] == 1:
            gesture_mode = "gesture"
        elif args.gesture[0] == 2:
            gesture_mode = "pointer"
        else:
            gesture_mode = "open-close"

        if len(args.gesture) == 2:
            label_window = args.gesture[1]

    hand_filter, hand_overlay, hand_title_options = build_hand_pipeline(
        enable_position_smoothing=hand_smoothing_enabled,
        position_window=max(1, hand_window),
        gesture_mode=gesture_mode,
        label_window=max(1, label_window),
    )
    pose_filter, pose_title_options = build_pose_pipeline(
        enable_position_smoothing=pose_smoothing_enabled,
        position_window=max(1, pose_window),
    )

    title = "MediaPipe Hand+Pose options | " + " | ".join(hand_title_options + pose_title_options)

    run_hand_pose_detection(
        window_title=title,
        hand_filter=hand_filter,
        pose_filter=pose_filter,
        hand_overlay_callback=hand_overlay,
        enable_avatar_window=args.avatar,
    )


if __name__ == "__main__":
    main()