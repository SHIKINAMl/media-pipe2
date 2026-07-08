import argparse

from mp_options.hand_detection import run_hand_detection
from mp_options.hand_gestures import (
    classify_gesture,
    classify_open_close,
    create_hand_label_overlay,
)
from mp_options.smoothing import create_multi_hand_landmark_smoother


def build_pipeline(
    enable_position_smoothing,
    position_window,
    gesture_mode,
    label_window,
):
    hand_filter = None
    overlay = None
    title_options = ["MediaPipe Hand options"]

    if enable_position_smoothing:
        hand_filter = create_multi_hand_landmark_smoother(window_size=position_window)
        title_options.append(f"PosSmooth{position_window}")

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
    else:
        title_options.append("DetectOnly")

    return hand_filter, overlay, " | ".join(title_options)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MediaPipe hand pipeline by composing detection/smoothing/gesture options."
    )
    parser.add_argument(
        "-p",
        "--position-smoothing",
        nargs="?",
        const=5,
        type=int,
        default=None,
        metavar="WINDOW",
        help="Enable landmark position smoothing with optional window size.",
    )
    parser.add_argument(
        "-g",
        "--gesture",
        nargs=2,
        type=int,
        choices=[0, 1],
        default=None,
        metavar=("MODE", "WINDOW"),
        help="Enable overlay mode with mode and label smoothing window: 0=open/close, 1=gesture.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    position_window = args.position_smoothing if args.position_smoothing is not None else 0
    enable_position_smoothing = args.position_smoothing is not None

    gesture_mode = "none"
    label_window = 5
    if args.gesture is not None:
        gesture_mode = "gesture" if args.gesture[0] == 1 else "open-close"
        label_window = args.gesture[1]

    hand_filter, overlay, title = build_pipeline(
        enable_position_smoothing=enable_position_smoothing,
        position_window=max(1, position_window),
        gesture_mode=gesture_mode,
        label_window=max(1, label_window),
    )

    run_hand_detection(
        window_title=title,
        hand_filter=hand_filter,
        overlay_callback=overlay,
    )


if __name__ == "__main__":
    main()
