import argparse

from mp_options.pose_detection import run_pose_detection
from mp_options.smoothing import create_single_pose_landmark_smoother


def build_pipeline(enable_position_smoothing, position_window):
    pose_filter = None
    title_options = ["MediaPipe Pose options"]

    if enable_position_smoothing:
        pose_filter = create_single_pose_landmark_smoother(window_size=position_window)
        title_options.append(f"PosSmooth{position_window}")
    else:
        title_options.append("DetectOnly")

    return pose_filter, " | ".join(title_options)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MediaPipe pose pipeline by composing detection/smoothing options."
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
    return parser.parse_args()


def main():
    args = parse_args()

    enable_position_smoothing = args.position_smoothing is not None
    position_window = args.position_smoothing if args.position_smoothing is not None else 0

    pose_filter, title = build_pipeline(
        enable_position_smoothing=enable_position_smoothing,
        position_window=max(1, position_window),
    )

    run_pose_detection(
        window_title=title,
        pose_filter=pose_filter,
    )


if __name__ == "__main__":
    main()
