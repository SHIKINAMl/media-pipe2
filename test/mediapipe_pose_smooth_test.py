from collections import deque
from types import SimpleNamespace

from mediapipe_pose_test import run_pose_test


SMOOTHING_WINDOW = 5


def create_pose_filter(window_size=SMOOTHING_WINDOW):
    history = deque(maxlen=window_size)

    def smooth_poses(poses):
        if not poses:
            history.clear()
            return poses

        current = poses[0]
        history.append(current)

        count = len(history)
        landmark_count = len(current)
        smoothed = []

        for i in range(landmark_count):
            sum_x = sum(pose[i].x for pose in history)
            sum_y = sum(pose[i].y for pose in history)
            sum_z = sum(getattr(pose[i], "z", 0.0) for pose in history)
            smoothed.append(
                SimpleNamespace(
                    x=sum_x / count,
                    y=sum_y / count,
                    z=sum_z / count,
                )
            )

        return [smoothed]

    return smooth_poses


def main():
    run_pose_test(
        window_title="MediaPipe Pose Smooth Test",
        pose_filter=create_pose_filter(),
    )


if __name__ == "__main__":
    main()
