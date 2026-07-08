from collections import deque
from types import SimpleNamespace


def _snapshot_landmarks(landmarks):
    return [
        SimpleNamespace(
            x=float(lm.x),
            y=float(lm.y),
            z=float(getattr(lm, "z", 0.0)),
        )
        for lm in landmarks
    ]


def create_multi_hand_landmark_smoother(window_size=5):
    histories = {}

    def smooth_hands(hands, handedness_labels=None):
        active_keys = set()
        smoothed_hands = []
        label_counts = {}

        for i, hand in enumerate(hands):
            track_key = i
            if handedness_labels is not None and i < len(handedness_labels):
                label = handedness_labels[i] or "Unknown"
                rank = label_counts.get(label, 0)
                label_counts[label] = rank + 1
                track_key = f"{label}:{rank}"

            active_keys.add(track_key)
            history = histories.setdefault(track_key, deque(maxlen=window_size))
            history.append(_snapshot_landmarks(hand))

            landmark_count = len(hand)
            count = len(history)
            smoothed_hand = []

            for j in range(landmark_count):
                sum_x = sum(past_hand[j].x for past_hand in history)
                sum_y = sum(past_hand[j].y for past_hand in history)
                sum_z = sum(getattr(past_hand[j], "z", 0.0) for past_hand in history)
                smoothed_hand.append(
                    SimpleNamespace(
                        x=sum_x / count,
                        y=sum_y / count,
                        z=sum_z / count,
                    )
                )

            smoothed_hands.append(smoothed_hand)

        for key in list(histories.keys()):
            if key not in active_keys:
                del histories[key]

        return smoothed_hands

    return smooth_hands


def create_single_pose_landmark_smoother(window_size=5):
    history = deque(maxlen=window_size)

    def smooth_poses(poses):
        if not poses:
            history.clear()
            return poses

        current = poses[0]
        history.append(_snapshot_landmarks(current))

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
