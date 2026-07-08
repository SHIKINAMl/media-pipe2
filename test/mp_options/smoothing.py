from collections import deque
from types import SimpleNamespace


def create_multi_hand_landmark_smoother(window_size=5):
    histories = {}

    def smooth_hands(hands):
        active_indices = set()
        smoothed_hands = []

        for i, hand in enumerate(hands):
            active_indices.add(i)
            history = histories.setdefault(i, deque(maxlen=window_size))
            history.append(hand)

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

        for i in list(histories.keys()):
            if i not in active_indices:
                del histories[i]

        return smoothed_hands

    return smooth_hands


def create_single_pose_landmark_smoother(window_size=5):
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
