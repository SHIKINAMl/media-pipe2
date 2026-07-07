import cv2
from collections import Counter, deque
from types import SimpleNamespace
from mediapipe_hand_test import run_hand_test


SMOOTHING_WINDOW = 5


def is_finger_extended(hand, tip_idx, pip_idx):
    wrist = hand[0]
    tip = hand[tip_idx]
    pip = hand[pip_idx]
    tip_dist2 = (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2
    pip_dist2 = (pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2
    return tip_dist2 > pip_dist2 * 1.05


def is_thumb_extended(hand, handedness_label):
    wrist = hand[0]
    tip = hand[4]
    ip = hand[3]
    tip_dist2 = (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2
    ip_dist2 = (ip.x - wrist.x) ** 2 + (ip.y - wrist.y) ** 2
    return tip_dist2 > ip_dist2 * 1.05


def classify_gesture(hand, handedness_label):
    index_up = is_finger_extended(hand, 8, 6)
    middle_up = is_finger_extended(hand, 12, 10)
    ring_up = is_finger_extended(hand, 16, 14)
    pinky_up = is_finger_extended(hand, 20, 18)
    thumb_up = is_thumb_extended(hand, handedness_label)

    up_count = sum([thumb_up, index_up, middle_up, ring_up, pinky_up])

    if index_up and middle_up and not ring_up and not pinky_up:
        return "PEACE"
    if up_count >= 4:
        return "OPEN"
    if up_count <= 1:
        return "FIST"
    return "OTHER"


def create_hand_filter(window_size=SMOOTHING_WINDOW):
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


def create_gesture_overlay(window_size=SMOOTHING_WINDOW):
    label_histories = {}

    def draw_gesture_labels(frame, hands, handedness_labels):
        h, w = frame.shape[:2]
        active_indices = set()

        for i, hand in enumerate(hands):
            active_indices.add(i)
            label = handedness_labels[i] if i < len(handedness_labels) else "Unknown"
            raw_gesture = classify_gesture(hand, label)

            history = label_histories.setdefault(i, deque(maxlen=window_size))
            history.append(raw_gesture)
            smoothed_gesture = Counter(history).most_common(1)[0][0]

            x = int(hand[0].x * w)
            y = max(20, int(hand[0].y * h) - 12)
            cv2.putText(
                frame,
                f"{label}: {smoothed_gesture}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for i in list(label_histories.keys()):
            if i not in active_indices:
                del label_histories[i]

    return draw_gesture_labels


def main():
    run_hand_test(
        window_title="MediaPipe Hand Gesture Test",
        overlay_callback=create_gesture_overlay(),
        hand_filter=create_hand_filter(),
    )


if __name__ == "__main__":
    main()
