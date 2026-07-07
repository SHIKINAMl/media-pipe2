import cv2
import math
from collections import Counter, deque
from types import SimpleNamespace
from mediapipe_hand_test import run_hand_test


SMOOTHING_WINDOW = 3
EXTENSION_THRESHOLD = 1.02


def joint_angle(a, b, c):
    ab_x = a.x - b.x
    ab_y = a.y - b.y
    cb_x = c.x - b.x
    cb_y = c.y - b.y

    ab_len = math.hypot(ab_x, ab_y)
    cb_len = math.hypot(cb_x, cb_y)
    if ab_len == 0 or cb_len == 0:
        return 0.0

    dot = ab_x * cb_x + ab_y * cb_y
    cosine = max(-1.0, min(1.0, dot / (ab_len * cb_len)))
    return math.degrees(math.acos(cosine))


def is_finger_extended(hand, tip_idx, pip_idx):
    wrist = hand[0]
    tip = hand[tip_idx]
    pip = hand[pip_idx]
    tip_dist2 = (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2
    pip_dist2 = (pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2
    return tip_dist2 > pip_dist2 * EXTENSION_THRESHOLD


def is_thumb_extended(hand, handedness_label):
    wrist = hand[0]
    cmc = hand[1]
    mcp = hand[2]
    tip = hand[4]
    ip = hand[3]
    index_mcp = hand[5]
    pinky_mcp = hand[17]

    angle_mcp = joint_angle(cmc, mcp, ip)
    angle_ip = joint_angle(mcp, ip, tip)

    palm_center_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3
    palm_center_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3

    tip_to_palm2 = (tip.x - palm_center_x) ** 2 + (tip.y - palm_center_y) ** 2
    ip_to_palm2 = (ip.x - palm_center_x) ** 2 + (ip.y - palm_center_y) ** 2

    return (
        angle_mcp > 145
        and angle_ip > 150
        and tip_to_palm2 > ip_to_palm2 * EXTENSION_THRESHOLD
    )


def classify_open_close(hand, handedness_label):
    index_up = is_finger_extended(hand, 8, 6)
    middle_up = is_finger_extended(hand, 12, 10)
    ring_up = is_finger_extended(hand, 16, 14)
    pinky_up = is_finger_extended(hand, 20, 18)
    thumb_up = is_thumb_extended(hand, handedness_label)

    up_count = sum([thumb_up, index_up, middle_up, ring_up, pinky_up])
    if up_count == 5:
        return "OPEN"
    if up_count == 0:
        return "CLOSE"
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


def create_open_close_overlay(window_size=SMOOTHING_WINDOW):
    label_histories = {}

    def draw_open_close_labels(frame, hands, handedness_labels):
        h, w = frame.shape[:2]
        active_indices = set()

        for i, hand in enumerate(hands):
            active_indices.add(i)
            label = handedness_labels[i] if i < len(handedness_labels) else "Unknown"
            raw_state = classify_open_close(hand, label)

            history = label_histories.setdefault(i, deque(maxlen=window_size))
            history.append(raw_state)

            open_close_votes = [s for s in history if s in ("OPEN", "CLOSE")]
            if open_close_votes:
                smoothed_state = Counter(open_close_votes).most_common(1)[0][0]
            else:
                smoothed_state = "OTHER"

            x = int(hand[0].x * w)
            y = max(20, int(hand[0].y * h) - 12)
            cv2.putText(
                frame,
                f"{label}: {smoothed_state}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for i in list(label_histories.keys()):
            if i not in active_indices:
                del label_histories[i]

    return draw_open_close_labels


def main():
    run_hand_test(
        window_title="MediaPipe Hand Open/Close Test",
        overlay_callback=create_open_close_overlay(),
        hand_filter=create_hand_filter(),
    )


if __name__ == "__main__":
    main()
