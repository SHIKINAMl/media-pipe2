import math
from collections import Counter, deque

import cv2


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


def is_finger_extended(hand, tip_idx, pip_idx, extension_threshold=EXTENSION_THRESHOLD):
    wrist = hand[0]
    tip = hand[tip_idx]
    pip = hand[pip_idx]
    tip_dist2 = (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2
    pip_dist2 = (pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2
    return tip_dist2 > pip_dist2 * extension_threshold


def is_thumb_extended(hand, _handedness_label=None, extension_threshold=EXTENSION_THRESHOLD):
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
        and tip_to_palm2 > ip_to_palm2 * extension_threshold
    )


def classify_gesture(hand, handedness_label):
    index_up = is_finger_extended(hand, 8, 6, extension_threshold=1.05)
    middle_up = is_finger_extended(hand, 12, 10, extension_threshold=1.05)
    ring_up = is_finger_extended(hand, 16, 14, extension_threshold=1.05)
    pinky_up = is_finger_extended(hand, 20, 18, extension_threshold=1.05)
    thumb_up = is_thumb_extended(hand, handedness_label, extension_threshold=1.05)

    up_count = sum([thumb_up, index_up, middle_up, ring_up, pinky_up])

    if index_up and middle_up and not ring_up and not pinky_up:
        return "PEACE"
    if up_count >= 4:
        return "OPEN"
    if up_count <= 1:
        return "FIST"
    return "OTHER"


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


def create_hand_label_overlay(classifier, window_size=5, vote_labels=None):
    label_histories = {}

    def draw_labels(frame, hands, handedness_labels):
        h, w = frame.shape[:2]
        active_indices = set()

        for i, hand in enumerate(hands):
            active_indices.add(i)
            label = handedness_labels[i] if i < len(handedness_labels) else "Unknown"
            raw_state = classifier(hand, label)

            history = label_histories.setdefault(i, deque(maxlen=window_size))
            history.append(raw_state)

            if vote_labels is None:
                smoothed_state = Counter(history).most_common(1)[0][0]
            else:
                filtered = [s for s in history if s in vote_labels]
                smoothed_state = Counter(filtered).most_common(1)[0][0] if filtered else "OTHER"

            x = int(hand[0].x * w)
            y = max(20, int(hand[0].y * h) - 12)
            cv2.putText(
                frame,
                f"{label}: {smoothed_state}",
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

    return draw_labels
