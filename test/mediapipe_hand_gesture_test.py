import cv2
from mediapipe_hand_test import run_hand_test


def is_finger_extended(hand, tip_idx, pip_idx):
    return hand[tip_idx].y < hand[pip_idx].y


def is_thumb_extended(hand, handedness_label):
    tip_x = hand[4].x
    ip_x = hand[3].x
    if handedness_label == "Right":
        return tip_x < ip_x
    if handedness_label == "Left":
        return tip_x > ip_x
    return abs(tip_x - ip_x) > 0.03


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


def draw_gesture_labels(frame, hands, handedness_labels):
    h, w = frame.shape[:2]
    for i, hand in enumerate(hands):
        label = handedness_labels[i] if i < len(handedness_labels) else "Unknown"
        gesture = classify_gesture(hand, label)
        x = int(hand[0].x * w)
        y = int(hand[0].y * h) - 12
        y = max(20, y)
        cv2.putText(
            frame,
            f"{label}: {gesture}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main():
    run_hand_test(
        window_title="MediaPipe Hand Gesture Test",
        overlay_callback=draw_gesture_labels,
    )


if __name__ == "__main__":
    main()
