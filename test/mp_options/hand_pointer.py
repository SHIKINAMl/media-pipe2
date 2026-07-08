from collections import Counter, deque

import cv2
import numpy as np

from mp_options.hand_gestures import classify_open_close


POINTER_WINDOW_TITLE = "MediaPipe Hand Pointer"
POINTER_CANVAS_WIDTH = 1280
POINTER_CANVAS_HEIGHT = 720


def _hand_center(hand):
    wrist = hand[0]
    index_mcp = hand[5]
    pinky_mcp = hand[17]
    center_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3
    center_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3
    return center_x, center_y


def create_hand_pointer_overlay(window_size=5, window_title=POINTER_WINDOW_TITLE):
    state_histories = {0: deque(maxlen=window_size)}
    point_histories = {0: deque(maxlen=window_size)}
    window_ready = False

    # Open the pointer window immediately at startup.
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, POINTER_CANVAS_WIDTH, POINTER_CANVAS_HEIGHT)
    cv2.imshow(
        window_title,
        np.zeros((POINTER_CANVAS_HEIGHT, POINTER_CANVAS_WIDTH, 3), dtype=np.uint8),
    )
    window_ready = True

    def draw_pointer(frame, hands, handedness_labels):
        nonlocal window_ready

        if not window_ready:
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_title, POINTER_CANVAS_WIDTH, POINTER_CANVAS_HEIGHT)
            window_ready = True

        canvas = np.zeros((POINTER_CANVAS_HEIGHT, POINTER_CANVAS_WIDTH, 3), dtype=np.uint8)

        if not hands:
            cv2.putText(canvas, "NO HAND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
            state_histories[0].clear()
            point_histories[0].clear()
            cv2.imshow(window_title, canvas)
            return

        hand = hands[0]
        label = handedness_labels[0] if handedness_labels else "Unknown"
        raw_state = classify_open_close(hand, label)

        history = state_histories.setdefault(0, deque(maxlen=window_size))
        history.append(raw_state)
        votes = [state for state in history if state in ("OPEN", "CLOSE")]
        smoothed_state = Counter(votes).most_common(1)[0][0] if votes else "OTHER"

        center_x, center_y = _hand_center(hand)
        # Invert both horizontal and vertical directions.
        raw_x = max(0, min(POINTER_CANVAS_WIDTH - 1, int((1.0 - center_x) * POINTER_CANVAS_WIDTH)))
        raw_y = max(0, min(POINTER_CANVAS_HEIGHT - 1, int((1.0 - center_y) * POINTER_CANVAS_HEIGHT)))

        point_history = point_histories.setdefault(0, deque(maxlen=window_size))
        point_history.append((raw_x, raw_y))
        x = int(sum(px for px, _ in point_history) / len(point_history))
        y = int(sum(py for _, py in point_history) / len(point_history))

        if smoothed_state == "CLOSE":
            pointer_color = (0, 0, 255)
            state_text = "LEFT DOWN"
        else:
            pointer_color = (0, 220, 0)
            state_text = "NO CLICK"

        cv2.circle(canvas, (x, y), 26, pointer_color, -1)
        cv2.circle(canvas, (x, y), 32, (255, 255, 255), 2)
        cv2.putText(
            canvas,
            f"{state_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            pointer_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_title, canvas)

    return draw_pointer
