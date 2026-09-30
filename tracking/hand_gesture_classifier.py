from __future__ import annotations

import math
import time
from collections import Counter, deque


FIST = "FIST"
PAPER = "PAPER"
SCISSORS = "SCISSORS"
UNKNOWN = "UNKNOWN"

LEFT = "Left"
RIGHT = "Right"

EXTENSION_THRESHOLD = 1.05

# MediaPipe assumes a mirrored (selfie) input when it labels handedness. The
# detector feeds raw camera frames, so the label names the opposite hand.
HANDEDNESS_LABEL_SWAPPED = True


def _extension_ratio(hand, tip: int, pip: int) -> float:
    wrist = hand[0]
    tip_distance = (hand[tip].x - wrist.x) ** 2 + (hand[tip].y - wrist.y) ** 2
    pip_distance = (hand[pip].x - wrist.x) ** 2 + (hand[pip].y - wrist.y) ** 2
    return tip_distance / max(pip_distance, 1e-9)


def _is_extended(hand, tip: int, pip: int, threshold: float = EXTENSION_THRESHOLD) -> bool:
    return _extension_ratio(hand, tip, pip) > threshold


def classify_hand_gesture_with_confidence(hand) -> tuple[str, float]:
    """Return the gesture and a 0-1 confidence based on how clearly each finger is open or folded."""
    if len(hand) < 21:
        return UNKNOWN, 0.0
    ratios = [_extension_ratio(hand, tip, pip) for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))]
    index, middle, ring, pinky = (ratio > EXTENSION_THRESHOLD for ratio in ratios)
    extended = sum((index, middle, ring, pinky))
    confidence = sum(min(1.0, abs(math.log(ratio / EXTENSION_THRESHOLD)) / 0.5) for ratio in ratios) / 4
    if index and middle and not ring and not pinky:
        return SCISSORS, confidence
    if extended >= 3:
        return PAPER, confidence
    if extended == 0:
        return FIST, confidence
    return UNKNOWN, 0.0


def classify_hand_gesture(hand) -> str:
    return classify_hand_gesture_with_confidence(hand)[0]


def palm_center(hand) -> tuple[float, float]:
    return (
        (hand[0].x + hand[5].x + hand[17].x) / 3,
        (hand[0].y + hand[5].y + hand[17].y) / 3,
    )


def hand_angle(hand) -> float:
    center_x, center_y = palm_center(hand)
    return math.atan2(hand[9].y - center_y, hand[9].x - center_x)


def _label_side(label: str) -> str | None:
    if label not in (LEFT, RIGHT):
        return None
    if HANDEDNESS_LABEL_SWAPPED:
        return RIGHT if label == LEFT else LEFT
    return label


def assign_hand_sides(hands: list, handedness: list[str], poses: list) -> list[str]:
    """Resolve each detected hand to the player's own left / right hand.

    The pose wrists are the primary reference because they are anatomical; the
    handedness label is used only when no pose is available.
    """
    sides: list[str] = []
    pose = poses[0] if poses else None
    if pose is not None and len(pose) > 16 and hands:
        left_wrist, right_wrist = pose[15], pose[16]
        costs = []
        for hand in hands:
            wrist = hand[0]
            left_cost = math.hypot(wrist.x - left_wrist.x, wrist.y - left_wrist.y)
            right_cost = math.hypot(wrist.x - right_wrist.x, wrist.y - right_wrist.y)
            costs.append((left_cost, right_cost))
        if len(hands) >= 2:
            straight = costs[0][0] + costs[1][1]
            crossed = costs[0][1] + costs[1][0]
            sides = [LEFT, RIGHT] if straight <= crossed else [RIGHT, LEFT]
            for extra in costs[2:]:
                sides.append(LEFT if extra[0] <= extra[1] else RIGHT)
        else:
            sides = [LEFT if costs[0][0] <= costs[0][1] else RIGHT]
        return sides
    used: set[str] = set()
    for index, _hand in enumerate(hands):
        label = handedness[index] if index < len(handedness) else ""
        side = _label_side(label)
        if side is None or side in used:
            side = RIGHT if LEFT in used else LEFT
        used.add(side)
        sides.append(side)
    return sides


class GestureSmoother:
    """Majority vote over recent frames (default: 3 of the last 5)."""

    def __init__(self, window_size: int = 5, minimum_votes: int = 3, unknown_hold: float = 0.15) -> None:
        self._history: deque[str] = deque(maxlen=window_size)
        self._minimum_votes = minimum_votes
        self._unknown_hold = unknown_hold
        self._current = FIST
        self._last_known = time.monotonic()

    def update(self, gesture: str) -> str:
        now = time.monotonic()
        self._history.append(gesture)
        if gesture != UNKNOWN:
            self._last_known = now
        votes = Counter(value for value in self._history if value != UNKNOWN)
        if votes:
            candidate, count = votes.most_common(1)[0]
            if count >= self._minimum_votes:
                self._current = candidate
        return self._current

    @property
    def current(self) -> str:
        return self._current

    @property
    def uncertain(self) -> bool:
        """True when the hand shape could not be recognised for longer than the hold time."""
        return time.monotonic() - self._last_known > self._unknown_hold

    def reset(self) -> None:
        self._history.clear()
        self._current = FIST
        self._last_known = time.monotonic()
