"""Rhythm pattern and note placement for Game B.

Both pieces are deliberately small and replaceable: the README marks the rhythm
rule as provisional (4-4-2) and describes candidate scoring for placement (4-4).
"""
from __future__ import annotations

import math
import random

from games.common_ui import CAMERA_ASPECT


SLOTS_PER_BAR = 8  # eighth-note grid, 4/4
MIN_NOTES_PER_BAR = 2
MAX_DENSITY_STEP = 1  # notes per bar may change by at most this much between bars


def bpm_for(base_bpm: float, note_count: int, gain: float = 12.0, scale: float = 20.0) -> float:
    """BPM(n) = B0 + k * log(1 + n / scale): rises forever, gently from the first note."""
    return base_bpm + gain * math.log1p(note_count / scale)


class EighthNoteRhythm:
    """One bar = 8 eighth-note slots.

    An off-beat slot may only hold a note when the on-beat slot just before it has one.
    """

    def __init__(self, rng: random.Random, on_beat_probability: float, off_beat_probability: float) -> None:
        self._rng = rng
        self.on_beat_probability = on_beat_probability
        self.off_beat_probability = off_beat_probability
        self._previous_count = MIN_NOTES_PER_BAR

    def generate_bar(self) -> list[int]:
        slots: list[int] = []
        for beat in range(SLOTS_PER_BAR // 2):
            if self._rng.random() >= self.on_beat_probability:
                continue
            slots.append(beat * 2)
            if self._rng.random() < self.off_beat_probability:
                slots.append(beat * 2 + 1)
        # Keep the density close to the previous bar so busy and empty bars do not alternate.
        lowest = max(MIN_NOTES_PER_BAR, self._previous_count - MAX_DENSITY_STEP)
        highest = self._previous_count + MAX_DENSITY_STEP
        while len(slots) > highest:
            off_beats = [slot for slot in slots if slot % 2 == 1]
            # Drop off-beats first; an on-beat may only go if its off-beat is not used.
            removable = off_beats or [slot for slot in slots if slot + 1 not in slots]
            slots.remove(self._rng.choice(removable))
        while len(slots) < lowest:
            free = [beat * 2 for beat in range(SLOTS_PER_BAR // 2) if beat * 2 not in slots]
            if not free:
                # Every on-beat is taken: fill off-beats (their on-beat exists, so the rule holds).
                free = [slot for slot in range(1, SLOTS_PER_BAR, 2) if slot not in slots]
            if not free:
                break
            slots.append(self._rng.choice(free))
        slots.sort()
        self._previous_count = len(slots)
        return slots


class NotePlacer:
    """Pick the next note position near the previous one by scoring random candidates.

    Positions are viewport-normalised; distances are in viewport-height units.
    """

    def __init__(
        self,
        rng: random.Random,
        area: tuple[float, float, float, float],
        distance: tuple[float, float],
        wall_reference: float = 0.15,
    ) -> None:
        self._rng = rng
        self.area = area
        self.distance = distance
        self.wall_reference = wall_reference

    def next(
        self,
        previous: tuple[float, float],
        direction: tuple[float, float] | None,
        side_bias: float | None = None,
        margin: float = 0.0,
    ) -> tuple[float, float]:
        left, right, top, bottom = self._inset(margin)
        direction = self._inward(previous, direction, (left, right, top, bottom))
        minimum, maximum = self.distance
        recommended = (minimum + maximum) / 2
        scored: list[tuple[float, float, float]] = []
        for _ in range(20):
            angle = self._rng.uniform(0, math.tau)
            distance = self._rng.uniform(minimum * 0.8, maximum * 1.1)
            x = previous[0] + distance * math.cos(angle) / CAMERA_ASPECT
            y = previous[1] + distance * math.sin(angle)
            if not (left <= x <= right and top <= y <= bottom):
                continue
            distance_score = max(0.0, 1.0 - abs(distance - recommended) / recommended)
            if direction is None:
                progress_score = 0.5
            else:
                progress_score = (1.0 + math.cos(angle) * direction[0] + math.sin(angle) * direction[1]) / 2
            wall = min((x - left) * CAMERA_ASPECT, (right - x) * CAMERA_ASPECT, y - top, bottom - y)
            wall_score = min(1.0, wall / self.wall_reference)
            side_score = 0.5 if side_bias is None else 1.0 - min(1.0, abs(x - side_bias) / 0.4)
            score = distance_score + 0.8 * progress_score + 0.7 * wall_score + 0.5 * side_score + self._rng.uniform(0.0, 0.3)
            scored.append((score, x, y))
        if not scored:
            center = ((left + right) / 2, (top + bottom) / 2)
            dx, dy = (center[0] - previous[0]) * CAMERA_ASPECT, center[1] - previous[1]
            length = max(math.hypot(dx, dy), 1e-6)
            step = min(recommended, length)
            return previous[0] + dx / length * step / CAMERA_ASPECT, previous[1] + dy / length * step
        scored.sort(reverse=True)
        top_candidates = scored[:4]
        weights = [candidate[0] for candidate in top_candidates]
        _score, x, y = self._rng.choices(top_candidates, weights=weights, k=1)[0]
        return x, y

    def _inset(self, margin: float) -> tuple[float, float, float, float]:
        left, right, top, bottom = self.area
        return left + margin / CAMERA_ASPECT, right - margin / CAMERA_ASPECT, top, bottom

    def _inward(self, previous, direction, bounds):
        """Near a wall, flip the travel direction so the path turns back inside."""
        if direction is None:
            return None
        left, right, top, bottom = bounds
        dx, dy = direction
        reference = self.wall_reference
        if (previous[0] - left) * CAMERA_ASPECT < reference and dx < 0:
            dx = -dx
        if (right - previous[0]) * CAMERA_ASPECT < reference and dx > 0:
            dx = -dx
        if previous[1] - top < reference and dy < 0:
            dy = -dy
        if bottom - previous[1] < reference and dy > 0:
            dy = -dy
        return dx, dy
