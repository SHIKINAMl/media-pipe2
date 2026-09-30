from __future__ import annotations

import math
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace

from tracking.hand_gesture_classifier import (
    FIST,
    LEFT,
    PAPER,
    RIGHT,
    SCISSORS,
    GestureSmoother,
    classify_hand_gesture,
    hand_angle,
    palm_center,
)


CAMERA_ASPECT = 16 / 9

# ---------------------------------------------------------------------------
# Palette (ideas/README.md 9-8, 9-12)
# ---------------------------------------------------------------------------
BACKGROUND = "#050505"
PART_BASE = "#CBD5E1"
PART_BACK = "#8B98AB"
HAND_COLOR = "#FBBF24"
OPERATING_HAND = "#FFFFFF"
TEXT = "#F8FAFC"
TEXT_SUB = "#94A3B8"
WARNING = "#F59E0B"
SUCCESS = "#34D399"
FAILURE = "#FB7185"
ACCENT = "#38BDF8"

GESTURE_COLORS = {FIST: "#FB7185", PAPER: "#FBBF24", SCISSORS: "#22D3EE"}
GESTURE_SYMBOLS = {FIST: "fist", PAPER: "paper", SCISSORS: "scissors"}

FONT_JA = "Yu Gothic UI"
FONT_MONO = "Consolas"

# Light comes from the upper left; every shaded part uses the same direction.
LIGHT = (-0.55, -0.83)


# ---------------------------------------------------------------------------
# Geometry helpers. Targets, pointer and avatar all share the same viewport.
# ---------------------------------------------------------------------------
def canvas_viewport(canvas: tk.Canvas) -> tuple[float, float, float, float]:
    width = max(canvas.winfo_width(), 1)
    height = max(canvas.winfo_height(), 1)
    if width / height > CAMERA_ASPECT:
        view_height = height
        view_width = height * CAMERA_ASPECT
    else:
        view_width = width
        view_height = width / CAMERA_ASPECT
    return ((width - view_width) / 2, (height - view_height) / 2, view_width, view_height)


def normalized_point(x: float, y: float, viewport: tuple[float, float, float, float]) -> tuple[float, float]:
    left, top, width, height = viewport
    return left + x * width, top + y * height


def mirrored(landmark) -> tuple[float, float]:
    """Camera landmark -> mirrored viewport-normalised coordinates."""
    return 1.0 - landmark.x, landmark.y


def view_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Isotropic distance between two viewport-normalised points, in viewport-height units."""
    return math.hypot((first[0] - second[0]) * CAMERA_ASPECT, first[1] - second[1])


def ui_scale(canvas: tk.Canvas) -> float:
    width = max(canvas.winfo_width(), 1)
    height = max(canvas.winfo_height(), 1)
    return min(width / 1920, height / 1080)


def ui_font(canvas: tk.Canvas, pixels_at_1080p: float, bold: bool = True, mono: bool = False) -> tuple:
    size = -max(9, int(pixels_at_1080p * ui_scale(canvas)))
    return (FONT_MONO if mono else FONT_JA, size, "bold" if bold else "normal")


def _mix(first: str, second: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    first, second = first.lstrip("#"), second.lstrip("#")
    channels = []
    for index in (0, 2, 4):
        a = int(first[index:index + 2], 16)
        b = int(second[index:index + 2], 16)
        channels.append(round(a + (b - a) * ratio))
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def dim(color: str, factor: float) -> str:
    """Blend ``color`` toward the background. Tk has no alpha, so brightness stands in for opacity."""
    return _mix(BACKGROUND, color, factor)


def visible(landmark, threshold: float = 0.5) -> bool:
    return getattr(landmark, "visibility", 1.0) >= threshold


def draw_marker(canvas: tk.Canvas, x: float, y: float, radius: float, symbol: str, color: str, tags, fill: bool = True, width: int = 3) -> None:
    """Shape symbols that pair colours with body parts / hand shapes, readable without text."""
    fill_color = color if fill else ""
    if symbol == "circle":
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill_color, outline=color, width=width, tags=tags)
    elif symbol == "diamond":
        canvas.create_polygon(x, y - radius * 1.2, x + radius, y, x, y + radius * 1.2, x - radius, y, fill=fill_color, outline=color, width=width, tags=tags)
    elif symbol == "triangle":
        canvas.create_polygon(x, y - radius * 1.15, x + radius * 1.1, y + radius * 0.8, x - radius * 1.1, y + radius * 0.8, fill=fill_color, outline=color, width=width, tags=tags)
    elif symbol == "square":
        side = radius * 0.9
        canvas.create_rectangle(x - side, y - side, x + side, y + side, fill=fill_color, outline=color, width=width, tags=tags)
    elif symbol == "foot":
        canvas.create_oval(x - radius * 0.6, y - radius * 1.1, x + radius * 0.6, y + radius * 1.1, fill=fill_color, outline=color, width=width, tags=tags)
    elif symbol in ("fist", "paper", "scissors"):
        palm = radius * (0.7 if symbol == "paper" else 0.8)
        canvas.create_oval(x - palm, y - palm + radius * 0.25, x + palm, y + palm + radius * 0.25, fill=color, outline="", tags=tags)
        if symbol == "paper":
            angles = (-150, -118, -90, -62, -32)
        elif symbol == "scissors":
            angles = (-112, -68)
        else:
            angles = ()
        finger = max(3, int(radius * 0.32))
        for angle in angles:
            rad = math.radians(angle)
            start = (x + math.cos(rad) * palm * 1.25, y + radius * 0.25 + math.sin(rad) * palm * 1.25)
            end = (x + math.cos(rad) * radius * 1.45, y + radius * 0.25 + math.sin(rad) * radius * 1.45)
            canvas.create_line(*start, *end, fill=color, width=finger, capstyle=tk.ROUND, tags=tags)
    else:
        canvas.create_text(x, y, text=symbol, fill=color, font=(FONT_JA, -int(radius * 1.6), "bold"), tags=tags)


def _snapshot_landmarks(landmarks) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            x=float(landmark.x),
            y=float(landmark.y),
            z=float(getattr(landmark, "z", 0.0)),
            visibility=float(getattr(landmark, "visibility", 1.0) or 0.0),
        )
        for landmark in landmarks
    ]


# ---------------------------------------------------------------------------
# Display-only smoothing (never used for hit tests)
# ---------------------------------------------------------------------------
class DisplayLandmarkSmoother:
    """Moving-average smoothing used only by the avatar renderer.

    Dropouts keep the last position for ``HOLD`` seconds and then fade the body
    out smoothly, so a flaky detection never makes the avatar blink.
    """

    HOLD = 0.3
    FADE = 0.8

    def __init__(self, window_size: int = 5) -> None:
        self._window_size = window_size
        self._pose_history: deque[list[SimpleNamespace]] = deque(maxlen=window_size)
        self._pose_last: list[SimpleNamespace] | None = None
        self._pose_seen = 0.0
        self._hand_histories: dict[str, deque[list[SimpleNamespace]]] = {}
        self._hand_last: dict[str, tuple[list[SimpleNamespace], float]] = {}

    def smooth_pose(self, poses: list, now: float) -> tuple[list[SimpleNamespace] | None, float]:
        if poses:
            self._pose_history.append(_snapshot_landmarks(poses[0]))
            self._pose_last = self._average(self._pose_history)
            self._pose_seen = now
            return self._pose_last, 1.0
        age = now - self._pose_seen
        if self._pose_last is None or age > self.FADE:
            self._pose_history.clear()
            self._pose_last = None
            return None, 0.0
        if age <= self.HOLD:
            return self._pose_last, 1.0
        return self._pose_last, 1.0 - (age - self.HOLD) / (self.FADE - self.HOLD)

    def smooth_hands(self, hands: list, sides: list[str], now: float) -> dict[str, list[SimpleNamespace]]:
        result: dict[str, list[SimpleNamespace]] = {}
        for index, hand in enumerate(hands):
            side = sides[index] if index < len(sides) else f"hand{index}"
            if side in result:
                continue
            history = self._hand_histories.setdefault(side, deque(maxlen=self._window_size))
            history.append(_snapshot_landmarks(hand))
            smoothed = self._average(history)
            self._hand_last[side] = (smoothed, now)
            result[side] = smoothed
        for side in list(self._hand_last):
            if side in result:
                continue
            landmarks, seen = self._hand_last[side]
            if now - seen <= self.HOLD:
                result[side] = landmarks
            else:
                # Past the hold time the renderer falls back to the pose wrist (no fade = no blink).
                self._hand_last.pop(side, None)
                self._hand_histories.pop(side, None)
        return result

    @staticmethod
    def _average(history: deque[list[SimpleNamespace]]) -> list[SimpleNamespace]:
        count = len(history)
        return [
            SimpleNamespace(
                x=sum(frame[index].x for frame in history) / count,
                y=sum(frame[index].y for frame in history) / count,
                z=sum(frame[index].z for frame in history) / count,
                visibility=sum(frame[index].visibility for frame in history) / count,
            )
            for index in range(len(history[-1]))
        ]


class _VisibilityGate:
    """Per-landmark hysteresis: show above 0.55, hide only after staying below 0.35 for a while."""

    SHOW = 0.55
    HIDE = 0.35
    HIDE_AFTER = 0.35

    def __init__(self) -> None:
        self._shown: dict[int, bool] = {}
        self._low_since: dict[int, float] = {}

    def update(self, pose, now: float) -> None:
        for index, landmark in enumerate(pose):
            value = getattr(landmark, "visibility", 1.0)
            if value >= self.SHOW:
                self._shown[index] = True
                self._low_since.pop(index, None)
            elif value < self.HIDE:
                since = self._low_since.setdefault(index, now)
                if now - since > self.HIDE_AFTER:
                    self._shown[index] = False
            else:
                self._low_since.pop(index, None)

    def reset(self) -> None:
        self._shown.clear()
        self._low_since.clear()

    def __call__(self, index: int) -> bool:
        return self._shown.get(index, False)


# ---------------------------------------------------------------------------
# Avatar
# ---------------------------------------------------------------------------
@dataclass
class AvatarStyle:
    """Per-screen emphasis. Brightness 1.0 = full colour, 0.0 = background."""

    brightness: float = 1.0
    part_brightness: dict[str, float] = field(default_factory=dict)
    part_colors: dict[str, str] = field(default_factory=dict)
    part_scale: dict[str, float] = field(default_factory=dict)
    part_marks: dict[str, tuple[str, str]] = field(default_factory=dict)
    hand_rings: dict[str, str] = field(default_factory=dict)
    show_gestures: bool = False
    operating_hand: str | None = None
    frozen: bool = False


SIDE_PARTS = {
    LEFT: {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "knee": 25, "ankle": 27, "heel": 29, "toe": 31},
    RIGHT: {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24, "knee": 26, "ankle": 28, "heel": 30, "toe": 32},
}


def side_prefix(side: str) -> str:
    return "left" if side == LEFT else "right"


def _tapered_outline(start, end, start_radius: float, end_radius: float, steps: int = 10) -> list[float]:
    """Polygon of a capsule whose radius changes from start to end (a rounded cone frustum)."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    axis = math.atan2(dy, dx)
    points: list[float] = []
    for step in range(steps + 1):
        angle = axis + math.pi / 2 + math.pi * step / steps
        points += [start[0] + math.cos(angle) * start_radius, start[1] + math.sin(angle) * start_radius]
    for step in range(steps + 1):
        angle = axis - math.pi / 2 + math.pi * step / steps
        points += [end[0] + math.cos(angle) * end_radius, end[1] + math.sin(angle) * end_radius]
    return points


class AvatarRenderer:
    """Separated, softly shaded primitives: spheres for head/hands, tapered capsules for limbs.

    No joints, bones or landmark points are drawn. Each part is a dark base with a
    lit body shifted toward the light and a small specular highlight, which reads as
    a solid object rather than a flat stick.
    """

    def __init__(self, canvas: tk.Canvas) -> None:
        self.canvas = canvas
        self._gesture_smoothers: dict[str, GestureSmoother] = {}
        self._landmark_smoother = DisplayLandmarkSmoother(window_size=5)
        self._gate = _VisibilityGate()
        self._last_pose: list[SimpleNamespace] | None = None
        self._last_pose_fade = 1.0
        self._last_hands: dict[str, list[SimpleNamespace]] = {}
        self.hand_screen_positions: dict[str, tuple[float, float]] = {}
        self.hand_screen_radius: dict[str, float] = {}

    def draw(self, snapshot, style: AvatarStyle | None = None) -> None:
        style = style or AvatarStyle()
        self.canvas.delete("avatar")
        self.hand_screen_positions = {}
        self.hand_screen_radius = {}
        if not style.frozen:
            now = time.monotonic()
            self._last_pose, self._last_pose_fade = self._landmark_smoother.smooth_pose(snapshot.poses, now)
            self._last_hands = self._landmark_smoother.smooth_hands(snapshot.hands, snapshot.hand_sides, now)
            if self._last_pose is None:
                self._gate.reset()
            else:
                self._gate.update(self._last_pose, now)
        pose = self._last_pose
        if pose is None or len(pose) < 33:
            return
        viewport = canvas_viewport(self.canvas)
        self._style = style
        self._fade = self._last_pose_fade
        self._viewport = viewport
        self._scale = max(0.75, min(1.5, viewport[3] / 1080))
        shoulder_width = max(self._distance(pose[11], pose[12]), 1.0)
        hip_width = max(self._distance(pose[23], pose[24]), 1.0)
        gap = max(4.0, viewport[3] * 0.008)

        sizes = {
            # (start, end) diameters: muscles swell near the body and taper toward the joints.
            "upper_arm": (self._clamp(shoulder_width * 0.22, 28, 54), self._clamp(shoulder_width * 0.15, 20, 38)),
            "forearm": (self._clamp(shoulder_width * 0.17, 22, 42), self._clamp(shoulder_width * 0.11, 15, 30)),
            "thigh": (self._clamp(shoulder_width * 0.30, 38, 72), self._clamp(shoulder_width * 0.19, 26, 48)),
            "shin": (self._clamp(shoulder_width * 0.21, 28, 52), self._clamp(shoulder_width * 0.12, 17, 32)),
            "hand": self._clamp(shoulder_width * 0.30, 38, 72),
            "foot": self._clamp(shoulder_width * 0.28, 34, 64),
            "head": self._clamp(shoulder_width * 0.58, 76, 156),
        }

        # Depth only decides draw order: the side with the larger z is farther away.
        left_depth = pose[11].z + pose[23].z
        right_depth = pose[12].z + pose[24].z
        back, front = (LEFT, RIGHT) if left_depth > right_depth else (RIGHT, LEFT)

        self._torso_top = None
        self._draw_side(pose, back, sizes, gap, back_side=True)
        self._draw_torso(pose, shoulder_width, hip_width, gap)
        self.canvas.create_line(0, 0, 0, 0, fill="", tags=("avatar", "avatar_mid"))
        self._draw_side(pose, front, sizes, gap, back_side=False)
        self._draw_head(pose, sizes["head"], gap)

    # -- shading primitives ---------------------------------------------------
    def _tones(self, color: str, brightness: float) -> tuple[str, str, str]:
        return (
            dim(_mix(color, "#000000", 0.55), brightness),
            dim(color, brightness),
            dim(_mix(color, "#FFFFFF", 0.4), brightness),
        )

    def _shaded_capsule(self, start, end, start_radius: float, end_radius: float, color: str, brightness: float) -> None:
        shadow, body, highlight = self._tones(color, brightness)
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max(math.hypot(dx, dy), 1e-6)
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux
        if nx * LIGHT[0] + ny * LIGHT[1] < 0:
            nx, ny = -nx, -ny
        self.canvas.create_polygon(_tapered_outline(start, end, start_radius, end_radius), fill=shadow, outline="", tags="avatar")
        shift_a, shift_b = start_radius * 0.16, end_radius * 0.16
        self.canvas.create_polygon(
            _tapered_outline((start[0] + nx * shift_a, start[1] + ny * shift_a), (end[0] + nx * shift_b, end[1] + ny * shift_b), start_radius * 0.8, end_radius * 0.8),
            fill=body, outline="", tags="avatar",
        )
        # Soft sheen on the lit side near the thick end (a short glow, not a line along the whole limb).
        sheen_length = min(length * 0.45, start_radius * 2.2)
        sheen_a = (start[0] + nx * start_radius * 0.38, start[1] + ny * start_radius * 0.38)
        sheen_b = (sheen_a[0] + ux * sheen_length, sheen_a[1] + uy * sheen_length)
        if sheen_length > start_radius * 0.3:
            self.canvas.create_polygon(_tapered_outline(sheen_a, sheen_b, start_radius * 0.26, start_radius * 0.12, steps=6), fill=highlight, outline="", tags="avatar")

    def _shaded_sphere(self, center, radius: float, color: str, brightness: float) -> None:
        shadow, body, highlight = self._tones(color, brightness)
        x, y = center
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=shadow, outline="", tags="avatar")
        shift = radius * 0.12
        inner = radius * 0.84
        bx, by = x + LIGHT[0] * shift, y + LIGHT[1] * shift
        self.canvas.create_oval(bx - inner, by - inner, bx + inner, by + inner, fill=body, outline="", tags="avatar")
        spot = radius * 0.24
        sx, sy = x + LIGHT[0] * radius * 0.42, y + LIGHT[1] * radius * 0.42
        self.canvas.create_oval(sx - spot * 1.2, sy - spot, sx + spot * 1.2, sy + spot, fill=highlight, outline="", tags="avatar")

    # -- parts ----------------------------------------------------------------
    def _part_brightness(self, part: str) -> float:
        return self._style.part_brightness.get(part, self._style.brightness) * self._fade

    def _part_scale(self, part: str) -> float:
        return self._style.part_scale.get(part, 1.0)

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum * self._scale, min(maximum * self._scale, value))

    def _point(self, landmark) -> tuple[float, float]:
        left, top, width, height = self._viewport
        return left + (1.0 - landmark.x) * width, top + landmark.y * height

    def _distance(self, first, second) -> float:
        _left, _top, width, height = self._viewport
        return math.hypot((first.x - second.x) * width, (first.y - second.y) * height)

    def _draw_side(self, pose, side: str, sizes: dict, gap: float, back_side: bool) -> None:
        indices = SIDE_PARTS[side]
        prefix = side_prefix(side)
        base = PART_BACK if back_side else PART_BASE

        hand_center, hand_radius = self._hand_geometry(pose, side, sizes["hand"])
        foot_geometry = self._foot_geometry(pose, side, sizes["foot"])
        foot_center = foot_geometry[0] if foot_geometry else None
        foot_keep_out = foot_geometry[2] + gap * 0.8 if foot_geometry else 0.0

        self._limb(pose, indices["shoulder"], indices["elbow"], sizes["upper_arm"], gap, base, f"{prefix}_upper_arm", start_extra=0.0)
        self._limb(pose, indices["elbow"], indices["wrist"], sizes["forearm"], gap, base, f"{prefix}_forearm",
                   end_override=(hand_center, hand_radius + gap * 0.8) if hand_center else None)
        self._limb(pose, indices["hip"], indices["knee"], sizes["thigh"], gap, base, f"{prefix}_thigh", start_extra=0.0)
        self._limb(pose, indices["knee"], indices["ankle"], sizes["shin"], gap, base, f"{prefix}_shin",
                   end_override=(foot_center, foot_keep_out) if foot_center else None)
        if foot_geometry is not None:
            self._draw_foot(side, foot_geometry)
        if hand_center is not None:
            self._draw_hand(side, hand_center, hand_radius)

    def _limb(self, pose, start_index: int, end_index: int, diameters: tuple[float, float], gap: float, base: str, part: str,
              end_override=None, start_extra: float = 0.0) -> None:
        if not (self._gate(start_index) and self._gate(end_index)):
            return
        brightness = self._part_brightness(part)
        if brightness <= 0.02:
            return
        color = self._style.part_colors.get(part, base)
        start = self._point(pose[start_index])
        end = self._point(pose[end_index])
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        scale = self._part_scale(part)
        start_radius, end_radius = diameters[0] / 2 * scale, diameters[1] / 2 * scale
        # Pull both ends inward so the rounded caps never touch the neighbouring parts.
        start_trim = gap / 2 + start_radius + start_extra
        start = (start[0] + ux * start_trim, start[1] + uy * start_trim)
        if end_override is not None:
            center, keep_out = end_override
            end_distance = (center[0] - start[0]) * ux + (center[1] - start[1]) * uy - keep_out - end_radius
        else:
            end_distance = length - start_trim - gap / 2 - end_radius
        if end_distance <= 2:
            return
        end = (start[0] + ux * end_distance, start[1] + uy * end_distance)
        self._shaded_capsule(start, end, start_radius, end_radius, color, brightness)

    def _draw_torso(self, pose, shoulder_width: float, hip_width: float, gap: float) -> None:
        if not all(self._gate(index) for index in (11, 12, 23, 24)):
            return
        brightness = self._part_brightness("torso")
        color = self._style.part_colors.get("torso", PART_BASE)
        shadow, body, highlight = self._tones(color, brightness)
        left_shoulder, right_shoulder = self._point(pose[11]), self._point(pose[12])
        left_hip, right_hip = self._point(pose[23]), self._point(pose[24])
        top = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
        bottom = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
        dx, dy = bottom[0] - top[0], bottom[1] - top[1]
        length = max(math.hypot(dx, dy), 1.0)
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux
        scale = self._part_scale("torso")
        chest = shoulder_width * 0.40 * scale
        waist = (shoulder_width * 0.30 + hip_width * 0.30) / 2 * scale
        pelvis = hip_width * 0.44 * scale
        inset_top = gap * 0.6
        inset_bottom = gap * 0.6
        self._torso_top = (top[0] + ux * inset_top, top[1] + uy * inset_top)

        def outline(shrink: float, shift: float) -> list[float]:
            # Rounded chest, narrower waist, pelvis: a smooth spline through (axis position, half width).
            profile = ((0.0, chest * 0.78), (0.10, chest), (0.42, chest * 0.9), (0.68, waist), (0.92, pelvis), (1.0, pelvis * 0.8))
            usable = length - inset_top - inset_bottom
            sx, sy = nx * shift, ny * shift
            right_side, left_side = [], []
            for position, half in profile:
                ax = top[0] + ux * (inset_top + usable * position) + sx
                ay = top[1] + uy * (inset_top + usable * position) + sy
                half *= shrink
                right_side += [ax + nx * half, ay + ny * half]
                left_side = [ax - nx * half, ay - ny * half] + left_side
            return right_side + left_side

        light_sign = 1.0 if nx * LIGHT[0] + ny * LIGHT[1] >= 0 else -1.0
        self.canvas.create_polygon(outline(1.0, 0.0), fill=shadow, outline="", smooth=True, tags="avatar")
        self.canvas.create_polygon(outline(0.84, light_sign * chest * 0.12), fill=body, outline="", smooth=True, tags="avatar")
        # Soft vertical highlight on the lit side of the chest.
        strip_top = (top[0] + ux * length * 0.16 + nx * light_sign * chest * 0.45, top[1] + uy * length * 0.16 + ny * light_sign * chest * 0.45)
        strip_bottom = (top[0] + ux * length * 0.5 + nx * light_sign * chest * 0.38, top[1] + uy * length * 0.5 + ny * light_sign * chest * 0.38)
        self.canvas.create_polygon(_tapered_outline(strip_top, strip_bottom, chest * 0.09, chest * 0.05, steps=6), fill=highlight, outline="", tags="avatar")

    def _draw_head(self, pose, diameter: float, gap: float) -> None:
        if not (self._gate(0) and (self._gate(7) or self._gate(8))):
            return
        brightness = self._part_brightness("head")
        nose = self._point(pose[0])
        ears = [self._point(pose[index]) for index in (7, 8)]
        center = ((nose[0] + (ears[0][0] + ears[1][0]) / 2) / 2, (nose[1] + (ears[0][1] + ears[1][1]) / 2) / 2)
        radius = diameter / 2 * self._part_scale("head")
        torso_top = self._torso_top
        if torso_top is not None:
            required = radius + gap * 1.3
            distance = math.dist(center, torso_top)
            if distance < required:
                if distance > 1:
                    direction = ((center[0] - torso_top[0]) / distance, (center[1] - torso_top[1]) / distance)
                else:
                    direction = (0.0, -1.0)
                center = (torso_top[0] + direction[0] * required, torso_top[1] + direction[1] * required)
        self._shaded_sphere(center, radius, self._style.part_colors.get("head", "#E2E8F0"), brightness)

    def _foot_geometry(self, pose, side: str, diameter: float):
        indices = SIDE_PARTS[side]
        if not self._gate(indices["ankle"]):
            return None
        heel = self._point(pose[indices["heel"]])
        toe = self._point(pose[indices["toe"]])
        ankle = self._point(pose[indices["ankle"]])
        center = ((heel[0] + toe[0] + ankle[0]) / 3, (heel[1] + toe[1] + ankle[1]) / 3)
        radius = diameter / 2 * self._part_scale(f"{side_prefix(side)}_foot")
        return center, (heel, toe), radius

    def _draw_foot(self, side: str, geometry) -> None:
        part = f"{side_prefix(side)}_foot"
        center, (heel, toe), radius = geometry
        brightness = self._part_brightness(part)
        color = self._style.part_colors.get(part, "#E2E8F0")
        dx, dy = toe[0] - heel[0], toe[1] - heel[1]
        length = math.hypot(dx, dy)
        if length < radius * 0.4:
            self._shaded_sphere(center, radius, color, brightness)
        else:
            # A short rounded wedge pointing at the toes: heel end rounder, toe end slimmer.
            ux, uy = dx / length, dy / length
            half = max(length * 0.55, radius * 0.7)
            start = (center[0] - ux * half * 0.5, center[1] - uy * half * 0.5)
            end = (center[0] + ux * half * 0.7, center[1] + uy * half * 0.7)
            self._shaded_capsule(start, end, radius * 0.72, radius * 0.5, color, brightness)
        self._draw_mark(part, center, radius)

    def _hand_geometry(self, pose, side: str, diameter: float):
        part = f"{side_prefix(side)}_hand"
        radius = diameter / 2 * self._part_scale(part)
        hand = self._last_hands.get(side)
        if hand is not None:
            center_x, center_y = palm_center(hand)
            left, top, width, height = self._viewport
            return (left + (1.0 - center_x) * width, top + center_y * height), radius
        if not self._gate(SIDE_PARTS[side]["wrist"]):
            return None, radius
        return self._point(pose[SIDE_PARTS[side]["wrist"]]), radius

    def _draw_hand(self, side: str, center, radius) -> None:
        part = f"{side_prefix(side)}_hand"
        self.hand_screen_positions[side] = center
        self.hand_screen_radius[side] = radius
        brightness = self._part_brightness(part)
        if self._style.operating_hand == side:
            color, brightness = OPERATING_HAND, max(brightness, 1.0 * self._fade)
        else:
            color = self._style.part_colors.get(part, HAND_COLOR)

        gesture = FIST
        angle = -math.pi / 2
        hand = self._last_hands.get(side)
        if hand is not None:
            angle = math.pi - hand_angle(hand)  # mirrored display
            if self._style.show_gestures:
                smoother = self._gesture_smoothers.setdefault(side, GestureSmoother(unknown_hold=0.4))
                if not self._style.frozen:
                    smoother.update(classify_hand_gesture(hand))
                gesture = smoother.current
        if self._style.show_gestures:
            color = GESTURE_COLORS.get(gesture, color)
        self._draw_hand_shape(center, radius, gesture, angle, color, brightness)
        ring_color = self._style.hand_rings.get(side)
        if ring_color:
            ring = radius * 1.45
            self.canvas.create_oval(center[0] - ring, center[1] - ring, center[0] + ring, center[1] + ring, outline=ring_color, width=5, tags="avatar")
        self._draw_mark(part, center, radius)

    def _draw_mark(self, part: str, center, radius) -> None:
        mark = self._style.part_marks.get(part)
        if mark is None:
            return
        symbol, color = mark
        ring = radius * 1.35
        self.canvas.create_oval(center[0] - ring, center[1] - ring, center[0] + ring, center[1] + ring, outline=color, width=5, tags="avatar")
        # Feet show the symbol below so it never overlaps the shin.
        offset = ring + radius * 0.45
        symbol_y = center[1] + offset if part.endswith("_foot") else center[1] - offset
        draw_marker(self.canvas, center[0], symbol_y, radius * 0.38, symbol, color, "avatar")

    def _draw_hand_shape(self, center, radius, gesture, angle, color, brightness) -> None:
        palm_radius = radius * (0.9 if gesture == PAPER else 1.0)
        finger_gap = max(3, min(6, radius * 0.12))
        finger_radius = max(4.0, radius * 0.25)
        if gesture == PAPER:
            # thumb, index, middle, ring, pinky: middle longest, thumb shortest
            fingers = ((-1.05, 0.62), (-0.52, 0.95), (0.0, 1.05), (0.46, 0.98), (0.88, 0.80))
        elif gesture == SCISSORS:
            spread = math.radians(45) / 2
            fingers = ((-spread, 1.05), (spread, 1.1))
        else:
            fingers = ()
        for offset, length_ratio in fingers:
            finger_angle = angle + offset
            start_distance = palm_radius + finger_gap + finger_radius
            start = (center[0] + math.cos(finger_angle) * start_distance, center[1] + math.sin(finger_angle) * start_distance)
            length = radius * length_ratio - finger_radius
            end = (start[0] + math.cos(finger_angle) * length, start[1] + math.sin(finger_angle) * length)
            self._shaded_capsule(start, end, finger_radius, finger_radius * 0.85, color, brightness)
        self._shaded_sphere(center, palm_radius, color, brightness)


# ---------------------------------------------------------------------------
# Hand cursors for menus
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CursorState:
    side: str
    x: float
    y: float
    radius: float
    pressed: bool
    grabbed: bool  # fist closed this frame (after the hand was opened since the last grab)


class HandCursors:
    """Both hands act as cursors, and each cursor is exactly where the avatar hand is drawn.

    Positions come from ``AvatarRenderer.hand_screen_positions`` so there is no separate
    mapping (and therefore no offset). When only the pose wrist is available the cursor
    still works for hover selection; the fist shortcut needs the hand landmarks.
    """

    def __init__(self, grab_cooldown: float = 0.7) -> None:
        self._gestures: dict[str, GestureSmoother] = {}
        self._pressed: dict[str, bool] = {}
        self._armed: dict[str, bool] = {}
        self._last_grab: dict[str, float] = {}
        self._grab_cooldown = grab_cooldown

    def reset(self) -> None:
        self._gestures.clear()
        self._pressed.clear()
        self._armed.clear()

    def update(self, snapshot, positions: dict[str, tuple[float, float]], radii: dict[str, float]) -> list[CursorState]:
        now = time.monotonic()
        cursors = []
        for side in (LEFT, RIGHT):
            position = positions.get(side)
            if position is None:
                self._pressed[side] = False
                continue
            hand = snapshot.hand_for_side(side)
            grabbed = False
            if hand is not None:
                gesture = self._gestures.setdefault(side, GestureSmoother()).update(classify_hand_gesture(hand))
                if gesture == FIST:
                    if not self._pressed.get(side) and self._armed.get(side) and now - self._last_grab.get(side, 0.0) >= self._grab_cooldown:
                        grabbed = True
                        self._last_grab[side] = now
                        self._armed[side] = False
                    self._pressed[side] = True
                else:
                    self._pressed[side] = False
                    if gesture == PAPER:
                        self._armed[side] = True
            cursors.append(CursorState(side, position[0], position[1], radii.get(side, 20.0), self._pressed.get(side, False), grabbed))
        return cursors


def draw_hand_cursor(canvas: tk.Canvas, cursor: CursorState, color: str = OPERATING_HAND, tags: str = "pointer") -> None:
    """A bright ring around the avatar hand, drawn on the top layer so buttons never hide it."""
    x, y = cursor.x, cursor.y
    ring = cursor.radius * 1.55
    glow = ring + max(4.0, cursor.radius * 0.25)
    canvas.create_oval(x - glow, y - glow, x + glow, y + glow, outline=dim(color, 0.35), width=max(3, int(cursor.radius * 0.18)), tags=tags)
    canvas.create_oval(x - ring, y - ring, x + ring, y + ring, outline=FAILURE if cursor.pressed else color, width=max(3, int(cursor.radius * 0.16)), tags=tags)


# ---------------------------------------------------------------------------
# Icons (menus use icons instead of written guidance)
# ---------------------------------------------------------------------------
def _star_points(x: float, y: float, radius: float) -> list[float]:
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        length = radius if index % 2 == 0 else radius * 0.45
        points += [x + math.cos(angle) * length, y + math.sin(angle) * length]
    return points


def draw_icon(canvas: tk.Canvas, icon: str, x: float, y: float, size: float, color: str, tags) -> None:
    """Simple line/shape icons sized to ``size`` (half extent)."""
    width = max(3, int(size * 0.14))
    if icon == "pose":
        # a little figure reaching for a target circle
        canvas.create_oval(x - size * 0.18, y - size * 0.78, x + size * 0.18, y - size * 0.42, fill=color, outline="", tags=tags)
        canvas.create_line(x, y - size * 0.35, x, y + size * 0.25, fill=color, width=width, capstyle=tk.ROUND, tags=tags)
        canvas.create_line(x - size * 0.5, y - size * 0.55, x, y - size * 0.2, x + size * 0.45, y + size * 0.05, fill=color, width=width, capstyle=tk.ROUND, joinstyle=tk.ROUND, tags=tags)
        canvas.create_line(x - size * 0.35, y + size * 0.8, x, y + size * 0.25, x + size * 0.35, y + size * 0.8, fill=color, width=width, capstyle=tk.ROUND, joinstyle=tk.ROUND, tags=tags)
        ring = size * 0.26
        canvas.create_oval(x - size * 0.72 - ring, y - size * 0.72 - ring, x - size * 0.72 + ring, y - size * 0.72 + ring, outline="#22D3EE", width=width, tags=tags)
    elif icon == "rhythm":
        for offset, symbol in ((-0.55, FIST), (0.0, SCISSORS), (0.55, PAPER)):
            draw_marker(canvas, x + size * offset, y + size * 0.15 - abs(offset) * size * 0.3, size * 0.24, GESTURE_SYMBOLS[symbol], GESTURE_COLORS[symbol], tags)
    elif icon == "trophy":
        canvas.create_polygon(x - size * 0.5, y - size * 0.6, x + size * 0.5, y - size * 0.6, x + size * 0.35, y + size * 0.05, x - size * 0.35, y + size * 0.05, fill=color, outline="", tags=tags)
        canvas.create_rectangle(x - size * 0.08, y + size * 0.05, x + size * 0.08, y + size * 0.4, fill=color, outline="", tags=tags)
        canvas.create_rectangle(x - size * 0.35, y + size * 0.4, x + size * 0.35, y + size * 0.55, fill=color, outline="", tags=tags)
        canvas.create_arc(x - size * 0.75, y - size * 0.55, x - size * 0.25, y - size * 0.05, start=90, extent=180, style=tk.ARC, outline=color, width=width, tags=tags)
        canvas.create_arc(x + size * 0.25, y - size * 0.55, x + size * 0.75, y - size * 0.05, start=-90, extent=180, style=tk.ARC, outline=color, width=width, tags=tags)
    elif icon == "exit":
        canvas.create_line(x - size * 0.45, y - size * 0.45, x + size * 0.45, y + size * 0.45, fill=color, width=int(width * 1.4), capstyle=tk.ROUND, tags=tags)
        canvas.create_line(x - size * 0.45, y + size * 0.45, x + size * 0.45, y - size * 0.45, fill=color, width=int(width * 1.4), capstyle=tk.ROUND, tags=tags)
    elif icon == "back":
        canvas.create_line(x + size * 0.5, y, x - size * 0.45, y, fill=color, width=int(width * 1.3), capstyle=tk.ROUND, tags=tags)
        canvas.create_line(x - size * 0.05, y - size * 0.42, x - size * 0.48, y, x - size * 0.05, y + size * 0.42, fill=color, width=int(width * 1.3), capstyle=tk.ROUND, joinstyle=tk.ROUND, tags=tags)
    elif icon == "retry":
        canvas.create_arc(x - size * 0.55, y - size * 0.55, x + size * 0.55, y + size * 0.55, start=100, extent=290, style=tk.ARC, outline=color, width=int(width * 1.3), tags=tags)
        tip = (x + math.cos(math.radians(100)) * size * 0.55, y - math.sin(math.radians(100)) * size * 0.55)
        canvas.create_polygon(tip[0] - size * 0.3, tip[1] - size * 0.22, tip[0] + size * 0.12, tip[1] - size * 0.02, tip[0] - size * 0.22, tip[1] + size * 0.3, fill=color, outline="", tags=tags)
    elif icon == "menu":
        cell = size * 0.36
        for dx in (-1, 1):
            for dy in (-1, 1):
                cx, cy = x + dx * cell * 0.62, y + dy * cell * 0.62
                canvas.create_rectangle(cx - cell * 0.45, cy - cell * 0.45, cx + cell * 0.45, cy + cell * 0.45, fill=color, outline="", tags=tags)
    elif icon.startswith("stars"):
        count = int(icon[-1])
        spacing = size * 0.62
        for index in range(count):
            sx = x + (index - (count - 1) / 2) * spacing
            canvas.create_polygon(_star_points(sx, y, size * 0.34), fill=color, outline="", tags=tags)
    elif icon == "pause":
        for dx in (-0.25, 0.25):
            canvas.create_rectangle(x + dx * size - size * 0.12, y - size * 0.45, x + dx * size + size * 0.12, y + size * 0.45, fill=color, outline="", tags=tags)
    elif icon == "crown":
        canvas.create_polygon(x - size * 0.6, y + size * 0.35, x - size * 0.6, y - size * 0.35, x - size * 0.3, y, x, y - size * 0.5,
                              x + size * 0.3, y, x + size * 0.6, y - size * 0.35, x + size * 0.6, y + size * 0.35, fill=color, outline="", tags=tags)


# ---------------------------------------------------------------------------
# Short feedback effects (ripples and judgement text), driven by game time
# ---------------------------------------------------------------------------
@dataclass
class _Effect:
    kind: str
    x: float
    y: float
    color: str
    text: str = ""
    size: float = 0.08
    age: float = 0.0
    duration: float = 0.45


class EffectLayer:
    def __init__(self) -> None:
        self._effects: list[_Effect] = []

    def ripple(self, x: float, y: float, color: str, size: float = 0.08) -> None:
        self._effects.append(_Effect("ripple", x, y, color, size=size))

    def text(self, x: float, y: float, text: str, color: str, duration: float = 0.6) -> None:
        self._effects.append(_Effect("text", x, y, color, text=text, duration=duration))

    def update(self, delta_time: float) -> None:
        for effect in self._effects:
            effect.age += delta_time
        self._effects = [effect for effect in self._effects if effect.age < effect.duration]

    def clear(self) -> None:
        self._effects.clear()

    def draw(self, canvas: tk.Canvas, tags: str = "game") -> None:
        viewport = canvas_viewport(canvas)
        for effect in self._effects:
            progress = effect.age / effect.duration
            x, y = normalized_point(effect.x, effect.y, viewport)
            if effect.kind == "ripple":
                radius = effect.size * viewport[3] * (1.0 + progress * 0.9)
                canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=dim(effect.color, 1.0 - progress), width=max(2, int(8 * (1 - progress))), tags=tags)
            else:
                canvas.create_text(x, y - progress * viewport[3] * 0.04, text=effect.text, fill=dim(effect.color, 1.0 - progress * 0.6), font=ui_font(canvas, 44), tags=tags)
