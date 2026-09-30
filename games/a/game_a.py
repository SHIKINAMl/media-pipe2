"""Game A: single-player Twister-like pose game (ideas/README.md section 3)."""
from __future__ import annotations

import math
import random
import tkinter as tk
from dataclasses import dataclass

from games.common_ui import (
    CAMERA_ASPECT,
    SUCCESS,
    TEXT,
    WARNING,
    AvatarStyle,
    EffectLayer,
    canvas_viewport,
    dim,
    draw_marker,
    mirrored,
    normalized_point,
    ui_font,
    view_distance,
    visible,
)
from games.game_controller import CalibrationCheck, DifficultyInfo, GameInfo, GameResult, pose_parts_visible
from tracking.hand_gesture_classifier import LEFT, RIGHT
from tracking.hand_pose_detector import DetectionSnapshot


@dataclass(frozen=True)
class PartSpec:
    landmark: int
    side: str
    kind: str
    label: str
    color: str
    symbol: str


PARTS = {
    "left_hand": PartSpec(15, LEFT, "hand", "ひだりて", "#22D3EE", "circle"),
    "right_hand": PartSpec(16, RIGHT, "hand", "みぎて", "#F472B6", "diamond"),
    "left_foot": PartSpec(27, LEFT, "foot", "ひだりあし", "#FACC15", "triangle"),
    "right_foot": PartSpec(28, RIGHT, "foot", "みぎあし", "#4ADE80", "square"),
}
KIND_LABEL = {"hand": "て", "foot": "あし"}
# Icon for "either side" instructions (no text on the play field).
KIND_SYMBOL = {"hand": "fist", "foot": "foot"}
# Pale colour for "either side" instructions; the zone takes the official colour once a side is used.
KIND_NEUTRAL = {"hand": "#C4B5FD", "foot": "#FDBA74"}

HOLD_TO_SUCCEED = 0.65
KEEP_MARGIN = 1.25
KEEP_GRACE = 0.4
MAX_TORSO_TILT = math.radians(35)

CONFIG = {
    "easy": {
        "kinds": ("hand",), "strict": False, "max_targets": 2, "time": 6.0, "radius": 0.13,
        "area": (0.26, 0.74, 0.20, 0.82), "reach": (0.45, 0.80),
    },
    "normal": {
        "kinds": ("hand", "foot"), "strict": False, "max_targets": 3, "time": 5.0, "radius": 0.11,
        "area": (0.16, 0.84, 0.14, 0.92), "reach": (0.55, 0.95),
    },
    "hard": {
        "kinds": ("hand", "foot"), "strict": True, "max_targets": 4, "time": 4.0, "radius": 0.095,
        "area": (0.08, 0.92, 0.08, 0.95), "reach": (0.60, 1.05),
    },
}

# Continuous difficulty ramp: both values shrink with the instruction count and
# converge toward 0; the floors are the exhibition-playable limits.
# value(n) = start / (1 + RAMP * n): gentle at first, still heading toward 0 in theory.
TIME_RAMP = 0.025
RADIUS_RAMP = 0.015
MIN_TIME = 1.5
MIN_RADIUS = 0.05


@dataclass
class PoseTarget:
    kind: str
    candidates: tuple[str, ...]
    x: float
    y: float
    radius: float
    time_limit: float
    bound: str | None = None
    hold: float = 0.0
    locked: bool = False
    out_time: float = 0.0
    touching: str | None = None

    def display_part(self) -> str | None:
        if self.bound is not None:
            return self.bound
        if len(self.candidates) == 1:
            return self.candidates[0]
        return None

    def color(self) -> str:
        part = self.display_part()
        return PARTS[part].color if part else KIND_NEUTRAL[self.kind]

    def label(self) -> str:
        part = self.display_part()
        return PARTS[part].label if part else KIND_LABEL[self.kind]


class GameAController:
    def __init__(self, canvas: tk.Canvas, difficulty: str) -> None:
        self.canvas = canvas
        self.difficulty = difficulty
        self._config = CONFIG[difficulty]
        self.finished = False
        self.score = 0
        self.successes = 0
        self.max_kept = 0
        self._elapsed = 0.0
        self._remaining = 0.0
        self._paused = False
        self._targets: list[PoseTarget] = []
        self._rng = random.Random()
        self._effects = EffectLayer()
        self._reason = ""
        self._posture_ok = True
        self._last_part: str | None = None
        self._last_pose = None
        self._seen: dict[str, bool] = {}

    # -- lifecycle ----------------------------------------------------------------
    def enter(self) -> None:
        self._targets = [self._new_target(None)]
        self._remaining = self._targets[-1].time_limit

    def update(self, snapshot: DetectionSnapshot, delta_time: float) -> None:
        self._effects.update(delta_time)
        if self.finished or self._paused or not self._targets:
            return
        self._elapsed += delta_time
        self._remaining -= delta_time
        current = self._targets[-1]
        if self._remaining <= 0:
            self._finish(f"じかん ぎれ（{current.label()}）")
            return
        if not snapshot.poses:
            return
        pose = snapshot.poses[0]
        self._last_pose = pose

        # Keep check: slightly looser than the success check, with a short grace period.
        for target in self._targets:
            if not target.locked:
                continue
            if self._part_inside(pose, target.bound, target, KEEP_MARGIN):
                target.out_time = 0.0
            else:
                target.out_time += delta_time
                if target.out_time > KEEP_GRACE:
                    self._finish(f"{PARTS[target.bound].label}が まるから でちゃった")
                    return

        self._posture_ok = self._torso_tilt(pose) <= MAX_TORSO_TILT
        touching = [part for part in self._free_candidates(current) if self._part_inside(pose, part, current, 1.0)]
        if touching and self._posture_ok:
            part = current.touching if current.touching in touching else touching[0]
            current.touching = part
            current.hold += delta_time
            if current.hold >= HOLD_TO_SUCCEED:
                self._succeed(current, part, pose)
        else:
            current.touching = None
            current.hold = 0.0

    def draw(self, canvas: tk.Canvas) -> None:
        canvas.delete("game")
        canvas.delete("hud")
        viewport = canvas_viewport(canvas)
        width, height = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
        for target in self._targets:
            self._draw_target(canvas, target, viewport)
        self._effects.draw(canvas)

        canvas.create_text(30, 24, text=f"{self.score:,}", anchor="nw", fill=TEXT, font=ui_font(canvas, 64, mono=True), tags="hud")
        time_color = WARNING if self._remaining < 1.5 else TEXT
        canvas.create_text(width / 2, 24, text=f"{max(self._remaining, 0):0.1f}", anchor="n", fill=time_color, font=ui_font(canvas, 72, mono=True), tags="hud")
        current = self._targets[-1] if self._targets else None
        if current is not None and current.time_limit > 0:
            bar_width = width * 0.2
            ratio = max(0.0, self._remaining / current.time_limit)
            top = 24 + 80 * (height / 1080)
            canvas.create_rectangle(width / 2 - bar_width / 2, top, width / 2 + bar_width / 2, top + 8, outline="#334155", tags="hud")
            canvas.create_rectangle(width / 2 - bar_width / 2, top, width / 2 - bar_width / 2 + bar_width * ratio, top + 8, fill=time_color, outline="", tags="hud")
        canvas.create_text(width - 30, 24, text=f"×{self.successes}", anchor="ne", fill=SUCCESS, font=ui_font(canvas, 56, mono=True), tags="hud")
        # Kept targets as dots (filled = currently kept) instead of a written counter.
        kept = sum(1 for target in self._targets if target.locked)
        dot = 10 * (height / 1080)
        for index in range(self._config["max_targets"] - 1):
            dot_x = width - 30 - dot - index * dot * 3
            dot_y = 24 + 90 * (height / 1080)
            filled = index < kept
            canvas.create_oval(dot_x - dot, dot_y - dot, dot_x + dot, dot_y + dot, fill=SUCCESS if filled else "", outline=SUCCESS if filled else "#334155", width=2, tags="hud")
        self._draw_part_indicator(canvas, width, height)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def exit(self) -> None:
        self.canvas.delete("game")
        self.canvas.delete("hud")
        self._targets.clear()
        self._effects.clear()

    def result(self) -> GameResult:
        return GameResult(
            self.score,
            {"せいこう": str(self.successes), "さいだい キープ": str(self.max_kept), "プレイ じかん": f"{self._elapsed:0.1f}びょう"},
            self._reason,
        )

    def avatar_style(self) -> AvatarStyle:
        style = AvatarStyle(brightness=0.65)
        if not self._posture_ok:
            # Leaning too far: tint the torso instead of showing a message.
            style.part_colors["torso"] = WARNING
            style.part_brightness["torso"] = 1.0
        for target in self._targets:
            parts = (target.bound,) if target.bound else self._free_candidates(target)
            for part in parts:
                spec = PARTS[part]
                color = spec.color if (target.bound or len(parts) == 1) else KIND_NEUTRAL[target.kind]
                if target.locked and target.out_time > 0:
                    color = WARNING
                style.part_brightness[part] = 1.0
                style.part_colors[part] = color
                style.part_scale[part] = 1.2
                style.part_marks[part] = (spec.symbol if (target.bound or len(parts) == 1) else KIND_SYMBOL[target.kind], color)
        return style

    # -- rules ----------------------------------------------------------------------
    def _free_candidates(self, target: PoseTarget) -> tuple[str, ...]:
        if target.bound is not None:
            return (target.bound,)
        taken = {other.bound for other in self._targets if other is not target and other.bound}
        return tuple(part for part in target.candidates if part not in taken)

    def _succeed(self, target: PoseTarget, part: str, pose) -> None:
        kept_before = sum(1 for other in self._targets if other.locked)
        target.locked = True
        target.bound = part
        target.hold = 0.0
        self.successes += 1
        points = 100 + int(self._remaining * 20) + 50 * kept_before
        self.score += points
        self.max_kept = max(self.max_kept, kept_before + 1)
        self._effects.ripple(target.x, target.y, PARTS[part].color, size=target.radius)
        self._effects.text(target.x, target.y - target.radius - 0.05, f"+{points}", SUCCESS)
        self._last_part = part
        if len(self._targets) >= self._config["max_targets"]:
            self._targets.pop(0)
        self._targets.append(self._new_target(pose))
        self._remaining = self._targets[-1].time_limit

    def _finish(self, reason: str) -> None:
        self.finished = True
        self._reason = reason

    def _new_target(self, pose) -> PoseTarget:
        bound = {target.bound for target in self._targets if target.bound}
        free = [name for name, spec in PARTS.items() if spec.kind in self._config["kinds"] and name not in bound]
        if self._config["strict"]:
            # Avoid repeating the same side too often in hard mode.
            last_side = PARTS[self._last_part].side if self._last_part else None
            preferred = [name for name in free if PARTS[name].side != last_side] or free
            part = self._rng.choice(preferred)
            kind, candidates = PARTS[part].kind, (part,)
        else:
            kinds = sorted({PARTS[name].kind for name in free})
            kind = self._rng.choice(kinds)
            candidates = tuple(name for name in free if PARTS[name].kind == kind)
        count = self.successes
        radius = max(MIN_RADIUS, self._config["radius"] / (1 + RADIUS_RAMP * count))
        time_limit = max(MIN_TIME, self._config["time"] / (1 + TIME_RAMP * count))
        x, y = self._place(kind, candidates, radius, pose)
        return PoseTarget(kind, candidates, x, y, radius, time_limit)

    def _place(self, kind: str, candidates: tuple[str, ...], radius: float, pose) -> tuple[float, float]:
        left, right, top, bottom = self._config["area"]
        best = None
        for _attempt in range(30):
            x, y = self._sample_position(kind, candidates, pose)
            x = max(left, min(right, x))
            y = max(top, min(bottom, y))
            clearance = min(
                (view_distance((x, y), (other.x, other.y)) - radius - other.radius for other in self._targets),
                default=1.0,
            )
            if clearance > 0.02:
                return x, y
            if best is None or clearance > best[0]:
                best = (clearance, x, y)
        return best[1], best[2]

    def _sample_position(self, kind: str, candidates: tuple[str, ...], pose) -> tuple[float, float]:
        """Place targets relative to the player's body so reach scales with body size."""
        reach_min, reach_max = self._config["reach"]
        side = PARTS[candidates[0]].side if len(candidates) == 1 else None
        sign = -1.0 if side == LEFT else 1.0 if side == RIGHT else self._rng.choice((-1.0, 1.0))
        left, right, top, bottom = self._config["area"]
        if pose is None or not all(visible(pose[index]) for index in (11, 12, 23, 24)):
            if kind == "foot":
                return self._rng.uniform(left, right), self._rng.uniform(bottom - 0.12, bottom)
            return self._rng.uniform(left, right), self._rng.uniform(top, top + (bottom - top) * 0.65)
        left_shoulder, right_shoulder = mirrored(pose[11]), mirrored(pose[12])
        shoulder_width = max(view_distance(left_shoulder, right_shoulder), 0.05)
        if kind == "hand":
            anchor = left_shoulder if side == LEFT else right_shoulder if side == RIGHT else (
                (left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
            reach = shoulder_width * 2.0 * self._rng.uniform(reach_min, reach_max)
            angle = self._rng.uniform(math.radians(-70), math.radians(35))  # up-and-out through slightly below shoulder
            dx, dy = sign * math.cos(angle), math.sin(angle)
            return anchor[0] + reach * dx / CAMERA_ASPECT, anchor[1] + reach * dy
        hip = mirrored(pose[23]) if side == LEFT else mirrored(pose[24]) if side == RIGHT else (
            (mirrored(pose[23])[0] + mirrored(pose[24])[0]) / 2, (pose[23].y + pose[24].y) / 2)
        ground = max(pose[27].y, pose[28].y) if visible(pose[27]) and visible(pose[28]) else bottom
        offset = shoulder_width * self._rng.uniform(0.3, 1.4) * (reach_min + reach_max) / 2
        lift = shoulder_width * self._rng.uniform(0.0, 0.35)
        return hip[0] + sign * offset / CAMERA_ASPECT, ground - lift

    @staticmethod
    def _part_inside(pose, part: str, target: PoseTarget, margin: float) -> bool:
        landmark = pose[PARTS[part].landmark]
        if not visible(landmark, 0.3):
            return False
        return view_distance(mirrored(landmark), (target.x, target.y)) <= target.radius * margin

    @staticmethod
    def _torso_tilt(pose) -> float:
        shoulder = ((pose[11].x + pose[12].x) / 2, (pose[11].y + pose[12].y) / 2)
        hip = ((pose[23].x + pose[24].x) / 2, (pose[23].y + pose[24].y) / 2)
        dx = (shoulder[0] - hip[0]) * CAMERA_ASPECT
        dy = hip[1] - shoulder[1]
        return abs(math.atan2(dx, max(dy, 1e-6)))

    # -- drawing ----------------------------------------------------------------------
    def _draw_target(self, canvas: tk.Canvas, target: PoseTarget, viewport) -> None:
        x, y = normalized_point(target.x, target.y, viewport)
        radius = target.radius * viewport[3]
        color = target.color()
        if target.locked and target.out_time > 0:
            # Slipping out: yellow outline and a shrinking grace gauge instead of instant red.
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=WARNING, width=7, tags="game")
            remaining = 1.0 - target.out_time / KEEP_GRACE
            ring = radius + 12
            canvas.create_arc(x - ring, y - ring, x + ring, y + ring, start=90, extent=359.9 * remaining, style=tk.ARC, outline=WARNING, width=6, tags="game")
        elif target.locked:
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=color, width=8, tags="game")
            inner = radius * 0.82
            canvas.create_oval(x - inner, y - inner, x + inner, y + inner, outline=dim(color, 0.35), width=2, tags="game")
        else:
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=color, width=6, dash=(12, 8), tags="game")
            if target.hold > 0:
                ring = radius + 12
                canvas.create_arc(x - ring, y - ring, x + ring, y + ring, start=90, extent=-359.9 * min(target.hold / HOLD_TO_SUCCEED, 1.0), style=tk.ARC, outline=SUCCESS, width=8, tags="game")
        part = target.display_part()
        symbol = PARTS[part].symbol if part else KIND_SYMBOL[target.kind]
        draw_marker(canvas, x, y, radius * 0.28, symbol, color, "game", fill=not target.locked)

    def _draw_part_indicator(self, canvas: tk.Canvas, width: float, height: float) -> None:
        """Small corner indicator showing which of the four parts are currently recognised."""
        pose = self._last_pose
        size = 16 * (height / 1080)
        x = width - 30 - size
        y = height - 30 - size
        for name in reversed(list(PARTS)):
            spec = PARTS[name]
            if spec.kind not in self._config["kinds"]:
                continue
            # Hysteresis so the indicator does not flicker on borderline visibility.
            was_seen = self._seen.get(name, False)
            seen = pose is not None and visible(pose[spec.landmark], 0.3 if was_seen else 0.55)
            self._seen[name] = seen
            draw_marker(canvas, x, y, size, spec.symbol, spec.color if seen else "#334155", "hud", fill=seen)
            x -= size * 5


def calibration_checks(snapshot: DetectionSnapshot, difficulty: str) -> list[CalibrationCheck]:
    checks = [
        CalibrationCheck("かた と こし", pose_parts_visible(snapshot, (11, 12, 23, 24))),
        CalibrationCheck("ひだりて", pose_parts_visible(snapshot, (15,))),
        CalibrationCheck("みぎて", pose_parts_visible(snapshot, (16,))),
    ]
    if "foot" in CONFIG[difficulty]["kinds"]:
        checks.append(CalibrationCheck("ひだりあし", pose_parts_visible(snapshot, (27,))))
        checks.append(CalibrationCheck("みぎあし", pose_parts_visible(snapshot, (28,))))
    return checks


def tracking_ok(snapshot: DetectionSnapshot, _difficulty: str) -> bool:
    return pose_parts_visible(snapshot, (11, 12), threshold=0.3)


GAME_INFO = GameInfo(
    game_id="GAME_A",
    title="ポーズ ゲーム",
    input_label="からだ ぜんぶ",
    tagline="おなじ いろの まるに てあしを あわせて キープ",
    accent="#22C55E",
    difficulties=(
        DifficultyInfo("easy", "かんたん", ("て だけ・ひだり みぎ どっちでも", "キープは さいだい 1かしょ")),
        DifficultyInfo("normal", "ふつう", ("て と あし・ひだり みぎ どっちでも", "キープは さいだい 2かしょ")),
        DifficultyInfo("hard", "むずかしい", ("ひだりて・みぎて・ひだりあし・みぎあし", "キープは さいだい 3かしょ")),
    ),
    factory=GameAController,
    calibration_checks=calibration_checks,
    tracking_ok=tracking_ok,
    calibration_hint="まると おなじ いろの てあしを、まるの なかで キープ",
    calibration_samples=lambda difficulty: tuple(
        (spec.symbol, spec.label, spec.color) for spec in PARTS.values() if spec.kind in CONFIG[difficulty]["kinds"]
    ),
)
