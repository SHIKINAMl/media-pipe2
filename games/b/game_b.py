"""Game B: osu!-like rhythm game with hand position + hand shape (ideas/README.md section 4)."""
from __future__ import annotations

import math
import random
import tkinter as tk
from dataclasses import dataclass

from games.b.rhythm import EighthNoteRhythm, NotePlacer, SLOTS_PER_BAR, bpm_for
from games.common_ui import (
    CAMERA_ASPECT,
    FAILURE,
    GESTURE_COLORS,
    GESTURE_SYMBOLS,
    SUCCESS,
    TEXT,
    TEXT_SUB,
    AvatarStyle,
    EffectLayer,
    canvas_viewport,
    dim,
    draw_marker,
    normalized_point,
    ui_font,
    view_distance,
)
from games.game_controller import CalibrationCheck, DifficultyInfo, GameInfo, GameResult, pose_parts_visible
from tracking.hand_gesture_classifier import (
    GestureSmoother,
    FIST,
    LEFT,
    PAPER,
    RIGHT,
    SCISSORS,
    classify_hand_gesture_with_confidence,
    palm_center,
)
from tracking.hand_pose_detector import DetectionSnapshot


GESTURE_LABEL = {FIST: "グー", PAPER: "パー", SCISSORS: "チョキ"}
ANY = "any"
BOTH = "both"

PERFECT = "PERFECT"
GOOD = "GOOD"
MISS = "MISS"
JUDGEMENT_LABEL = {PERFECT: "パーフェクト", GOOD: "グッド", MISS: "ミス"}

CONFIG = {
    "easy": {
        "bpm": 80, "gestures": (FIST, PAPER), "on_beat": 0.55, "off_beat": 0.10, "window": 0.30,
        "radius": 0.085, "distance": (0.14, 0.24), "both_notes": False, "left_right": False,
    },
    "normal": {
        "bpm": 100, "gestures": (FIST, PAPER, SCISSORS), "on_beat": 0.70, "off_beat": 0.22, "window": 0.23,
        "radius": 0.078, "distance": (0.15, 0.26), "both_notes": True, "left_right": False,
        "both_probability": 0.12, "both_min_gap": 8,
    },
    "hard": {
        "bpm": 120, "gestures": (FIST, PAPER, SCISSORS), "on_beat": 0.80, "off_beat": 0.35, "window": 0.18,
        "radius": 0.072, "distance": (0.16, 0.28), "both_notes": True, "left_right": True,
        "run_length": (4, 8), "both_min_gap": 6,
    },
}
NOTE_AREA = (0.12, 0.88, 0.30, 0.88)
BOTH_HALF_GAP = 0.13  # half distance between the two circles of a both-hands note (height units)
PERFECT_RATIO = 0.4
MISS_LIMIT = 5
LEAD_IN = 1.6
COMBO_BONUS = 5  # optional per-combo bonus (set to 0 to disable)


@dataclass
class RhythmNote:
    hit_time: float
    x: float
    y: float
    gesture: str
    hand: str
    x2: float | None = None
    y2: float | None = None
    judgement: str | None = None
    early_touch: bool = False
    miss_reason: str = ""

    def positions(self) -> list[tuple[float, float]]:
        if self.hand == BOTH and self.x2 is not None and self.y2 is not None:
            return [(self.x, self.y), (self.x2, self.y2)]
        return [(self.x, self.y)]


@dataclass(frozen=True)
class HandInput:
    side: str
    position: tuple[float, float]
    gesture: str
    confidence: float


@dataclass
class _Bar:
    start: float
    beat: float


class GameBController:
    def __init__(self, canvas: tk.Canvas, difficulty: str) -> None:
        self.canvas = canvas
        self.difficulty = difficulty
        self._config = CONFIG[difficulty]
        self.finished = False
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.misses = 0
        self.perfects = 0
        self.goods = 0
        self._paused = False
        self._rng = random.Random()
        self._time = 0.0
        self._effects = EffectLayer()
        self._rhythm = EighthNoteRhythm(self._rng, self._config["on_beat"], self._config["off_beat"])
        self._placer = NotePlacer(self._rng, NOTE_AREA, self._config["distance"])
        self._notes: list[RhythmNote] = []
        self._bars: list[_Bar] = []
        self._next_bar_time = LEAD_IN
        self._generated = 0
        self._bpm = float(self._config["bpm"])
        self._previous_position = (0.5, 0.6)
        self._previous_direction: tuple[float, float] | None = None
        self._current_side = self._rng.choice((LEFT, RIGHT))
        self._run_remaining = self._rng.randint(*self._config.get("run_length", (4, 8)))
        self._since_both = 0
        self._hands: list[HandInput] = []
        self._reason = ""
        # Display-only smoothing for the corner readout (judging uses the raw per-frame shape).
        self._display_shapes = {LEFT: GestureSmoother(), RIGHT: GestureSmoother()}
        self._display_seen = {LEFT: -10.0, RIGHT: -10.0}

    # -- lifecycle ----------------------------------------------------------------
    def enter(self) -> None:
        self._ensure_chart()

    def update(self, snapshot: DetectionSnapshot, delta_time: float) -> None:
        self._effects.update(delta_time)
        if self.finished or self._paused:
            return
        self._time += delta_time
        self._ensure_chart()
        self._hands = self._hand_inputs(snapshot)
        for hand in self._hands:
            if hand.side in self._display_shapes:
                self._display_shapes[hand.side].update(hand.gesture)
                self._display_seen[hand.side] = self._time
        window = self._config["window"]

        note = self._next_note()
        while note is not None and note.hit_time - self._time < -window:
            if note.early_touch:
                self._judge(note, GOOD)
            else:
                self._judge(note, MISS)
                if self.finished:
                    return
            note = self._next_note()
        if note is None or abs(note.hit_time - self._time) > window:
            return
        offset = note.hit_time - self._time
        matched, reason, taken_by = self._match(note)
        if matched:
            if abs(offset) <= window * PERFECT_RATIO:
                self._judge(note, PERFECT, taken_by)
            elif offset < 0:
                self._judge(note, GOOD, taken_by)
            else:
                # Early but inside the window: wait for a better timing unless the hand leaves.
                note.early_touch = True
        else:
            if note.early_touch:
                self._judge(note, GOOD)
            elif reason:
                note.miss_reason = reason

    def draw(self, canvas: tk.Canvas) -> None:
        canvas.delete("game")
        canvas.delete("hud")
        viewport = canvas_viewport(canvas)
        width, height = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
        self._draw_field(canvas, viewport)
        self._effects.draw(canvas)
        self._draw_timeline(canvas, width, height)
        canvas.create_text(30, 24, text=f"{self.score:,}", anchor="nw", fill=TEXT, font=ui_font(canvas, 64, mono=True), tags="hud")
        canvas.create_text(width - 30, 24, text=f"×{self.combo}", anchor="ne", fill=SUCCESS, font=ui_font(canvas, 56, mono=True), tags="hud")
        miss_marks = "✕" * self.misses + "・" * (MISS_LIMIT - self.misses)
        canvas.create_text(width - 30, 24 + 64 * (height / 1080), text=miss_marks, anchor="ne", fill=FAILURE if self.misses else TEXT_SUB, font=ui_font(canvas, 34), tags="hud")
        self._draw_hand_state(canvas, width, height)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def exit(self) -> None:
        self.canvas.delete("game")
        self.canvas.delete("hud")
        self._notes.clear()
        self._effects.clear()

    def result(self) -> GameResult:
        judged = self.perfects + self.goods + self.misses
        rate = (self.perfects + self.goods) / judged * 100 if judged else 0.0
        return GameResult(
            self.score,
            {"さいだい コンボ": str(self.max_combo), "せいこう りつ": f"{rate:0.0f}%", "パーフェクト": str(self.perfects)},
            self._reason,
        )

    def avatar_style(self) -> AvatarStyle:
        style = AvatarStyle(brightness=0.45, show_gestures=True)
        for prefix in ("left", "right"):
            style.part_brightness[f"{prefix}_forearm"] = 1.0
            style.part_brightness[f"{prefix}_hand"] = 1.0
            for leg in ("thigh", "shin", "foot"):
                style.part_brightness[f"{prefix}_{leg}"] = 0.3
        note = self._next_note()
        if note is None:
            return style
        if note.hand in (LEFT, RIGHT):
            target = "left" if note.hand == LEFT else "right"
            other = "right" if note.hand == LEFT else "left"
            style.part_scale[f"{target}_hand"] = 1.25
            style.part_scale[f"{target}_forearm"] = 1.2
            style.part_brightness[f"{other}_hand"] = 0.5
            style.part_brightness[f"{other}_forearm"] = 0.5
        for hand in self._hands:
            if hand.gesture == note.gesture and (note.hand in (ANY, BOTH) or note.hand == hand.side):
                style.hand_rings[hand.side] = SUCCESS
        return style

    # -- chart generation -------------------------------------------------------------
    def _ensure_chart(self) -> None:
        lookahead = self._time + 4 * 60.0 / self._bpm + 2.5
        while self._next_bar_time < lookahead:
            self._bpm = bpm_for(self._config["bpm"], self._generated)
            beat = 60.0 / self._bpm
            bar = _Bar(self._next_bar_time, beat)
            self._bars.append(bar)
            eighth = beat / 2
            previous_gesture: str | None = None
            previous_hand: str | None = None
            for slot in self._rhythm.generate_bar():
                off_beat = slot % 2 == 1
                gesture = previous_gesture if off_beat and previous_gesture else self._rng.choice(self._config["gestures"])
                hand = self._choose_hand(off_beat, previous_hand)
                note = self._place_note(bar.start + slot * eighth, gesture, hand)
                self._notes.append(note)
                previous_gesture, previous_hand = gesture, hand
                self._generated += 1
            self._next_bar_time += beat * SLOTS_PER_BAR / 2
        cutoff = self._time - 2.0
        self._notes = [note for note in self._notes if note.judgement is None or note.hit_time > cutoff]
        self._bars = [bar for bar in self._bars if bar.start + bar.beat * 4 > cutoff]

    def _choose_hand(self, off_beat: bool, previous_hand: str | None) -> str:
        config = self._config
        self._since_both += 1
        if off_beat and previous_hand in (LEFT, RIGHT, ANY):
            return previous_hand
        if config["left_right"]:
            # Same hand in a row; switch sides only through a both-hands note.
            if self._run_remaining <= 0 and self._since_both >= config["both_min_gap"] and not off_beat:
                self._since_both = 0
                self._current_side = RIGHT if self._current_side == LEFT else LEFT
                self._run_remaining = self._rng.randint(*config["run_length"])
                return BOTH
            self._run_remaining -= 1
            return self._current_side
        if config["both_notes"] and not off_beat and self._since_both >= config["both_min_gap"]:
            if self._rng.random() < config["both_probability"]:
                self._since_both = 0
                return BOTH
        return ANY

    def _place_note(self, hit_time: float, gesture: str, hand: str) -> RhythmNote:
        side_bias = 0.36 if hand == LEFT else 0.64 if hand == RIGHT else (0.5 if hand == BOTH else None)
        margin = BOTH_HALF_GAP + self._config["radius"] if hand == BOTH else 0.0
        x, y = self._placer.next(self._previous_position, self._previous_direction, side_bias, margin)
        dx, dy = (x - self._previous_position[0]) * CAMERA_ASPECT, y - self._previous_position[1]
        length = math.hypot(dx, dy)
        if length > 1e-6:
            self._previous_direction = (dx / length, dy / length)
        self._previous_position = (x, y)
        if hand == BOTH:
            # Mirrored display: the player's left hand appears on the left of the screen.
            offset = BOTH_HALF_GAP / CAMERA_ASPECT
            return RhythmNote(hit_time, x - offset, y, gesture, hand, x + offset, y)
        return RhythmNote(hit_time, x, y, gesture, hand)

    # -- judging ------------------------------------------------------------------------
    def _next_note(self) -> RhythmNote | None:
        return next((note for note in self._notes if note.judgement is None), None)

    def _hand_inputs(self, snapshot: DetectionSnapshot) -> list[HandInput]:
        inputs = []
        for index, hand in enumerate(snapshot.hands):
            side = snapshot.hand_sides[index] if index < len(snapshot.hand_sides) else ANY
            gesture, confidence = classify_hand_gesture_with_confidence(hand)
            center_x, center_y = palm_center(hand)
            inputs.append(HandInput(side, (1.0 - center_x, center_y), gesture, confidence))
        return inputs

    def _match(self, note: RhythmNote) -> tuple[bool, str, list[HandInput]]:
        """Return (success, miss reason hint, hands that took the note). Uses raw per-frame input."""
        radius = self._config["radius"]
        if note.hand == BOTH:
            return self._match_both(note, radius)
        position = (note.x, note.y)
        eligible = [hand for hand in self._hands if note.hand == ANY or hand.side == note.hand]
        inside = [hand for hand in eligible if view_distance(hand.position, position) <= radius]
        good = [hand for hand in inside if hand.gesture == note.gesture]
        if good:
            # Two hands on one note count once: prefer the closer, more confident hand.
            best = min(good, key=lambda hand: view_distance(hand.position, position) - 0.02 * hand.confidence)
            return True, "", [best]
        if inside:
            return False, "てのかたち", []
        wrong_side = [hand for hand in self._hands if hand not in eligible and view_distance(hand.position, position) <= radius]
        if wrong_side:
            return False, "はんたいの て", []
        near = [hand for hand in eligible if view_distance(hand.position, position) <= radius * 2.0]
        return False, "ばしょ" if near else "", []

    def _match_both(self, note: RhythmNote, radius: float) -> tuple[bool, str, list[HandInput]]:
        circles = note.positions()
        if len(self._hands) < 2:
            return False, "りょうて", []
        first, second = self._hands[0], self._hands[1]
        if self._config["left_right"]:
            left = next((hand for hand in self._hands if hand.side == LEFT), None)
            right = next((hand for hand in self._hands if hand.side == RIGHT), None)
            assignments = [(left, right)] if left and right else []
        else:
            assignments = [(first, second), (second, first)]
        shape_problem = False
        for hand_a, hand_b in assignments:
            if view_distance(hand_a.position, circles[0]) <= radius and view_distance(hand_b.position, circles[1]) <= radius:
                if hand_a.gesture == note.gesture and hand_b.gesture == note.gesture:
                    return True, "", [hand_a, hand_b]
                shape_problem = True
        return False, "てのかたち" if shape_problem else "りょうて", []

    def _judge(self, note: RhythmNote, judgement: str, hands: list[HandInput] | None = None) -> None:
        note.judgement = judgement
        color = GESTURE_COLORS[note.gesture]
        label_y = note.y - self._config["radius"] - 0.06
        if judgement == MISS:
            self.misses += 1
            self.combo = 0
            reason = f" ({note.miss_reason})" if note.miss_reason else ""
            self._effects.text(note.x, label_y, f"ミス{reason}", FAILURE, duration=0.7)
            if self.misses >= MISS_LIMIT:
                self.finished = True
                self._reason = f"ミスが {MISS_LIMIT}かい"
            return
        points = 300 if judgement == PERFECT else 100
        if note.hand == BOTH:
            points = int(points * 1.5)
        points += self.combo * COMBO_BONUS
        self.score += points
        self.combo += 1
        self.max_combo = max(self.max_combo, self.combo)
        if judgement == PERFECT:
            self.perfects += 1
        else:
            self.goods += 1
        # Ripple only where the hand(s) that actually took the note are.
        ripple_points = [hand.position for hand in hands] if hands else note.positions()
        for x, y in ripple_points:
            self._effects.ripple(x, y, color, size=self._config["radius"])
        self._effects.text(note.x, label_y, JUDGEMENT_LABEL[judgement], SUCCESS if judgement == PERFECT else TEXT)

    # -- drawing --------------------------------------------------------------------------
    def _draw_field(self, canvas: tk.Canvas, viewport) -> None:
        approach = 2 * 60.0 / self._bpm
        upcoming = [note for note in self._notes if note.judgement is None and note.hit_time - self._time <= approach]
        radius_px = self._config["radius"] * viewport[3]
        # Flow line through the visible notes so the path reads as a curve.
        path = [normalized_point(*note.positions()[0], viewport) for note in upcoming[:4]]
        if len(path) >= 2:
            flat = [value for point in path for value in point]
            canvas.create_line(*flat, fill="#1E293B", width=max(3, int(radius_px * 0.25)), smooth=True, capstyle=tk.ROUND, tags="game")
        for order, note in reversed(list(enumerate(upcoming))):
            offset = note.hit_time - self._time
            brightness = 1.0 if order == 0 else max(0.35, 0.8 - order * 0.15)
            color = dim(GESTURE_COLORS[note.gesture], brightness)
            points = [normalized_point(x, y, viewport) for x, y in note.positions()]
            if len(points) == 2:
                canvas.create_line(*points[0], *points[1], fill=color, width=4, dash=(8, 6), tags="game")
            for x, y in points:
                ring = radius_px * (1.0 + 1.3 * max(0.0, offset) / approach)
                canvas.create_oval(x - ring, y - ring, x + ring, y + ring, outline=color, width=3, tags="game")
                canvas.create_oval(x - radius_px, y - radius_px, x + radius_px, y + radius_px, outline=color, width=8 if order == 0 else 5, tags="game")
                draw_marker(canvas, x, y, radius_px * 0.42, GESTURE_SYMBOLS[note.gesture], color, "game")

    def _draw_timeline(self, canvas: tk.Canvas, width: float, height: float) -> None:
        """One bar of upcoming hit timings scrolling toward the hit line (top lane)."""
        lane_left, lane_right = width * 0.26, width * 0.74
        lane_y = height * 0.075
        lane_half = height * 0.035
        bar_length = 4 * 60.0 / self._bpm
        canvas.create_rectangle(lane_left, lane_y - lane_half, lane_right, lane_y + lane_half, fill="#0B1220", outline="#1E293B", width=2, tags="hud")
        scale = (lane_right - lane_left) / bar_length
        for bar in self._bars:
            for beat_index in range(4):
                beat_time = bar.start + beat_index * bar.beat - self._time
                if 0 <= beat_time <= bar_length:
                    x = lane_left + beat_time * scale
                    canvas.create_line(x, lane_y - lane_half, x, lane_y + lane_half, fill="#334155" if beat_index else "#64748B", width=2, tags="hud")
        canvas.create_line(lane_left, lane_y - lane_half - 8, lane_left, lane_y + lane_half + 8, fill=TEXT, width=5, tags="hud")
        visible_notes = [note for note in self._notes if note.judgement is None and -0.1 <= note.hit_time - self._time <= bar_length]
        for order, note in reversed(list(enumerate(visible_notes))):
            offset = note.hit_time - self._time
            x = lane_left + max(0.0, offset) * scale
            nearest = order == 0
            size = lane_half * (0.85 if nearest else 0.55)
            color = dim(GESTURE_COLORS[note.gesture], 1.0 if nearest else max(0.35, 1.0 - offset / bar_length))
            if note.hand == BOTH:
                canvas.create_oval(x - size, lane_y - size * 1.5, x + size, lane_y - size * 0.1, outline=color, width=3, tags="hud")
                canvas.create_oval(x - size, lane_y + size * 0.1, x + size, lane_y + size * 1.5, outline=color, width=3, tags="hud")
            else:
                draw_marker(canvas, x, lane_y, size * 0.8, GESTURE_SYMBOLS[note.gesture], color, "hud")

    def _draw_hand_state(self, canvas: tk.Canvas, width: float, height: float) -> None:
        # Recognised hand shapes as icons (left hand on the left, mirrored like the avatar).
        size = 26 * (height / 1080)
        y = height - 30 - size * 1.6
        x = 30 + size * 1.5
        for side in (LEFT, RIGHT):
            gesture = self._display_shapes[side].current
            seen = self._time - self._display_seen[side] <= 0.4
            draw_marker(canvas, x, y, size, GESTURE_SYMBOLS[gesture] if seen else "fist", GESTURE_COLORS[gesture] if seen else "#334155", "hud")
            x += size * 4


def calibration_checks(snapshot: DetectionSnapshot, difficulty: str) -> list[CalibrationCheck]:
    checks = [CalibrationCheck("からだの うえ はんぶん", pose_parts_visible(snapshot, (11, 12)))]
    if difficulty == "easy":
        checks.append(CalibrationCheck("て（かたてで オッケー）", bool(snapshot.hands)))
    else:
        checks.append(CalibrationCheck("ひだりて", snapshot.hand_for_side(LEFT) is not None))
        checks.append(CalibrationCheck("みぎて", snapshot.hand_for_side(RIGHT) is not None))
    return checks


def tracking_ok(snapshot: DetectionSnapshot, _difficulty: str) -> bool:
    return bool(snapshot.hands) or pose_parts_visible(snapshot, (11, 12), threshold=0.3)


GAME_INFO = GameInfo(
    game_id="GAME_B",
    title="リズム ゲーム",
    input_label="ての ばしょと かたち",
    tagline="リズムに あわせて グー・チョキ・パーで タッチ",
    accent="#F59E0B",
    difficulties=(
        DifficultyInfo("easy", "かんたん", ("かたて・グーと パーだけ", "テンポ 80 から")),
        DifficultyInfo("normal", "ふつう", ("グー・チョキ・パー・りょうて ノーツ あり", "テンポ 100 から")),
        DifficultyInfo("hard", "むずかしい", ("ひだりて みぎてを してい・りょうてで きりかえ", "テンポ 120 から")),
    ),
    factory=GameBController,
    calibration_checks=calibration_checks,
    tracking_ok=tracking_ok,
    calibration_hint="てを まるに かさねて、おなじ てのかたちで タイミングよく タッチ",
    calibration_samples=lambda difficulty: tuple(
        (GESTURE_SYMBOLS[gesture], GESTURE_LABEL[gesture], GESTURE_COLORS[gesture]) for gesture in CONFIG[difficulty]["gestures"]
    ),
)
