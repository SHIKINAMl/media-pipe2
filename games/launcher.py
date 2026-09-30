"""Single-window screen host (ideas/README.md 2-4, 9-5, 9-13).

One Tk root, one Canvas, one ``mainloop``. Every screen, game, overlay and error
display is drawn on the same canvas; the camera / MediaPipe detector is opened
once and shared by all screens.

Menus avoid written guidance: round icon buttons are placed within arm's reach
around the player's body, both avatar hands act as cursors, and a button is chosen
by holding a hand over it (a closed fist chooses immediately).
"""
from __future__ import annotations

import math
import time
import tkinter as tk
from dataclasses import dataclass
from typing import Callable

from games.common_ui import (
    ACCENT,
    BACKGROUND,
    FAILURE,
    OPERATING_HAND,
    SUCCESS,
    TEXT,
    TEXT_SUB,
    WARNING,
    AvatarRenderer,
    AvatarStyle,
    HandCursors,
    canvas_viewport,
    dim,
    draw_hand_cursor,
    draw_icon,
    draw_marker,
    normalized_point,
    ui_font,
    ui_scale,
    visible,
)
from games.game_controller import GameController, GameInfo, GameResult
from games.ranking_store import RankingOutcome, add_score, load_rankings, reset_rankings
from games.registry import GAMES
from games.session_log import log_event, log_exception
from tracking.hand_pose_detector import (
    STATUS_CAMERA_LOST,
    STATUS_CAMERA_UNAVAILABLE,
    STATUS_MODEL_MISSING,
    STATUS_READY,
    DetectionSnapshot,
    HandPoseDetector,
)


ATTRACT = "attract"
SELECT = "select"
DIFFICULTY = "difficulty"
RANKING = "ranking"
CALIBRATION = "calibration"
COUNTDOWN = "countdown"
PLAY = "play"
RECOVERY = "recovery"
RESULT = "result"

POINTER_SCREENS = (SELECT, DIFFICULTY, RANKING, CALIBRATION, RESULT)
IDLE_TIMEOUT = {SELECT: 30.0, DIFFICULTY: 30.0, RANKING: 30.0, CALIBRATION: 30.0, RESULT: 20.0}
DIFFICULTIES = ("easy", "normal", "hard")
DIFFICULTY_ICON = {"easy": "stars1", "normal": "stars2", "hard": "stars3"}
DIFFICULTY_COLOR = {"easy": "#4ADE80", "normal": "#FACC15", "hard": "#FB7185"}
GAME_ICON = {"GAME_A": "pose", "GAME_B": "rhythm"}

# Provisional values; tune on site.
ATTRACT_STABLE = 0.8
DWELL_TIME = 1.0          # hold a hand over a button to choose it
EXIT_DWELL_TIME = 2.0     # irreversible actions need a longer hold
GRAB_MIN_HOVER = 0.2      # a closed fist chooses immediately after this much hover
SCREEN_INPUT_DELAY = 0.5
DECISION_FLASH = 0.12
CALIBRATION_STABLE = 1.0
CALIBRATION_CANCEL_GRACE = 1.0
COUNTDOWN_STEP = 1.0
LOSS_TOLERANCE = 0.5
RECOVERY_TIMEOUT = 10.0
RECOVERY_STABLE = 0.5
RESUME_STEP = 0.7
RESULT_COUNTDOWN_FROM = 10.0
START_BANNER = 0.6

FRAME_MS = 16

ERROR_MESSAGES = {
    STATUS_MODEL_MISSING: ("ただいま じゅんびちゅう です", "スタッフを よんでね", "E01"),
    STATUS_CAMERA_UNAVAILABLE: ("カメラが みつかりません", "スタッフを よんでね", "E02"),
    STATUS_CAMERA_LOST: ("カメラの えいぞうが とまりました", "スタッフを よんでね", "E03"),
}


@dataclass
class UiButton:
    """Round icon button placed within arm's reach of the player."""

    key: str
    center: tuple[float, float]
    radius: float
    icon: str
    accent: str
    command: Callable[[], None]
    dwell: float = DWELL_TIME
    label: str = ""
    selected: bool = False

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return math.dist((x, y), self.center) <= self.radius + margin


class MenuAnchor:
    """Tracks the player's shoulders (heavily smoothed) so menu buttons sit around the body.

    Buttons are placed on arcs of arm length around each shoulder: they never cover
    the body, and every button is reachable with the real hand position (no stretched
    pointer mapping, so the cursor is exactly the avatar hand).
    """

    REACH = 1.5  # arm length in shoulder widths (shoulder -> palm)
    TIME_CONSTANT = 0.5

    def __init__(self) -> None:
        self.center: tuple[float, float] | None = None
        self.shoulder_width = 0.0
        self._viewport_size: tuple[float, float] | None = None

    def update(self, snapshot: DetectionSnapshot, viewport, delta_time: float, frozen: bool) -> None:
        left, top, width, height = viewport
        target_center = (left + width * 0.5, top + height * 0.42)
        target_width = height * 0.13
        pose = snapshot.poses[0] if snapshot.poses else None
        if pose is not None and len(pose) > 12 and visible(pose[11]) and visible(pose[12]):
            first = (left + (1.0 - pose[11].x) * width, top + pose[11].y * height)
            second = (left + (1.0 - pose[12].x) * width, top + pose[12].y * height)
            target_center = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
            target_width = math.dist(first, second)
        target_width = max(height * 0.09, min(height * 0.22, target_width))
        if self.center is None or self._viewport_size != (width, height):
            # First frame or window resized: jump straight to the body.
            self.center, self.shoulder_width = target_center, target_width
            self._viewport_size = (width, height)
            return
        if frozen and math.dist(target_center, self.center) < self.shoulder_width * 0.6:
            # Don't slide a button away from a hand that is on it, unless the player really moved.
            return
        alpha = 1.0 - math.exp(-delta_time / self.TIME_CONSTANT)
        self.center = (
            self.center[0] + (target_center[0] - self.center[0]) * alpha,
            self.center[1] + (target_center[1] - self.center[1]) * alpha,
        )
        self.shoulder_width += (target_width - self.shoulder_width) * alpha

    def side_point(self, side: int, elevation_degrees: float, factor: float = 0.9) -> tuple[float, float]:
        """Point reached by the arm on screen side ``side`` (-1 left, +1 right), raised by ``elevation_degrees``."""
        center = self.center or (0.0, 0.0)
        shoulder = (center[0] + side * self.shoulder_width / 2, center[1])
        reach = self.shoulder_width * self.REACH * factor
        angle = math.radians(elevation_degrees)
        return shoulder[0] + side * math.cos(angle) * reach, shoulder[1] - math.sin(angle) * reach

    def button_radius(self, scale: float = 1.0) -> float:
        return self.shoulder_width * 0.42 * scale


class LauncherApp:
    def __init__(self, root: tk.Tk, detector: HandPoseDetector | None = None) -> None:
        self.root = root
        self.root.title("モーション ゲーム")
        self.root.geometry("1280x720")
        self.root.minsize(960, 540)
        self.root.configure(bg=BACKGROUND)
        self.canvas = tk.Canvas(root, bg=BACKGROUND, highlightthickness=0, cursor="none")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.renderer = AvatarRenderer(self.canvas)
        self.cursors = HandCursors()
        self.anchor = MenuAnchor()
        self.detector = detector or HandPoseDetector()
        self.snapshot = DetectionSnapshot()

        self.screen = ATTRACT
        self.game_id = "GAME_A"
        self.difficulty = "easy"
        self.ranking_game = "GAME_A"
        self.ranking_difficulty = "easy"
        self.game: GameController | None = None
        self.last_result: GameResult | None = None
        self.last_outcome: RankingOutcome | None = None

        now = time.monotonic()
        self._screen_since = now
        self._last_input = now
        self._last_frame = now
        self._buttons: list[UiButton] = []
        self._dwell: dict[str, float] = {}
        self._hovered: set[str] = set()
        self._pending: tuple[str, Callable[[], None], float] | None = None
        self._attract_since: float | None = None
        self._calibration_ok_since: float | None = None
        self._countdown_since = now
        self._lost_since: float | None = None
        self._recovery_since = now
        self._recovery_ok_since: float | None = None
        self._start_banner_until = 0.0
        self._system_message: tuple[str, float] | None = None
        self._ranking_reset_armed_until = 0.0
        self._show_diagnostics = False
        self._stable_flags: dict[str, tuple[bool, bool, float]] = {}
        self._frame_times: list[float] = []
        self._last_status = ""
        self._tick_job: str | None = None
        self._closed = False

        self.canvas.bind("<Button-1>", self._on_mouse_click)
        self.canvas.bind("<Motion>", lambda _event: self._touch_input())
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("<Escape>", lambda _event: self.root.attributes("-fullscreen", False))
        self.root.bind("<F1>", self._toggle_diagnostics)
        self.root.bind("<F5>", self._operator_reset)
        self.root.bind("<Control-Shift-R>", self._ranking_reset)
        self.root.bind("<BackSpace>", self._operator_abort)
        self.root.bind("<Return>", self._operator_force_start)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        log_event("app_start")
        self.root.after(1, self._start_detector)
        self._tick_job = self.root.after(FRAME_MS, self._tick)

    # ------------------------------------------------------------------ main loop
    def _start_detector(self) -> None:
        self.detector.start()
        self._report_status()

    def _tick(self) -> None:
        if self._closed:
            return
        now = time.monotonic()
        delta_time = min(now - self._last_frame, 0.1)
        self._last_frame = now
        self._frame_times = [value for value in self._frame_times if now - value < 1.0] + [now]
        try:
            self.snapshot = self.detector.update()
        except Exception:
            log_exception("detector update failed")
            self.snapshot = DetectionSnapshot()
        self._report_status()

        self.canvas.delete("screen", "panel", "overlay", "pointer")
        self._buttons = []
        self._run_pending(now)
        if self._closed:
            return
        self._update_screen(now, delta_time)
        if self._closed:
            return
        self.anchor.update(self.snapshot, canvas_viewport(self.canvas), delta_time, frozen=bool(self._hovered) or self._pending is not None)
        self.renderer.draw(self.snapshot, self._avatar_style())
        self._draw_screen(now)
        self._handle_cursors(now, delta_time)
        self._draw_overlays(now)
        self._arrange_layers()
        self._tick_job = self.root.after(FRAME_MS, self._tick)

    def _arrange_layers(self) -> None:
        canvas = self.canvas
        # Buttons sit under the avatar so hands are always visible on top of them;
        # they are placed around the body, so the body itself never covers them.
        canvas.tag_lower("avatar")
        canvas.tag_lower("screen")
        # Targets / notes sit between the torso and the front limbs.
        if canvas.find_withtag("avatar_mid") and canvas.find_withtag("game"):
            canvas.tag_raise("game", "avatar_mid")
        for tag in ("hud", "panel", "overlay", "pointer"):
            canvas.tag_raise(tag)

    # ------------------------------------------------------------------ transitions
    def _go(self, screen: str) -> None:
        log_event("screen", previous=self.screen, next=screen, game=self.game_id, difficulty=self.difficulty)
        now = time.monotonic()
        self.screen = screen
        self._screen_since = now
        self._last_input = now
        self._dwell.clear()
        self._hovered.clear()
        self._pending = None
        self._attract_since = None
        self._calibration_ok_since = None
        if screen == ATTRACT:
            self.cursors.reset()

    def _activate(self, button: UiButton, source: str) -> None:
        if self._pending is not None:
            return
        log_event("decision", screen=self.screen, button=button.key, source=source)
        self._touch_input()
        # Flash the chosen button briefly before switching, and lock input meanwhile.
        self._pending = (button.key, button.command, time.monotonic() + DECISION_FLASH)

    def _run_pending(self, now: float) -> None:
        if self._pending is not None and now >= self._pending[2]:
            _key, command, _until = self._pending
            self._pending = None
            self._dwell.clear()
            command()

    def _input_locked(self, now: float) -> bool:
        return self._pending is not None or now - self._screen_since < SCREEN_INPUT_DELAY

    def _touch_input(self) -> None:
        self._last_input = time.monotonic()

    # ------------------------------------------------------------------ screen logic
    def _update_screen(self, now: float, delta_time: float) -> None:
        timeout = IDLE_TIMEOUT.get(self.screen)
        if timeout is not None and now - self._last_input > timeout:
            log_event("idle_timeout", screen=self.screen)
            self._go(ATTRACT)
            return
        if self.screen == ATTRACT:
            present = bool(self.snapshot.poses) and bool(self.snapshot.hands)
            if not present:
                self._attract_since = None
            elif self._attract_since is None:
                self._attract_since = now
            elif now - self._attract_since >= ATTRACT_STABLE:
                self._go(SELECT)
        elif self.screen == CALIBRATION:
            if self.snapshot.poses:
                self._touch_input()  # someone is standing there and working on it
            if all(check.ok for check in self._calibration_checks()):
                if self._calibration_ok_since is None:
                    self._calibration_ok_since = now
                elif now - self._calibration_ok_since >= CALIBRATION_STABLE + CALIBRATION_CANCEL_GRACE:
                    self._begin_countdown()
            else:
                self._calibration_ok_since = None
        elif self.screen == COUNTDOWN:
            if not self._tracking_ok(now):
                log_event("countdown_tracking_lost")
                self._discard_game()
                self._go(CALIBRATION)
            elif now - self._countdown_since >= COUNTDOWN_STEP * 3:
                self._start_play(now)
        elif self.screen == PLAY:
            self._update_play(now, delta_time)
        elif self.screen == RECOVERY:
            self._update_recovery(now)

    def _update_play(self, now: float, delta_time: float) -> None:
        if self.game is None:
            self._go(SELECT)
            return
        if not self._tracking_ok(now):
            log_event("tracking_lost", game=self.game_id)
            self.game.pause()
            self._recovery_since = now
            self._recovery_ok_since = None
            self.screen = RECOVERY
            self._screen_since = now
            return
        try:
            self.game.update(self.snapshot, delta_time)
        except Exception:
            self._game_crashed("update")
            return
        if self.game.finished:
            self._finish_game()

    def _update_recovery(self, now: float) -> None:
        if now - self._recovery_since > RECOVERY_TIMEOUT:
            log_event("recovery_timeout", game=self.game_id)
            self._discard_game()  # not registered in the ranking
            self._go(ATTRACT)
            return
        info = self._info()
        if info.tracking_ok(self.snapshot, self.difficulty) and not self.detector.has_error:
            if self._recovery_ok_since is None:
                self._recovery_ok_since = now
            elif now - self._recovery_ok_since >= RECOVERY_STABLE + RESUME_STEP * 3:
                log_event("tracking_recovered", seconds=f"{now - self._recovery_since:0.1f}")
                self._lost_since = None
                self.screen = PLAY
                self._screen_since = now
                if self.game is not None:
                    self.game.resume()
        else:
            self._recovery_ok_since = None

    def _tracking_ok(self, now: float) -> bool:
        """Ignore dropouts shorter than LOSS_TOLERANCE."""
        ok = self._info().tracking_ok(self.snapshot, self.difficulty) and not self.detector.has_error
        if ok:
            self._lost_since = None
            return True
        if self._lost_since is None:
            self._lost_since = now
        return now - self._lost_since < LOSS_TOLERANCE

    def _calibration_checks(self):
        return self._info().calibration_checks(self.snapshot, self.difficulty)

    def _info(self) -> GameInfo:
        return GAMES[self.game_id]

    # ------------------------------------------------------------------ game flow
    def _choose_game(self, game_id: str) -> None:
        self.game_id = game_id
        self._go(DIFFICULTY)

    def _choose_difficulty(self, difficulty: str) -> None:
        self.difficulty = difficulty
        self._go(CALIBRATION)

    def _begin_countdown(self) -> None:
        self._discard_game()
        self.game = self._info().factory(self.canvas, self.difficulty)
        self.game.enter()
        self.game.pause()
        self._lost_since = None
        self._go(COUNTDOWN)
        self._countdown_since = time.monotonic()

    def _start_play(self, now: float) -> None:
        if self.game is None:
            self._go(CALIBRATION)
            return
        self.game.resume()
        self._lost_since = None
        self._start_banner_until = now + START_BANNER
        log_event("play_start", game=self.game_id, difficulty=self.difficulty)
        self.screen = PLAY
        self._screen_since = now

    def _finish_game(self) -> None:
        if self.game is None:
            return
        result = self.game.result()
        self._discard_game()
        self.last_result = result
        try:
            self.last_outcome = add_score(self.game_id, self.difficulty, result.score)
        except OSError:
            log_exception("ranking save failed")
            self.last_outcome = None
        log_event("result", game=self.game_id, difficulty=self.difficulty, score=result.score,
                  rank=self.last_outcome.rank if self.last_outcome else None, reason=result.reason)
        self._go(RESULT)

    def _discard_game(self) -> None:
        if self.game is not None:
            try:
                self.game.exit()
            except Exception:
                log_exception("game exit failed")
            self.game = None
        self.canvas.delete("game", "hud")

    def _game_crashed(self, stage: str) -> None:
        log_exception(f"game {stage} failed")
        self.game = None
        self.canvas.delete("game", "hud")
        self._system_message = ("ゲームを とめました（スタッフを よんでね）", time.monotonic() + 6.0)
        self._go(SELECT)

    # ------------------------------------------------------------------ hand cursors
    def _handle_cursors(self, now: float, delta_time: float) -> None:
        if self.screen not in POINTER_SCREENS:
            self._dwell.clear()
            self._hovered.clear()
            return
        cursors = self.cursors.update(self.snapshot, self.renderer.hand_screen_positions, self.renderer.hand_screen_radius)
        locked = self._input_locked(now)
        hovered: set[str] = set()
        cursor_colors: dict[str, str] = {}
        for button in self._buttons:
            on_button = [cursor for cursor in cursors if button.contains(cursor.x, cursor.y, margin=cursor.radius * 0.6)]
            if not on_button:
                self._dwell.pop(button.key, None)
                continue
            hovered.add(button.key)
            for cursor in on_button:
                cursor_colors[cursor.side] = button.accent
            if button.key not in self._hovered:
                self._touch_input()
            if locked:
                continue
            dwell = self._dwell.get(button.key, 0.0) + delta_time
            self._dwell[button.key] = dwell
            grabbed = any(cursor.grabbed for cursor in on_button)
            if dwell >= button.dwell:
                self._activate(button, "hand_dwell")
            elif grabbed and dwell >= GRAB_MIN_HOVER and button.dwell <= DWELL_TIME:
                self._activate(button, "hand_grab")
        self._hovered = hovered
        for cursor in cursors:
            draw_hand_cursor(self.canvas, cursor, cursor_colors.get(cursor.side, OPERATING_HAND))
        # Redraw hovered buttons now so their fill / progress reflects this frame.
        for button in self._buttons:
            if button.key in hovered:
                self._draw_button(button)

    def _on_mouse_click(self, event: tk.Event) -> None:
        self._touch_input()
        if self.screen == ATTRACT:
            self._go(SELECT)
            return
        if self._pending is not None:
            return
        for button in reversed(self._buttons):
            if button.contains(event.x, event.y):
                self._activate(button, "mouse")
                return

    # ------------------------------------------------------------------ avatar style
    def _avatar_style(self) -> AvatarStyle:
        if self.screen == ATTRACT:
            return AvatarStyle(brightness=0.4)
        if self.screen in (SELECT, DIFFICULTY, RANKING, RESULT):
            # Hands (the cursors) stay bright, the rest of the body steps back.
            body = 0.25 if self.screen in (RANKING, RESULT) else 0.55
            style = AvatarStyle(brightness=body)
            for part in ("left_hand", "right_hand", "left_forearm", "right_forearm"):
                style.part_brightness[part] = 1.0
            return style
        if self.screen == CALIBRATION:
            return AvatarStyle(brightness=1.0)
        if self.screen == COUNTDOWN:
            style = self.game.avatar_style() if self.game else AvatarStyle()
            style.brightness = min(style.brightness, 0.75)
            return style
        if self.screen == RECOVERY:
            return AvatarStyle(brightness=0.25, frozen=True)
        if self.screen == PLAY and self.game is not None:
            return self.game.avatar_style()
        return AvatarStyle()

    # ------------------------------------------------------------------ drawing helpers
    def _size(self) -> tuple[int, int]:
        return max(self.canvas.winfo_width(), 320), max(self.canvas.winfo_height(), 180)

    def _px(self, value: float) -> float:
        return value * ui_scale(self.canvas)

    def _text(self, x, y, text, size, color=TEXT, mono=False, bold=True, anchor="center", tags="panel") -> None:
        self.canvas.create_text(x, y, text=text, fill=color, font=ui_font(self.canvas, size, bold=bold, mono=mono), anchor=anchor, tags=tags)

    def _button(self, key: str, center: tuple[float, float], icon: str, accent: str, command: Callable[[], None],
                scale: float = 1.0, dwell: float = DWELL_TIME, label: str = "", selected: bool = False) -> UiButton:
        width, height = self._size()
        radius = max(self._px(56), min(self._px(130), self.anchor.button_radius(scale)))
        margin = self._px(12)
        # Keep the whole button on screen.
        x = max(radius + margin, min(width - radius - margin, center[0]))
        y = max(radius + margin, min(height - radius - margin - (self._px(40) if label else 0), center[1]))
        button = UiButton(key, (x, y), radius, icon, accent, command, dwell, label, selected)
        self._buttons.append(button)
        return button

    def _draw_button(self, button: UiButton) -> None:
        canvas = self.canvas
        x, y = button.center
        hovered = button.key in self._hovered
        flashing = self._pending is not None and self._pending[0] == button.key
        radius = button.radius * (1.08 if hovered else 1.0)
        if flashing:
            fill, icon_color = button.accent, "#111111"
        elif hovered or button.selected:
            fill, icon_color = dim(button.accent, 0.22), "#FFFFFF"
        else:
            fill, icon_color = "#0E1116", button.accent
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline=button.accent, width=7 if hovered else 4, tags="screen")
        draw_icon(canvas, button.icon, x, y, radius * 0.55, icon_color, "screen")
        progress = min(self._dwell.get(button.key, 0.0) / button.dwell, 1.0)
        if progress > 0:
            ring = radius + self._px(14)
            canvas.create_arc(x - ring, y - ring, x + ring, y + ring, start=90, extent=-359.9 * progress, style=tk.ARC,
                              outline=button.accent, width=max(4, int(self._px(12))), tags="screen")
        if button.label:
            self._text(x, y + radius + self._px(34), button.label, 34, button.accent, tags="screen")

    def _draw_screen(self, now: float) -> None:
        drawer = {
            ATTRACT: self._draw_attract,
            SELECT: self._draw_select,
            DIFFICULTY: self._draw_difficulty,
            RANKING: self._draw_ranking,
            CALIBRATION: self._draw_calibration,
            COUNTDOWN: self._draw_countdown,
            PLAY: self._draw_play,
            RECOVERY: self._draw_recovery,
            RESULT: self._draw_result,
        }[self.screen]
        drawer(now)
        for button in self._buttons:
            self._draw_button(button)

    def _progress_ring(self, x: float, y: float, radius: float, progress: float, color: str, width: float = 12) -> None:
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline="#1F2937", width=max(3, int(self._px(width))), tags="panel")
        if progress > 0:
            self.canvas.create_arc(x - radius, y - radius, x + radius, y + radius, start=90, extent=-359.9 * min(progress, 1.0),
                                   style=tk.ARC, outline=color, width=max(3, int(self._px(width))), tags="panel")

    def _back_button(self, command: Callable[[], None]) -> None:
        # Level with the shoulder and fully stretched: far from hands resting at the sides.
        self._button("back", self.anchor.side_point(-1, -5, 1.05), "back", "#94A3B8", command, scale=0.75)

    def _draw_silhouette(self, color: str) -> None:
        """Outline of a standing person showing where to stand (no text)."""
        viewport = canvas_viewport(self.canvas)
        head_x, head_y = normalized_point(0.5, 0.30, viewport)
        unit = viewport[3] * 0.06
        width = max(3, int(self._px(5)))
        self.canvas.create_oval(head_x - unit, head_y - unit, head_x + unit, head_y + unit, outline=color, width=width, tags="panel")
        torso = [head_x - unit * 1.5, head_y + unit * 1.6, head_x + unit * 1.5, head_y + unit * 1.6,
                 head_x + unit * 1.1, head_y + unit * 5.2, head_x - unit * 1.1, head_y + unit * 5.2]
        self.canvas.create_polygon(torso, outline=color, fill="", width=width, smooth=True, tags="panel")
        for side in (-1, 1):
            self.canvas.create_line(head_x + side * unit * 1.9, head_y + unit * 2.0, head_x + side * unit * 2.6, head_y + unit * 5.0,
                                    fill=color, width=width, capstyle=tk.ROUND, tags="panel")
            self.canvas.create_line(head_x + side * unit * 0.6, head_y + unit * 5.8, head_x + side * unit * 0.9, head_y + unit * 10.2,
                                    fill=color, width=width, capstyle=tk.ROUND, tags="panel")

    # ------------------------------------------------------------------ screens
    def _draw_attract(self, now: float) -> None:
        width, height = self._size()
        self._text(width / 2, height * 0.1, "モーション ゲーム", 96)
        has_pose = self._stable_flag("attract_pose", bool(self.snapshot.poses), now, delay=0.4)
        if not has_pose:
            self._draw_silhouette(dim(WARNING, 0.8))
        elif self._attract_since is not None:
            progress = (now - self._attract_since) / ATTRACT_STABLE
            self._progress_ring(width / 2, height * 0.9, self._px(34), progress, SUCCESS)

    def _draw_select(self, now: float) -> None:
        for side, (game_id, info) in zip((-1, 1), GAMES.items()):
            self._button(game_id, self.anchor.side_point(side, 35), GAME_ICON[game_id], info.accent,
                         lambda value=game_id: self._choose_game(value))
        self._button("ranking", self.anchor.side_point(1, -5, 1.05), "trophy", ACCENT, self._open_ranking, scale=0.75)
        # Exit is for staff: fixed in the top-right corner with a long hold, out of normal reach.
        width, _height = self._size()
        self._button("exit", (width, 0), "exit", FAILURE, self.close, scale=0.55, dwell=EXIT_DWELL_TIME)

    def _open_ranking(self) -> None:
        self.ranking_game = self.game_id
        self._go(RANKING)

    def _draw_difficulty(self, now: float) -> None:
        width, _height = self._size()
        info = self._info()
        draw_icon(self.canvas, GAME_ICON[self.game_id], width / 2, self._px(70), self._px(46), info.accent, "panel")
        # 2x2 around the shoulders (above the head is often off screen or behind the head).
        positions = (self.anchor.side_point(-1, 40), self.anchor.side_point(1, 40), self.anchor.side_point(1, -5, 1.05))
        for difficulty, center in zip(DIFFICULTIES, positions):
            self._button(difficulty, center, DIFFICULTY_ICON[difficulty], DIFFICULTY_COLOR[difficulty],
                         lambda value=difficulty: self._choose_difficulty(value), scale=0.9)
        self._back_button(lambda: self._go(SELECT))

    def _draw_ranking(self, now: float) -> None:
        width, height = self._size()
        info = GAMES[self.ranking_game]
        panel_width = self._px(440)
        top = self._px(40)
        row = self._px(76)
        self.canvas.create_rectangle(width / 2 - panel_width / 2, top, width / 2 + panel_width / 2, top + row * 6.3,
                                     fill="#0B0F14", outline="#1F2937", width=2, tags="panel")
        draw_icon(self.canvas, GAME_ICON[self.ranking_game], width / 2 - self._px(70), top + row * 0.6, self._px(30), info.accent, "panel")
        draw_icon(self.canvas, DIFFICULTY_ICON[self.ranking_difficulty], width / 2 + self._px(70), top + row * 0.6, self._px(34),
                  DIFFICULTY_COLOR[self.ranking_difficulty], "panel")
        entries = load_rankings(self.ranking_game, self.ranking_difficulty)[:5]
        for index in range(5):
            y = top + row * (1.5 + index)
            color = "#FBBF24" if index == 0 else TEXT
            if index == 0:
                draw_icon(self.canvas, "crown", width / 2 - panel_width / 2 + self._px(60), y, self._px(22), color, "panel")
            else:
                self._text(width / 2 - panel_width / 2 + self._px(60), y, str(index + 1), 40, TEXT_SUB, mono=True)
            score = f"{entries[index].score:,}" if index < len(entries) else "—"
            self._text(width / 2 + panel_width / 2 - self._px(40), y, score, 48, color if index < len(entries) else "#334155", mono=True, anchor="e")
        # Toggle buttons: tap to switch game / difficulty (fewer buttons to reach).
        other_game = next(game_id for game_id in GAMES if game_id != self.ranking_game)
        next_difficulty = DIFFICULTIES[(DIFFICULTIES.index(self.ranking_difficulty) + 1) % len(DIFFICULTIES)]
        # Buttons at full reach, level with the shoulders, so the score panel never covers them.
        self._button("ranking_game", self.anchor.side_point(-1, 0, 1.1), GAME_ICON[other_game], GAMES[other_game].accent,
                     lambda: self._set_ranking(game=other_game), scale=0.8)
        self._button("ranking_difficulty", self.anchor.side_point(1, 0, 1.1), DIFFICULTY_ICON[next_difficulty], DIFFICULTY_COLOR[next_difficulty],
                     lambda: self._set_ranking(difficulty=next_difficulty), scale=0.8)
        self._button("back", self.anchor.side_point(-1, -45, 1.05), "back", "#94A3B8", lambda: self._go(SELECT), scale=0.7)

    def _set_ranking(self, game: str | None = None, difficulty: str | None = None) -> None:
        if game is not None:
            self.ranking_game = game
        if difficulty is not None:
            self.ranking_difficulty = difficulty
        self._dwell.clear()
        # Stay on the same screen; allow the next choice after a short pause.
        self._screen_since = time.monotonic() - SCREEN_INPUT_DELAY / 2

    def _draw_calibration(self, now: float) -> None:
        width, height = self._size()
        info = self._info()
        viewport = canvas_viewport(self.canvas)
        guide_left, guide_top = normalized_point(0.08, 0.08, viewport)
        guide_right, guide_bottom = normalized_point(0.92, 0.95, viewport)
        # Display-only debounce so flaky detection never makes the indicators flicker.
        checks = [self._stable_flag(f"check:{check.label}", check.ok, now) for check in self._calibration_checks()]
        all_ok = all(checks)
        self.canvas.create_rectangle(guide_left, guide_top, guide_right, guide_bottom, outline=SUCCESS if all_ok else "#475569", width=3, tags="panel")
        # One dot per required body part (green = seen).
        spacing = self._px(46)
        start_x = width / 2 - spacing * (len(checks) - 1) / 2
        dot = self._px(12)
        for index, ok in enumerate(checks):
            x = start_x + index * spacing
            y = guide_top + self._px(34)
            self.canvas.create_oval(x - dot, y - dot, x + dot, y + dot, fill=SUCCESS if ok else "#334155", outline="", tags="panel")
        # What the game uses: part colours (game A) or hand shapes (game B), icons only.
        samples = info.calibration_samples(self.difficulty)
        sample_spacing = self._px(120)
        sample_x = width / 2 - sample_spacing * (len(samples) - 1) / 2
        for index, (symbol, _label, color) in enumerate(samples):
            draw_marker(self.canvas, sample_x + index * sample_spacing, guide_top + self._px(100), self._px(24), symbol, color, "panel")
        if self._calibration_ok_since is not None:
            progress = (now - self._calibration_ok_since) / (CALIBRATION_STABLE + CALIBRATION_CANCEL_GRACE)
            self._progress_ring(width / 2, height * 0.5, self._px(70), progress, SUCCESS, width=16)
        self._back_button(lambda: self._go(DIFFICULTY))

    def _draw_countdown(self, now: float) -> None:
        width, height = self._size()
        self._draw_game_layer()
        value = 3 - int((now - self._countdown_since) / COUNTDOWN_STEP)
        self._text(width / 2, height / 2, str(max(value, 1)), 220, mono=True)

    def _draw_play(self, now: float) -> None:
        width, height = self._size()
        self._draw_game_layer()
        if now < self._start_banner_until:
            # A quick expanding ring marks the start instead of a word.
            progress = 1.0 - (self._start_banner_until - now) / START_BANNER
            radius = self._px(80) + progress * height * 0.4
            self.canvas.create_oval(width / 2 - radius, height / 2 - radius, width / 2 + radius, height / 2 + radius,
                                    outline=dim(SUCCESS, 1.0 - progress), width=max(3, int(self._px(18) * (1 - progress))), tags="panel")

    def _draw_recovery(self, now: float) -> None:
        width, height = self._size()
        self._draw_game_layer()
        size = self._px(150)
        self.canvas.create_oval(width / 2 - size, height / 2 - size, width / 2 + size, height / 2 + size, fill="#0B0B0B", outline="", tags="panel")
        if self._recovery_ok_since is not None and now - self._recovery_ok_since >= RECOVERY_STABLE:
            value = 3 - int((now - self._recovery_ok_since - RECOVERY_STABLE) / RESUME_STEP)
            self._text(width / 2, height / 2, str(max(value, 1)), 140, SUCCESS, mono=True)
            return
        draw_icon(self.canvas, "pause", width / 2, height / 2, self._px(70), WARNING, "panel")
        remaining = max(0.0, 1.0 - (now - self._recovery_since) / RECOVERY_TIMEOUT)
        self._progress_ring(width / 2, height / 2, size * 0.85, remaining, WARNING, width=10)

    def _draw_game_layer(self) -> None:
        if self.game is None:
            return
        try:
            self.game.draw(self.canvas)
        except Exception:
            self._game_crashed("draw")

    def _draw_result(self, now: float) -> None:
        width, height = self._size()
        result = self.last_result or GameResult(0)
        info = self._info()
        draw_icon(self.canvas, GAME_ICON[self.game_id], width / 2 - self._px(70), self._px(60), self._px(34), info.accent, "panel")
        draw_icon(self.canvas, DIFFICULTY_ICON[self.difficulty], width / 2 + self._px(70), self._px(60), self._px(36), DIFFICULTY_COLOR[self.difficulty], "panel")
        self._text(width / 2, self._px(190), f"{result.score:,}", 150, mono=True)
        outcome = self.last_outcome
        if outcome is not None and outcome.rank is not None and outcome.rank <= 5:
            draw_icon(self.canvas, "crown", width / 2 - self._px(60), self._px(305), self._px(30), "#FBBF24", "panel")
            self._text(width / 2 + self._px(30), self._px(305), str(outcome.rank), 60, "#FBBF24", mono=True)
        if result.reason:
            self._text(width / 2, self._px(370), result.reason, 32, FAILURE)
        slot_width = self._px(300)
        metrics = list(result.metrics.items())[:3]
        start_x = width / 2 - slot_width * (len(metrics) - 1) / 2
        for index, (label, value) in enumerate(metrics):
            x = start_x + index * slot_width
            self._text(x, height - self._px(120), value, 48, mono=True)
            self._text(x, height - self._px(76), label, 26, TEXT_SUB)
        self._button("retry", self.anchor.side_point(-1, 30), "retry", SUCCESS, lambda: self._go(CALIBRATION))
        self._button("select", self.anchor.side_point(1, 30), "menu", ACCENT, lambda: self._go(SELECT))
        idle = now - self._last_input
        if idle >= RESULT_COUNTDOWN_FROM:
            # A shrinking bar shows the automatic return, without words.
            remaining = max(0.0, (IDLE_TIMEOUT[RESULT] - idle) / (IDLE_TIMEOUT[RESULT] - RESULT_COUNTDOWN_FROM))
            bar = width * 0.4
            self.canvas.create_rectangle(width / 2 - bar / 2, height - self._px(22), width / 2 - bar / 2 + bar * remaining, height - self._px(14),
                                         fill=TEXT_SUB, outline="", tags="panel")

    # ------------------------------------------------------------------ overlays
    def _draw_overlays(self, now: float) -> None:
        width, height = self._size()
        if self.detector.has_error:
            title, action, code = ERROR_MESSAGES.get(self.detector.status, ("システム エラー", "スタッフを よんでね", "E04"))
            self.canvas.create_rectangle(width * 0.22, height * 0.36, width * 0.78, height * 0.64, fill="#111111", outline=FAILURE, width=4, tags="overlay")
            self._text(width / 2, height * 0.43, title, 56, FAILURE, tags="overlay")
            self._text(width / 2, height * 0.52, action, 36, TEXT, tags="overlay")
            self._text(width / 2, height * 0.59, code, 24, TEXT_SUB, mono=True, tags="overlay")
        if self._system_message is not None:
            message, until = self._system_message
            if now < until:
                self._text(width / 2, height * 0.24, message, 40, FAILURE, tags="overlay")
            else:
                self._system_message = None
        if now < self._ranking_reset_armed_until:
            self._text(width / 2, height * 0.06, "もういちど おすと ランキングを リセット します", 30, WARNING, tags="overlay")
        # Small input state indicator instead of a permanent error string.
        if self.detector.has_error:
            color = FAILURE
        elif self.detector.status != STATUS_READY:
            color = TEXT_SUB
        elif self.snapshot.poses:
            color = SUCCESS
        else:
            color = "#475569"
        radius = self._px(9)
        x, y = width - self._px(24), height - self._px(22)
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="", tags="overlay")
        if self._show_diagnostics:
            self._draw_diagnostics()

    def _draw_diagnostics(self) -> None:
        fps = len(self._frame_times)
        lines = [
            f"FPS {fps}   screen {self.screen}   game {self.game_id}/{self.difficulty}",
            f"detector {self.detector.status} {self.detector.error_detail}",
            f"poses {len(self.snapshot.poses)}   hands {len(self.snapshot.hands)} {self.snapshot.hand_sides}",
            f"hovered {sorted(self._hovered)}   canvas items {len(self.canvas.find_all())}",
            "F5 reset  F11 fullscreen  Ctrl+Shift+R ranking reset  BackSpace abort  Enter force start",
        ]
        for index, line in enumerate(lines):
            self.canvas.create_text(12, 12 + index * 18, text=line, anchor="nw", fill="#A3E635", font=("Consolas", 10), tags="overlay")

    def _stable_flag(self, key: str, value: bool, now: float, delay: float = 0.3) -> bool:
        """Return ``value`` only after it has held for ``delay`` seconds (display debounce)."""
        shown, pending, since = self._stable_flags.get(key, (value, value, now))
        if value != pending:
            pending, since = value, now
        if pending != shown and now - since >= delay:
            shown = pending
        self._stable_flags[key] = (shown, pending, since)
        return shown

    def _report_status(self) -> None:
        status = f"{self.detector.status}:{self.detector.error_detail}"
        if status != self._last_status:
            self._last_status = status
            log_event("detector_status", status=self.detector.status, detail=self.detector.error_detail)

    # ------------------------------------------------------------------ operator keys
    def _toggle_fullscreen(self, _event: tk.Event) -> None:
        self.root.attributes("-fullscreen", not bool(self.root.attributes("-fullscreen")))

    def _toggle_diagnostics(self, _event: tk.Event) -> None:
        self._show_diagnostics = not self._show_diagnostics

    def _operator_reset(self, _event: tk.Event) -> None:
        log_event("operator_reset", screen=self.screen)
        self._discard_game()
        if self.detector.has_error or self.detector.status != STATUS_READY:
            self.detector.start()
        self._go(ATTRACT)

    def _operator_abort(self, _event: tk.Event) -> None:
        if self.screen in (COUNTDOWN, PLAY, RECOVERY):
            log_event("operator_abort", game=self.game_id)
            self._discard_game()
            self._go(SELECT)

    def _operator_force_start(self, _event: tk.Event) -> None:
        if self.screen == CALIBRATION:
            log_event("operator_force_start")
            self._begin_countdown()

    def _ranking_reset(self, _event: tk.Event) -> None:
        now = time.monotonic()
        if now < self._ranking_reset_armed_until:
            reset_rankings()
            log_event("ranking_reset")
            self._ranking_reset_armed_until = 0.0
            self._system_message = ("ランキングを リセット しました", now + 3.0)
        else:
            self._ranking_reset_armed_until = now + 3.0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        log_event("app_close")
        if self._tick_job is not None:
            try:
                self.root.after_cancel(self._tick_job)
            except tk.TclError:
                pass
        self._discard_game()
        self.detector.close()
        self.root.destroy()


def run_launcher() -> None:
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_launcher()
