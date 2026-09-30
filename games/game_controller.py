from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, field
from typing import Callable, Protocol

from games.common_ui import AvatarStyle, visible
from tracking.hand_pose_detector import DetectionSnapshot


@dataclass(frozen=True)
class GameResult:
    score: int
    # Up to three display metrics, label -> formatted value, shown in the same slots for every game.
    metrics: dict[str, str] = field(default_factory=dict)
    # Why the play ended (shown on the result screen so the player can tell what went wrong).
    reason: str = ""


@dataclass(frozen=True)
class CalibrationCheck:
    label: str
    ok: bool


class GameController(Protocol):
    """Lifecycle used by the launcher host (ideas/README.md 9-13).

    Games never open windows, cameras or event loops; they only own their state
    and the "game" / "hud" canvas tags.
    """

    difficulty: str
    finished: bool

    def enter(self) -> None: ...

    def update(self, snapshot: DetectionSnapshot, delta_time: float) -> None: ...

    def draw(self, canvas: tk.Canvas) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def exit(self) -> None: ...

    def result(self) -> GameResult: ...

    def avatar_style(self) -> AvatarStyle: ...


@dataclass(frozen=True)
class DifficultyInfo:
    key: str
    label: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class GameInfo:
    game_id: str
    title: str
    input_label: str
    tagline: str
    accent: str
    difficulties: tuple[DifficultyInfo, ...]
    factory: Callable[[tk.Canvas, str], GameController]
    calibration_checks: Callable[[DetectionSnapshot, str], list[CalibrationCheck]]
    tracking_ok: Callable[[DetectionSnapshot, str], bool]
    calibration_hint: str
    # One-time explanation on the calibration screen: (marker symbol, label, colour) per item.
    calibration_samples: Callable[[str], tuple[tuple[str, str, str], ...]] = lambda _difficulty: ()


def pose_parts_visible(snapshot: DetectionSnapshot, indices: tuple[int, ...], threshold: float = 0.5) -> bool:
    if not snapshot.poses:
        return False
    pose = snapshot.poses[0]
    if len(pose) <= max(indices):
        return False
    return all(visible(pose[index], threshold) for index in indices)
