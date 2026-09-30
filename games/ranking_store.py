from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parent.parent
RANKING_FILE = ROOT_DIR / "games" / "ranking.json"


@dataclass
class RankingEntry:
    game_id: str
    difficulty: str
    score: int
    timestamp: str


@dataclass(frozen=True)
class RankingOutcome:
    rank: int | None
    total: int


def load_rankings(game_id: str | None = None, difficulty: str | None = None) -> list[RankingEntry]:
    if not RANKING_FILE.exists():
        return []

    try:
        data = json.loads(RANKING_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    entries: list[RankingEntry] = []
    for item in data:
        try:
            entries.append(
                RankingEntry(
                    game_id=str(item["game_id"]),
                    difficulty=str(item.get("difficulty", "unknown")),
                    score=int(item["score"]),
                    timestamp=str(item["timestamp"]),
                )
            )
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
    entries.sort(key=lambda entry: entry.score, reverse=True)
    return [
        entry
        for entry in entries
        if (game_id is None or entry.game_id == game_id)
        and (difficulty is None or entry.difficulty == difficulty)
    ]


def save_rankings(entries: Iterable[RankingEntry]) -> None:
    payload = [asdict(entry) for entry in entries]
    temporary = RANKING_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, RANKING_FILE)


def add_score(game_id: str, difficulty: str, score: int, keep_top: int = 20) -> RankingOutcome:
    entries = load_rankings()
    new_entry = RankingEntry(
        game_id=game_id,
        difficulty=difficulty,
        score=int(score),
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    entries.append(new_entry)

    entries.sort(key=lambda entry: entry.score, reverse=True)
    counts: dict[tuple[str, str], int] = {}
    trimmed = []
    for entry in entries:
        key = (entry.game_id, entry.difficulty)
        if counts.get(key, 0) >= keep_top:
            continue
        trimmed.append(entry)
        counts[key] = counts.get(key, 0) + 1
    save_rankings(trimmed)
    category = [entry for entry in trimmed if entry.game_id == game_id and entry.difficulty == difficulty]
    rank = next((index + 1 for index, entry in enumerate(category) if entry is new_entry), None)
    return RankingOutcome(rank, len(category))


def reset_rankings() -> None:
    save_rankings([])
