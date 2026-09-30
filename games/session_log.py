"""Operator log for exhibition runs (screen changes, pointer decisions, tracking loss, results)."""
from __future__ import annotations

import logging
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

_logger = logging.getLogger("motion_game")


def _configure() -> None:
    if _logger.handlers:
        return
    _logger.setLevel(logging.INFO)
    try:
        LOG_DIR.mkdir(exist_ok=True)
        handler: logging.Handler = logging.FileHandler(LOG_DIR / "session.log", encoding="utf-8")
    except OSError:
        handler = logging.NullHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _logger.addHandler(handler)


def log_event(event: str, **fields) -> None:
    _configure()
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    _logger.info(f"{event} {details}".rstrip())


def log_exception(message: str) -> None:
    _configure()
    _logger.exception(message)
