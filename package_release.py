"""Build the release zip: only what is needed to play (code, models, launch scripts).

Usage: python package_release.py <version>   ->   dist/media-pipe2-<version>.zip
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DIST_DIR = ROOT_DIR / "dist"

FILES = ("main.py", "README.md", "requirements.txt", "setup.bat", "run.bat")
PACKAGES = ("games", "tracking")
# The detector uses the first pose model it finds, so the lite one is enough.
MODELS = ("models/hand/hand_landmarker.task", "models/pose/pose_landmarker_lite.task")
EMPTY_RANKING = "games/ranking.json"


def release_files() -> list[Path]:
    paths = [ROOT_DIR / name for name in FILES + MODELS]
    for package in PACKAGES:
        paths.extend(sorted((ROOT_DIR / package).rglob("*.py")))
    missing = [str(path.relative_to(ROOT_DIR)) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing files: {', '.join(missing)}")
    return paths


def build(version: str) -> Path:
    name = f"media-pipe2-{version}"
    DIST_DIR.mkdir(exist_ok=True)
    target = DIST_DIR / f"{name}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in release_files():
            archive.write(path, f"{name}/{path.relative_to(ROOT_DIR).as_posix()}")
        # Ship an empty ranking instead of the local play records.
        archive.writestr(f"{name}/{EMPTY_RANKING}", "[]")
    return target


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = build(sys.argv[1])
    print(f"{target} ({target.stat().st_size / 1_000_000:0.1f} MB)")


if __name__ == "__main__":
    main()
