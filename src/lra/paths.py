"""Repository-relative paths. Everything is resolved from the repo root so that
scripts behave the same no matter where they are launched from."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TASKS_DIR = DATA_DIR / "tasks"
PROMPTS_DIR = DATA_DIR / "prompts"
SPLITS_DIR = DATA_DIR / "splits"

DATA_CONFIG = CONFIGS_DIR / "data.yaml"
PROMPTS_CONFIG = CONFIGS_DIR / "prompts.yaml"


def ensure_dirs() -> None:
    for d in (RAW_DIR, TASKS_DIR, PROMPTS_DIR, SPLITS_DIR):
        d.mkdir(parents=True, exist_ok=True)
