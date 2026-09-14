"""Canonical item schema shared by every task.

One `Item` is one two-alternative question. The generation code never needs to
know which task an item came from: it only reads `question`, `options` and
`label`. Task-specific covariates live in `difficulty`; provenance in `meta`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

TASKS: tuple[str, ...] = ("strategyqa", "prontoqa", "comparison")


@dataclass
class Item:
    item_id: str
    task: str
    question: str
    options: list[str]
    label: str
    difficulty: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"{self.item_id}: unknown task {self.task!r}")
        if not self.item_id.startswith(f"{self.task}:"):
            raise ValueError(f"{self.item_id}: item_id must start with '{self.task}:'")
        if len(self.options) != 2 or len(set(self.options)) != 2:
            raise ValueError(f"{self.item_id}: need exactly two distinct options, got {self.options}")
        if self.label not in self.options:
            raise ValueError(f"{self.item_id}: label {self.label!r} not in options {self.options}")
        if not self.question.strip():
            raise ValueError(f"{self.item_id}: empty question")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Item":
        return cls(**d)


def stable_hash(obj: Any, n: int = 12) -> str:
    """Deterministic short hash of any JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_items(path: Path, items: Iterable[Item]) -> int:
    rows = []
    seen: set[str] = set()
    for it in items:
        it.validate()
        if it.item_id in seen:
            raise ValueError(f"duplicate item_id {it.item_id}")
        seen.add(it.item_id)
        rows.append(it.to_dict())
    return write_jsonl(path, rows)


def read_items(path: Path) -> list[Item]:
    return [Item.from_dict(r) for r in read_jsonl(path)]
