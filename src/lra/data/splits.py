"""Stratified item selections. `pilot` is nested inside `main` so that pilot
traces can be folded into the main analysis instead of being thrown away."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from lra.data.schema import Item


def _stratum_key(item: Item) -> tuple:
    if item.task == "strategyqa":
        return (item.label,)
    if item.task == "prontoqa":
        return (item.difficulty["hops"], item.label)
    if item.task == "comparison":
        return (item.meta["attribute"], item.difficulty["ratio_bin"], item.meta["direction"], item.label)
    raise ValueError(item.task)


def _one_order_per_pair(items: list[Item], rng: random.Random) -> list[Item]:
    """Comparison items come in mirrored pairs (A/B swapped). Keep one per pair."""
    by_pair: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_pair[it.meta["pair_id"]].append(it)
    kept = []
    for pair_id in sorted(by_pair):
        kept.append(rng.choice(sorted(by_pair[pair_id], key=lambda x: x.item_id)))
    return kept


def stratified_sample(items: list[Item], n: int, rng: random.Random) -> list[Item]:
    """Draw n items as evenly as possible across strata, without replacement."""
    strata: dict[tuple, list[Item]] = defaultdict(list)
    for it in items:
        strata[_stratum_key(it)].append(it)
    keys = sorted(strata)
    for k in keys:
        rng.shuffle(strata[k])
    chosen: list[Item] = []
    # round-robin over strata so counts differ by at most one
    pointers = {k: 0 for k in keys}
    while len(chosen) < n:
        progressed = False
        for k in keys:
            if len(chosen) >= n:
                break
            if pointers[k] < len(strata[k]):
                chosen.append(strata[k][pointers[k]])
                pointers[k] += 1
                progressed = True
        if not progressed:
            raise ValueError(f"not enough items: wanted {n}, have {len(items)}")
    return chosen


def comparison_sample(items: list[Item], n: int, rng: random.Random) -> list[Item]:
    """Exact attribute x difficulty quotas with balanced label/direction margins.

    Solve integer counts over at most 120 strata, then randomly draw within
    strata. Item-level optimisation is unnecessary. The caller removes mirrors.
    Infeasible quotas fail explicitly rather than dropping a category silently.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    strata: dict[tuple, list[Item]] = defaultdict(list)
    for item in items:
        strata[_stratum_key(item)].append(item)
    keys = sorted(strata)
    attrs = sorted({k[0] for k in keys})
    bins = sorted({k[1] for k in keys})
    if n <= 0 or n % (len(attrs) * len(bins)):
        raise ValueError("comparison size must be a positive multiple of attribute x bin cells")
    per_cell = n // (len(attrs) * len(bins))
    rows, lows, highs = [], [], []

    def constraint(predicate, low, high):
        rows.append([int(predicate(k)) for k in keys])
        lows.append(low)
        highs.append(high)

    def balance(predicate, size):
        for direction in ("larger", "smaller"):
            constraint(lambda k: predicate(k) and k[2] == direction, size // 2, (size + 1) // 2)
        for label in ("A", "B"):
            constraint(lambda k: predicate(k) and k[3] == label, size // 2, (size + 1) // 2)

    for attr in attrs:
        for bin_index in bins:
            cell = lambda k: k[:2] == (attr, bin_index)
            constraint(cell, per_cell, per_cell)
            balance(cell, per_cell)
            for direction in ("larger", "smaller"):
                for label in ("A", "B"):
                    constraint(lambda k: cell(k) and k[2:] == (direction, label),
                               per_cell // 4, (per_cell + 3) // 4)
        balance(lambda k: k[0] == attr, per_cell * len(bins))
    for bin_index in bins:
        balance(lambda k: k[1] == bin_index, per_cell * len(attrs))
    balance(lambda k: True, n)
    for direction in ("larger", "smaller"):
        for label in ("A", "B"):
            constraint(lambda k: k[2:] == (direction, label), n // 4, (n + 3) // 4)
    matrix = np.array(rows, dtype=float)
    result = milp(
        c=np.array([rng.random() for _ in keys]), integrality=np.ones(len(keys)),
        bounds=Bounds(np.zeros(len(keys)), [len(strata[k]) for k in keys]),
        constraints=LinearConstraint(matrix, lows, highs),
        options={"time_limit": 30},
    )
    if not result.success:
        raise ValueError(f"Cannot satisfy comparison split quotas: {result.message}")
    counts = np.rint(result.x).astype(int)
    margins = matrix @ counts
    if np.any(margins < lows) or np.any(margins > highs):
        raise ValueError("Integer split solution violates quotas")
    chosen = []
    for key, count in zip(keys, counts):
        pool = sorted(strata[key], key=lambda i: i.item_id)
        rng.shuffle(pool)
        chosen.extend(pool[:count])
    return chosen


def build_splits(cfg: dict[str, Any], items_by_task: dict[str, list[Item]], out_dir: Path, seed: int) -> dict[str, dict[str, list[str]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    main_sizes = cfg["main"]
    pilot_sizes = cfg["pilot"]
    result: dict[str, dict[str, list[str]]] = {"main": {}, "pilot": {}}
    for task, items in sorted(items_by_task.items()):
        rng = random.Random(f"{seed}:{task}")
        pool = _one_order_per_pair(items, rng) if task == "comparison" else list(items)
        sample = comparison_sample if task == "comparison" else stratified_sample
        main = sample(pool, int(main_sizes[task]), rng)
        pilot = sample(main, int(pilot_sizes[task]), rng)
        result["main"][task] = sorted(i.item_id for i in main)
        result["pilot"][task] = sorted(i.item_id for i in pilot)
    for name in ("main", "pilot"):
        payload = {"seed": seed, "tasks": result[name], "sizes": {t: len(v) for t, v in result[name].items()}}
        (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
