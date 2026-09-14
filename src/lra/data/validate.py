"""Validate the current frozen datasets, source provenance, prompts and splits.

Run after building: python -m lra.data.validate
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from collections import Counter
from pathlib import Path

from lra.data import build, comparison, prompts, prontoqa
from lra.data.schema import TASKS, read_items, read_jsonl
from lra.paths import DATA_DIR, RAW_DIR, SPLITS_DIR, TASKS_DIR, PROMPTS_DIR


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def literal(phrase: str) -> tuple[str, bool]:
    phrase = phrase.lower()
    negative = phrase.startswith("not ")
    if negative:
        phrase = phrase[4:]
    phrase = re.sub(r"^an? ", "", phrase)
    # All fictional nouns in the selected original release use pus/puses.
    if phrase.endswith("puses"):
        phrase = phrase[:-2]
    require(re.fullmatch(r"[a-z]+", phrase) is not None, f"Unsupported predicate: {phrase}")
    return phrase, negative


def validate_proof(item) -> None:
    """Independently solve the controlled English, without using the gold proof."""
    context, query = item.question.split(" True or false: ")
    rules, facts = [], []
    for sentence in context.removesuffix(".").split(". "):
        rule = (re.fullmatch(r"(?:Each|Every) ([a-z]+) is (.+)", sentence)
                or re.fullmatch(r"([A-Z][a-z]+) are (.+)", sentence))
        if rule:
            rules.append(tuple(literal(p) for p in rule.groups()))
        else:
            fact = re.fullmatch(r"([A-Z][a-z]+) is (.+)", sentence)
            require(fact is not None, f"Unsupported sentence: {sentence}")
            facts.append((fact[1], literal(fact[2])))
    target = re.fullmatch(r"([A-Z][a-z]+) is (.+)\.", query)
    require(target is not None and len(facts) == 1 and facts[0][0] == target[1], item.item_id)
    distances = {facts[0][1]: 0}
    changed = True
    while changed:
        changed = False
        for left, right in rules:
            if left in distances and distances.get(right, math.inf) > distances[left] + 1:
                distances[right] = distances[left] + 1
                changed = True
    goal = literal(target[2])
    opposite = goal[0], not goal[1]
    require((goal in distances) != (opposite in distances), f"Undecidable query: {item.item_id}")
    require(not any((p, not n) in distances for p, n in distances), f"Contradiction: {item.item_id}")
    answer = "True" if goal in distances else "False"
    conclusion = goal if answer == "True" else opposite
    require(answer == item.label and distances[conclusion] == item.difficulty["hops"], f"Invalid proof: {item.item_id}")


def main() -> int:
    cfg, pcfg = build.load_data_config(), prompts.load_prompt_config()
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        require(hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() == expected, f"Changed artifact: {name}")
    items = {task: read_items(TASKS_DIR / f"{task}.jsonl") for task in TASKS}
    for task, rows in items.items():
        require(manifest["task_builds"][task]["dependencies"] == build.task_dependencies(task, cfg, pcfg), f"Stale inputs: {task}")
        require(len({r.item_id for r in rows}) == len(rows), f"Duplicate IDs: {task}")
        for row in rows:
            row.validate()
        actual = list(read_jsonl(PROMPTS_DIR / f"{task}.jsonl"))
        require(actual == list(prompts.render_rows(pcfg, rows)), f"Prompt mismatch: {task}")
    with tempfile.TemporaryDirectory() as temporary:
        original = prontoqa.build(cfg["prontoqa"], Path(temporary))
        require(original == items["prontoqa"], "PrOntoQA differs from the pinned source")
    for item in items["prontoqa"]:
        validate_proof(item)

    ccfg = cfg["comparison"]
    resolved = {attr: {r["qid"]: r for r in comparison.load_or_fetch_entities(attr, acfg, ccfg, RAW_DIR / "comparison")}
                for attr, acfg in ccfg["attributes"].items()}
    comparison_map = {i.item_id: i for i in items["comparison"]}
    for item in items["comparison"]:
        a, b = item.meta["entity_a"], item.meta["entity_b"]
        attr = item.meta["attribute"]
        require(a == resolved[attr][a["qid"]] and b == resolved[attr][b["qid"]], f"Source mismatch: {item.item_id}")
        require(comparison.pair_is_eligible(a, b, ccfg["attributes"][attr].get("population_policy")), f"Ambiguous pair: {item.item_id}")
        expected = "A" if a["value"] > b["value"] else "B"
        if item.meta["direction"] == "smaller":
            expected = "B" if expected == "A" else "A"
        require(item.label == expected, f"Wrong label: {item.item_id}")
        mirror = comparison_map[item.meta["mirror_of"]]
        require(mirror.meta["mirror_of"] == item.item_id and mirror.label != item.label
                and mirror.meta["entity_a"] == b and mirror.meta["entity_b"] == a, f"Invalid mirror: {item.item_id}")
        ratio = abs(math.log10(a["value"] / b["value"]))
        bin_index = item.difficulty["ratio_bin"]
        low, high = ccfg["ratio_bins"][bin_index]
        require(low <= ratio < high or (bin_index == len(ccfg["ratio_bins"]) - 1 and ratio == high), f"Wrong bin: {item.item_id}")
        require(math.isclose(ratio, item.difficulty["abs_log10_ratio"]), f"Wrong ratio: {item.item_id}")

    selected = {name: json.loads((SPLITS_DIR / f"{name}.json").read_text())["tasks"] for name in ("main", "pilot")}
    for task in TASKS:
        require(set(selected["pilot"][task]) <= set(selected["main"][task]), f"Pilot not nested: {task}")
        mapping = {i.item_id: i for i in items[task]}
        for name in selected:
            ids = selected[name][task]
            n = cfg["splits"][name][task]
            require(len(ids) == len(set(ids)) == n and set(ids) <= set(mapping), f"Invalid split: {name}/{task}")
            rows = [mapping[i] for i in ids]
            require(set(Counter(i.label for i in rows).values()) == {n // 2}, f"Unbalanced labels: {name}/{task}")
            if task == "prontoqa":
                cells = Counter((i.difficulty["hops"], i.label) for i in rows)
                require(len(cells) == 6 and set(cells.values()) == {n // 6}, f"Unbalanced hops: {name}")
            if task == "comparison":
                cells = Counter((i.meta["attribute"], i.difficulty["ratio_bin"]) for i in rows)
                require(len(cells) == 30 and set(cells.values()) == {n // 30}, f"Unbalanced comparison cells: {name}")
                require(len({i.meta["pair_id"] for i in rows}) == n, f"Repeated pair: {name}")
                require(Counter(i.meta["direction"] for i in rows) == {"larger": n // 2, "smaller": n // 2}, f"Unbalanced directions: {name}")
    print("Validated sources, labels, PrOntoQA proofs, prompts, splits and manifest:")
    for task in TASKS:
        print(f"  {task}: {len(items[task])} items, {len(items[task]) * len(pcfg['conditions'])} prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
