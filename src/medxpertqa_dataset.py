"""MedXpertQA two-option diagnosis task: frozen source -> data/tasks, data/prompts, data/selections.

    python src/medxpertqa_dataset.py fetch     # download the source files at the pinned revision (network)
    python src/medxpertqa_dataset.py build     # source -> data/tasks, data/prompts, data/verification
    python src/medxpertqa_dataset.py verify    # check the files against a rebuild and the source hashes
    python src/medxpertqa_dataset.py select    # screening selection: every item, baseline condition

Source: MedXpertQA (Zuo et al., ICML 2025; arXiv:2501.18362), Hugging Face dataset
TsinghuaC3I/MedXpertQA, config "Text", split "test": 2,450 exam questions with 10
options each, rewritten by the authors to limit data leakage; MIT licence.

Questions kept: medical_task == "Diagnosis" and the last sentence of the stem (the
text before the "Answer Choices:" line) contains "most likely diagnosis".

Items: the correct diagnosis is paired with each of the question's 9 distractors and
each pair is asked in both orders (ab: correct option first, ba: distractor first),
so a source question gives 9 pairs and 18 items. The question text is the stem
verbatim followed by the two options as "A. ..." / "B. ...", the form of the
comparison task. Reducing the option count has precedent in Wang et al.
(arXiv:2402.01349 v1/v2: MMLU and MedMCQA reduced to 2 options by removing
distractors) and in the dataset authors' own option augmentation (4-5 -> 10).
Instruction and answer lines are the comparison task's; see docs/PROMPT_SOURCES.md.

Difficulty is not a property of the source: data/pairs/ (src/pair_distance.py) gives
every pair a distance from screening accuracy, the model's 10-option prior and ICD-10.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "medxpertqa"
MANIFEST_PATH = RAW_DIR / "source_manifest.json"
TASKS_PATH = ROOT / "data" / "tasks" / "medxpertqa.jsonl"
PROMPTS_PATH = ROOT / "data" / "prompts" / "medxpertqa.jsonl"
REPORT_PATH = ROOT / "data" / "verification" / "medxpertqa.jsonl"
SELECTIONS_DIR = ROOT / "data" / "selections"

TASK = "medxpertqa"
DATA_VERSION = "v1"
HF_REPO = "TsinghuaC3I/MedXpertQA"
HF_REVISION = "7e7c465a68eb2b866926bfa59c8c9d17a8daba65"
HF_LICENSE = "MIT (dataset card)"
# Frozen source files (path in the HF repo -> sha256). `fetch` refuses to overwrite with different bytes.
RAW_FILES = {
    "Text/test.jsonl": "07ea977f7e326909b51c883b65f6ad7f399fc6db07408ef3630f4e3048ec71b2",
    "Text/dev.jsonl": "1030a180a00ea2762ce9ac2a4fa4d5db4543b7e4aa25b0fdd2ec44149e2242fa",
}
SOURCE_FILE = "Text/test.jsonl"
SOURCE_LETTERS = "ABCDEFGHIJ"
OPTIONS = ["A", "B"]
CHOICES_MARKER = "Answer Choices:"
KEEP_TASK = "Diagnosis"
KEEP_PHRASE = "most likely diagnosis"

# Conditions: an instruction sentence placed between the question and the answer line, and/or a
# suffix appended to the user message. Wording and sources in docs/PROMPT_SOURCES.md.
CONDITIONS = {
    "C0_baseline": {"instruction": "", "suffix": ""},
    "C1_caution_think": {"instruction": "Think as carefully as possible before answering, even if this takes longer.", "suffix": ""},
    "C2_speed_think": {"instruction": "Think as briefly as possible before answering, even if this leads to more errors.", "suffix": ""},
    # Qwen3 soft switch: "/no_think" at the end of the user message yields an empty thinking block.
    "C3_no_think": {"instruction": "", "suffix": " /no_think"},
    # Chain of Draft (Xu et al. 2025), the reasoning instruction verbatim; the answer line replaces their "####" rule.
    "C4_draft": {"instruction": "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most.", "suffix": ""},
}
ANSWER_INSTRUCTION = "Answer with only A or B. No other words."

# ------------------------------------------------------------------ helpers


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def stable_hash(obj: Any, n: int = 12) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n]


def _jsonl_lines(rows: list[dict[str, Any]]) -> list[str]:
    return [json.dumps(r, ensure_ascii=False) + "\n" for r in rows]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_jsonl_lines(rows)), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources() -> None:
    """Every frozen file must be present with the recorded bytes."""
    for rel, want in RAW_FILES.items():
        path = RAW_DIR / rel
        if not path.exists():
            raise SystemExit(f"missing {path.relative_to(ROOT)}: run `python src/medxpertqa_dataset.py fetch`")
        got = sha256_file(path)
        if got != want:
            raise SystemExit(f"{path.relative_to(ROOT)} has sha256 {got}, expected {want}")


def load_source() -> list[dict[str, Any]]:
    check_sources()
    return read_jsonl(RAW_DIR / SOURCE_FILE)


# ------------------------------------------------------------------- stems


_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")


def split_stem(question: str) -> str:
    """The question text before the "Answer Choices:" line."""
    if question.count(CHOICES_MARKER) != 1:
        raise ValueError(f"expected exactly one {CHOICES_MARKER!r}: {question[:80]!r}")
    return question.split(CHOICES_MARKER)[0].strip()


def last_sentence(text: str) -> str:
    return _SENTENCE_SPLIT.split(text.strip())[-1]


def keep_reason(row: dict[str, Any]) -> str | None:
    """None if the question enters the dataset, otherwise why it does not."""
    if row["medical_task"] != KEEP_TASK:
        return f"medical_task {row['medical_task']}"
    if len(row["options"]) != len(SOURCE_LETTERS) or list(row["options"]) != list(SOURCE_LETTERS):
        return "options are not A-J"
    if row["label"] not in row["options"]:
        return "label not among options"
    if KEEP_PHRASE not in last_sentence(split_stem(row["question"])).lower():
        return "other wording"
    texts = [t.strip() for t in row["options"].values()]
    if len(set(texts)) != len(texts) or any("\n" in t or not t for t in texts):
        return "option texts not unique single lines"
    return None


# ------------------------------------------------------------------- build


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Items and a per-question report from the frozen source."""
    items, report = [], []
    for row in load_source():
        reason = keep_reason(row)
        stem = split_stem(row["question"]) if row["question"].count(CHOICES_MARKER) == 1 else None
        report.append({
            "source_id": row["id"], "medical_task": row["medical_task"], "question_type": row["question_type"],
            "body_system": row["body_system"], "kept": reason is None, "reason": reason,
            "last_sentence": last_sentence(stem) if stem else None, "stem_hash": stable_hash(stem) if stem else None,
        })
        if reason is not None:
            continue
        correct = row["label"]
        opts = {k: v.strip() for k, v in row["options"].items()}
        for distractor in SOURCE_LETTERS:
            if distractor == correct:
                continue
            pair_id = f"{row['id']}:{correct}-{distractor}"
            for order in ("ab", "ba"):
                a, b = (correct, distractor) if order == "ab" else (distractor, correct)
                items.append({
                    "item_id": f"{TASK}:{pair_id}:{order}",
                    "task": TASK,
                    "question": f"{stem}\nA. {opts[a]}\nB. {opts[b]}",
                    "options": list(OPTIONS),
                    "label": "A" if a == correct else "B",
                    "difficulty": {},  # filled per pair by src/pair_distance.py into data/pairs/
                    "meta": {
                        "data_version": DATA_VERSION,
                        "source": {"repo": HF_REPO, "revision": HF_REVISION, "file": SOURCE_FILE},
                        "source_id": row["id"],
                        "question_type": row["question_type"],
                        "body_system": row["body_system"],
                        "pair_id": pair_id,
                        "order": order,
                        "mirror_of": f"{TASK}:{pair_id}:{'ba' if order == 'ab' else 'ab'}",
                        "correct": {"letter": correct, "text": opts[correct]},
                        "distractor": {"letter": distractor, "text": opts[distractor]},
                        "option_a": {"letter": a, "text": opts[a]},
                        "option_b": {"letter": b, "text": opts[b]},
                        "n_source_options": len(SOURCE_LETTERS),
                        "stem_hash": stable_hash(stem),
                    },
                })
    return items, report


def user_message(question: str, condition: str) -> str:
    c = CONDITIONS[condition]
    parts = [question, c["instruction"], ANSWER_INSTRUCTION]
    return "\n\n".join(p for p in parts if p) + c["suffix"]


def build_prompts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        for condition in CONDITIONS:
            messages = [{"role": "user", "content": user_message(item["question"], condition)}]
            rows.append({
                "prompt_id": f"{item['item_id']}|{condition}",
                "item_id": item["item_id"],
                "task": item["task"],
                "condition": condition,
                "messages": messages,
                "options": item["options"],
                "label": item["label"],
                "prompt_hash": stable_hash(messages),
            })
    return rows


def print_counts(items: list[dict[str, Any]], report: list[dict[str, Any]]) -> None:
    reasons = Counter(r["reason"] or "kept" for r in report)
    print("Source questions by outcome:", ", ".join(f"{k} {v}" for k, v in reasons.most_common()))
    kept = [r for r in report if r["kept"]]
    print(f"Kept {len(kept)} questions -> {len(items) // 2} pairs, {len(items)} items; "
          f"question_type {dict(Counter(r['question_type'] for r in kept))}; "
          f"body systems {len(set(r['body_system'] for r in kept))}")


def cmd_build(args: argparse.Namespace) -> None:
    items, report = build()
    prompts = build_prompts(items)
    write_jsonl(TASKS_PATH, items)
    write_jsonl(PROMPTS_PATH, prompts)
    write_jsonl(REPORT_PATH, report)
    print_counts(items, report)
    print(f"wrote {TASKS_PATH.relative_to(ROOT)} ({len(items)}), {PROMPTS_PATH.relative_to(ROOT)} ({len(prompts)}), "
          f"{REPORT_PATH.relative_to(ROOT)} ({len(report)})")


# ------------------------------------------------------------------ verify


class Checks:
    def __init__(self) -> None:
        self.counts: dict[str, list[int]] = {}
        self.examples: dict[str, list[str]] = defaultdict(list)

    def __call__(self, name: str, ok: bool, detail: str = "") -> None:
        total, failed = self.counts.setdefault(name, [0, 0])
        self.counts[name][0] = total + 1
        if not ok:
            self.counts[name][1] = failed + 1
            if len(self.examples[name]) < 3:
                self.examples[name].append(detail)

    def report(self) -> bool:
        width = max(len(k) for k in self.counts)
        for name, (total, failed) in self.counts.items():
            print(f"  {'FAIL' if failed else 'ok  '}  {name:<{width}}  {total - failed}/{total}")
            for ex in self.examples[name]:
                print(f"          e.g. {ex}")
        return not any(f for _, f in self.counts.values())


def check_items(items: list[dict[str, Any]], source: dict[str, dict[str, Any]], check: Checks) -> None:
    by_id = {it["item_id"]: it for it in items}
    check("unique item_id", len(by_id) == len(items), f"{len(items) - len(by_id)} duplicates")
    per_question: dict[str, Counter] = defaultdict(Counter)
    for it in items:
        iid, m = it["item_id"], it["meta"]
        row = source.get(m["source_id"])
        check("source question exists and is a kept Diagnosis question", row is not None and keep_reason(row) is None, iid)
        if row is None:
            continue
        stem = split_stem(row["question"])
        a, b = m["option_a"], m["option_b"]
        check("options are [A, B]", it["options"] == OPTIONS, iid)
        check("option texts are the source's", a["text"] == row["options"][a["letter"]].strip()
              and b["text"] == row["options"][b["letter"]].strip(), iid)
        check("correct option is the source label", m["correct"]["letter"] == row["label"]
              and m["correct"]["letter"] in (a["letter"], b["letter"]) and m["distractor"]["letter"] != row["label"], iid)
        check("label marks the correct option", it["label"] == ("A" if a["letter"] == row["label"] else "B"), iid)
        check("question text is the stem plus the two options",
              it["question"] == f"{stem}\nA. {a['text']}\nB. {b['text']}", iid)
        check("stem ends with the diagnosis question", KEEP_PHRASE in last_sentence(stem).lower(), iid)
        mirror = by_id.get(m["mirror_of"])
        check("mirror item swaps A/B and flips the label", mirror is not None
              and mirror["meta"]["option_a"] == b and mirror["meta"]["option_b"] == a
              and mirror["meta"]["pair_id"] == m["pair_id"] and mirror["label"] != it["label"], iid)
        per_question[m["source_id"]][it["label"]] += 1
        per_question[m["source_id"]]["pairs"] += m["order"] == "ab"
        per_question[m["source_id"]]["distractors"] += 0
    for sid, c in per_question.items():
        check("9 pairs and 18 items per question", c["pairs"] == 9 and c["A"] + c["B"] == 18, f"{sid}: {dict(c)}")
        check("A/B labels balanced per question", c["A"] == c["B"], f"{sid}: {dict(c)}")
    kept_source = {sid for sid, row in source.items() if keep_reason(row) is None}
    check("every kept source question has items", kept_source == set(per_question),
          f"{sorted(kept_source ^ set(per_question))[:3]}")


def check_prompts(items: list[dict[str, Any]], prompts: list[dict[str, Any]], check: Checks) -> None:
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in prompts:
        by_item[p["item_id"]].append(p)
    check("every prompt belongs to an item", set(by_item) == {it["item_id"] for it in items})
    for it in items:
        rows = by_item.get(it["item_id"], [])
        check("one prompt per condition", [p["condition"] for p in rows] == list(CONDITIONS), it["item_id"])
        for p in rows:
            msgs = p["messages"]
            ok = (p["prompt_id"] == f"{it['item_id']}|{p['condition']}" and p["label"] == it["label"]
                  and p["options"] == it["options"] and p["prompt_hash"] == stable_hash(msgs)
                  and msgs == [{"role": "user", "content": user_message(it["question"], p["condition"])}])
            check("prompt carries the item's question, label and condition text", ok, p["prompt_id"])


def cmd_verify(args: argparse.Namespace) -> None:
    check = Checks()
    for rel, want in RAW_FILES.items():
        path = RAW_DIR / rel
        check("frozen source file matches the recorded sha256", path.exists() and sha256_file(path) == want, rel)
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    check("source manifest records the pinned revision", manifest.get("revision") == HF_REVISION
          and {k: v["sha256"] for k, v in manifest.get("files", {}).items()} == RAW_FILES)
    on_disk = {path: path.read_text(encoding="utf-8").splitlines(keepends=True)
               for path in (TASKS_PATH, PROMPTS_PATH, REPORT_PATH)}
    items = [json.loads(line) for line in on_disk[TASKS_PATH]]
    prompts = [json.loads(line) for line in on_disk[PROMPTS_PATH]]
    built_items, built_report = build()
    for path, rows in ((TASKS_PATH, built_items), (PROMPTS_PATH, build_prompts(built_items)), (REPORT_PATH, built_report)):
        lines = _jsonl_lines(rows)
        diff = [i for i, (x, y) in enumerate(zip(lines, on_disk[path])) if x != y]
        check(f"{path.name} in {path.parent.name}/ equals a rebuild from the frozen source",
              len(lines) == len(on_disk[path]) and not diff,
              f"{len(lines)} vs {len(on_disk[path])} lines, first differing line {diff[:1]}")
    source = {row["id"]: row for row in load_source()}
    check_items(items, source, check)
    check_prompts(items, prompts, check)
    print(f"Checks on {len(items)} items and {len(prompts)} prompts:")
    ok = check.report()
    print_counts(items, built_report)
    sys.exit(0 if ok else 1)


# ------------------------------------------------------------------- fetch


def cmd_fetch(args: argparse.Namespace) -> None:
    import requests

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for rel, want in RAW_FILES.items():
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{rel}"
        path = RAW_DIR / rel
        if path.exists() and sha256_file(path) == want and not args.force:
            print(f"  {rel}: present, sha256 ok")
        else:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            got = hashlib.sha256(r.content).hexdigest()
            if got != want and not args.force:
                raise SystemExit(f"{rel}: downloaded sha256 {got} differs from the frozen {want}; "
                                 f"use --force only to change the dataset")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            print(f"  {rel}: downloaded {len(r.content):,} bytes, sha256 {got[:12]}")
        files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size, "url": url}
    manifest = {
        "repo": HF_REPO, "revision": HF_REVISION, "license": HF_LICENSE, "fetched_at": _now(),
        "citation": "Zuo et al. (2025). MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding. ICML. arXiv:2501.18362",
        "files": files,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}")


# ------------------------------------------------------------------ select


def cmd_select(args: argparse.Namespace) -> None:
    """A selection file for src/generate.py: every item of one condition (screening)."""
    items = read_jsonl(TASKS_PATH)
    prompts = {p["item_id"]: p for p in read_jsonl(PROMPTS_PATH) if p["condition"] == args.condition}
    rows = []
    for it in items:
        m, p = it["meta"], prompts[it["item_id"]]
        rows.append({
            "prompt_id": p["prompt_id"], "item_id": it["item_id"], "pair_id": m["pair_id"],
            "source_id": m["source_id"], "order": m["order"],
            "correct": m["correct"]["letter"], "distractor": m["distractor"]["letter"],
            "correct_text": m["correct"]["text"], "distractor_text": m["distractor"]["text"],
            "body_system": m["body_system"], "question_type": m["question_type"],
            "options": p["options"], "label": p["label"], "messages": p["messages"],
        })
    out = {
        "task": TASK, "purpose": args.name, "condition": args.condition,
        "select_seed": f"20260930:{args.name}",
        "source": {"tasks_sha256": sha256_file(TASKS_PATH), "prompts_sha256": sha256_file(PROMPTS_PATH)},
        "prompts": rows,
    }
    path = SELECTIONS_DIR / f"{args.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} prompts ({len(rows) // 2} pairs, {len({r['source_id'] for r in rows})} questions) "
          f"-> {path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download the frozen source files from Hugging Face (network)")
    f.add_argument("--force", action="store_true", help="accept files whose bytes differ (changes the dataset)")
    f.set_defaults(fn=cmd_fetch)
    sub.add_parser("build", help="frozen source -> data/tasks, data/prompts, data/verification").set_defaults(fn=cmd_build)
    sub.add_parser("verify", help="check every item and prompt against a rebuild and the source").set_defaults(fn=cmd_verify)
    s = sub.add_parser("select", help="write a selection file with every item of one condition")
    s.add_argument("--condition", default="C0_baseline", choices=list(CONDITIONS))
    s.add_argument("--name", default="medxpertqa_screen", help="data/selections/<name>.json")
    s.set_defaults(fn=cmd_select)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
