"""The two question sets of the experiment and their files.

    python src/selection.py helps                       # data/selections/medxpertqa_helps.json (the main run's file)
    python src/selection.py hurts --conditions C0_baseline   # data/selections/medxpertqa_hurts.json
    python src/selection.py export                      # pair lists, condition wording, full prompts

A selection file lists the prompts of one generation run (src/generate.py): the pairs of one
thinking-gain class in both option orders, times the conditions, with every field a trace should
carry and the random seed of the run. The classes come from data/pairs/medxpertqa_gain.csv
(thinking gain = accuracy with thinking minus accuracy without, from the 2026-10 screening):
helps = gain >= 0.2 in each order, hurts = gain <= -0.2 in each order; NON_DIAGNOSIS_PAIRS are left
out. The helps file was written on 2026-10-03 as medxpertqa_helps_pilot.json and renamed; its
seed string must stay as it is, because the sample seeds of the main run derive from it.

export writes the same material as plain files:
  data/pairs/medxpertqa_helps70.csv, medxpertqa_hurts40.csv   one row per pair: options, screening
      numbers, and the accuracy per condition where a run exists
  data/prompts/conditions.json         the three conditions' wording and the prompt template
  data/prompts/medxpertqa_main.jsonl   the full prompts: 110 pairs x 2 orders x 3 conditions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from medxpertqa_dataset import (  # noqa: E402
    ANSWER_INSTRUCTION, CONDITIONS, PROMPTS_PATH, ROOT, SELECTIONS_DIR, TASKS_PATH, read_jsonl, sha256_file, user_message)

PAIRS_DIR = ROOT / "data" / "pairs"
MAIN_CONDITIONS = ("C0_baseline", "C1_caution_think", "C2_speed_think")


# Pairs whose options are tests or management rather than diagnoses (the stem filter admits a few such
# questions); excluded from the helps pilot and the main run at the user's request (2026-10-03).
NON_DIAGNOSIS_PAIRS = {
    **{f"Text-164:G-{d}": "options are tests ('No additional tests are required' vs a test)" for d in "ABDEFHIJ"},
    "Text-500:F-J": "both options name the same diagnosis and differ only in the imaging test",
}


def class_pairs(band: str):
    """The pairs of one thinking-gain class in both orders (data/pairs/medxpertqa_gain.csv, src/thinking_gain.py),
    minus NON_DIAGNOSIS_PAIRS; returns the table and the excluded pairs with reasons."""
    gain = pd.read_csv(PAIRS_DIR / "medxpertqa_gain.csv")
    chosen = gain[gain[f"{band}_both_orders"] == True].sort_values("pair_id")  # noqa: E712
    excluded = {pid: why for pid, why in NON_DIAGNOSIS_PAIRS.items() if pid in set(chosen.pair_id)}
    return chosen[~chosen.pair_id.isin(excluded)], excluded


def cmd_class_selection(args: argparse.Namespace) -> None:
    """Selection file for the pairs where reasoning helps (or hurts) in both orders: both orders x the given
    conditions; rows ordered pair, order, condition. `helps-pilot` wrote the main run's selection; `hurts` writes
    the mirror set."""
    chosen, excluded = class_pairs(args.band)
    items = {it["meta"]["pair_id"] + ":" + it["meta"]["order"]: it for it in read_jsonl(TASKS_PATH)}
    prompts = {(p["item_id"], p["condition"]): p for p in read_jsonl(PROMPTS_PATH)}
    conds = args.conditions.split(",")
    rows = []
    for r in chosen.itertuples():
        for order in ("ab", "ba"):
            it = items[f"{r.pair_id}:{order}"]
            m = it["meta"]
            for cond in conds:
                p = prompts[(it["item_id"], cond)]
                rows.append({
                    "prompt_id": p["prompt_id"], "item_id": it["item_id"], "pair_id": r.pair_id, "source_id": m["source_id"],
                    "order": order, "condition": cond, "band": args.band, "gain": float(r.gain),
                    "acc_think_screen": float(r.acc_think), "acc_nothink_screen": float(r.acc_nothink),
                    "correct": m["correct"]["letter"], "distractor": m["distractor"]["letter"],
                    "correct_text": m["correct"]["text"], "distractor_text": m["distractor"]["text"],
                    "body_system": m["body_system"], "question_type": m["question_type"],
                    "options": p["options"], "label": p["label"], "messages": p["messages"],
                })
    sign = ">= 0.2" if args.band == "helps" else "<= -0.2"
    out = {"task": "medxpertqa", "purpose": args.purpose, "conditions": conds, "select_seed": f"{args.seed_date}:{args.name}",
           "rule": f"pairs with {args.band}_both_orders in medxpertqa_gain.csv (thinking gain {sign} in each order), minus NON_DIAGNOSIS_PAIRS",
           "excluded": excluded,
           "n_pairs": int(len(chosen)), "n_questions": int(chosen.source_id.nunique()),
           "source": {"prompts_sha256": sha256_file(PROMPTS_PATH), "gain_csv_sha256": sha256_file(PAIRS_DIR / "medxpertqa_gain.csv")},
           "prompts": rows}
    path = SELECTIONS_DIR / f"{args.name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(chosen)} pairs, {chosen.source_id.nunique()} questions, {len(rows)} prompts ({len(conds)} conditions) -> {path.relative_to(ROOT)}")


def cmd_export_main(args: argparse.Namespace) -> None:
    """The main experiment's material as plain files:
      data/pairs/medxpertqa_helps70.csv, medxpertqa_hurts40.csv   one row per pair: options, screening numbers,
          and the accuracy per condition where a run exists (main run for helps, instruction pilot for hurts)
      data/prompts/conditions.json         the three conditions' wording and the prompt template
      data/prompts/medxpertqa_main.jsonl   the full prompts of both classes: pairs x 2 orders x 3 conditions
    """
    items = {it["meta"]["pair_id"] + ":" + it["meta"]["order"]: it for it in read_jsonl(TASKS_PATH)}
    prompts = {(p["item_id"], p["condition"]): p for p in read_jsonl(PROMPTS_PATH)}
    acc = {}
    for band, run in (("helps", ROOT / "runs/medxpertqa/main-qwen3-8b"), ("hurts", ROOT / "runs/medxpertqa/pilot-qwen3-8b")):
        f = run / "analysis" / ("per_prompt.csv" if band == "helps" else "per_prompt.csv")
        if f.exists():
            acc[band] = pd.read_csv(f)
    main_rows, counts = [], {}
    for band, name in (("helps", "medxpertqa_helps70"), ("hurts", "medxpertqa_hurts40")):
        chosen, _ = class_pairs(band)
        rows = []
        for r in chosen.itertuples():
            m = items[f"{r.pair_id}:ab"]["meta"]
            row = {"pair_id": r.pair_id, "source_id": r.source_id, "band": band, "body_system": m["body_system"], "question_type": m["question_type"],
                   "correct_letter": m["correct"]["letter"], "correct_text": m["correct"]["text"],
                   "distractor_letter": m["distractor"]["letter"], "distractor_text": m["distractor"]["text"],
                   "acc_think_screen": r.acc_think, "acc_think_screen_ab": r.acc_think_ab, "acc_think_screen_ba": r.acc_think_ba,
                   "acc_nothink_screen": r.acc_nothink, "gain": r.gain, "think_tokens_screen": r.think_tokens}
            if band in acc:
                a = acc[band]
                a = a[a.pair_id == r.pair_id] if "pair_id" in a else a[a.item_id.str.contains(r.pair_id + ":")]
                for cond in ("baseline", "careful", "speed"):
                    col = "cond" if "cond" in a else "condition"
                    g = a[a[col].astype(str).str.contains(cond)]
                    if len(g):
                        row[f"acc_{cond}_run"] = float((g.acc * g.n).sum() / g.n.sum()) if "n" in g else float(g.acc.mean())
                        row[f"think_{cond}_run"] = float(g.think.mean()) if "think" in g else None
            rows.append(row)
            for order in ("ab", "ba"):
                it = items[f"{r.pair_id}:{order}"]
                for cond in MAIN_CONDITIONS:
                    p = prompts[(it["item_id"], cond)]
                    main_rows.append({"prompt_id": p["prompt_id"], "item_id": it["item_id"], "pair_id": r.pair_id, "source_id": r.source_id,
                                      "band": band, "order": order, "condition": cond, "label": p["label"], "options": p["options"],
                                      "messages": p["messages"]})
        pd.DataFrame(rows).to_csv(PAIRS_DIR / f"{name}.csv", index=False)
        counts[band] = (len(rows), int(chosen.source_id.nunique()))
    example = items["Text-1074:A-B:ab"]["question"]
    conditions = {
        "answer_line": ANSWER_INSTRUCTION,
        "template": "question text, then the instruction sentence (none for the baseline), then the answer line, separated by blank lines; "
                    "sent as a single user message to Qwen3-8B in thinking mode (chat template with enable_thinking=True)",
        "conditions": {c: {"instruction": CONDITIONS[c]["instruction"], "example_user_message": user_message(example, c)} for c in MAIN_CONDITIONS},
        "sources": "docs/PROMPT_SOURCES.md",
    }
    (PROMPTS_PATH.parent / "conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with (PROMPTS_PATH.parent / "medxpertqa_main.jsonl").open("w", encoding="utf-8") as fh:
        for row in main_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"helps {counts['helps'][0]} pairs / {counts['helps'][1]} questions, hurts {counts['hurts'][0]} pairs / {counts['hurts'][1]} questions; "
          f"{len(main_rows)} prompts -> data/pairs/medxpertqa_helps70.csv, medxpertqa_hurts40.csv, data/prompts/conditions.json, medxpertqa_main.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for band, name, purpose, seed_date in (("helps", "medxpertqa_helps", "helps_pilot", "20261003"), ("hurts", "medxpertqa_hurts", "hurts_main", "20261005")):
        s = sub.add_parser(band, help=f"selection file of the pairs where reasoning {band} in both orders")
        s.add_argument("--conditions", default=",".join(MAIN_CONDITIONS))
        s.add_argument("--name", default=name)
        s.set_defaults(fn=cmd_class_selection, band=band, purpose=purpose, seed_date=seed_date)
    e = sub.add_parser("export", help="pair lists, condition wording and full prompts as plain files")
    e.set_defaults(fn=cmd_export_main)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
