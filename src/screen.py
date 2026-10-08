"""Classify every prompt by what thinking does to it, from the two screening runs.

    python src/screen.py --nothink runs/medxpertqa/nothink-qwen3-8b-ab runs/medxpertqa/nothink-qwen3-8b-ba \
                         --think   runs/medxpertqa/think-qwen3-8b-ab   runs/medxpertqa/think-qwen3-8b-ba

Both options take one or more run directories (one per option order when the orders ran on
separate pods). Rule, with T = 70% of the thinking samples (--threshold):
  no-thinking answer wrong:  helps  if >= T of the thinking samples answer correctly;
                             hurts  if >= T have a lower gap than without thinking (more wrong);
  no-thinking answer right:  hurts  if >= T of the thinking samples answer wrongly;
                             helps  if >= T have a higher gap than without thinking (more confident);
  otherwise neutral.
gap = logp(correct letter) - logp(wrong letter) at the answer position, i.e. the logit difference
(src/generate.py). Shares are over all thinking samples of the prompt; a sample without an answer
position (no </think> before the token cap) counts as neither correct, wrong nor moved.
The no-thinking answer is the letter the model wrote; if it wrote none, the larger of logp_A and
logp_B at the answer position (nothink_answer_source says which).

Letter bias: a pair whose no-thinking answer is right in one order and wrong in the other gave
the same letter both times, so that answer is the position, not the question. Both its prompts
get cls "excluded" (column letter_biased = True) and stay out of the counts; --keep-biased
classifies them like the others.

Writes data/screen/medxpertqa_prompts.csv (one row per prompt = pair x order),
medxpertqa_pairs.csv (one row per pair: the class in each order and pair_cls, the shared class or
"mixed") and medxpertqa_summary.json, and prints the counts.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from generate import ROOT, iter_traces, load_tasks

OUT = ROOT / "data/screen"
FIELDS = ("sample_id", "item_id", "answer", "correct", "logp_A", "logp_B", "p_A", "p_B", "gap", "X_bound",
          "think_tokens", "finish_reason")
TASK_FIELDS = ("item_id", "pair_id", "order", "source_id", "question_type", "body_system", "label",
               "correct_text", "distractor_text")


def nothink_rows(runs: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for r in (rec for run in runs for rec in iter_traces(run, FIELDS)):
        if r["item_id"] in rows:
            raise ValueError(f"more than one no-thinking answer for {r['item_id']}")
        answer, source = r["answer"], "text"
        if answer is None and (r["logp_A"] is not None or r["logp_B"] is not None):
            la = r["logp_A"] if r["logp_A"] is not None else float("-inf")
            lb = r["logp_B"] if r["logp_B"] is not None else float("-inf")
            answer, source = ("A" if la >= lb else "B"), "readout"
        rows[r["item_id"]] = {
            "nothink_answer": answer, "nothink_answer_source": source if answer else None,
            "nothink_logp_A": r["logp_A"], "nothink_logp_B": r["logp_B"], "nothink_p_A": r["p_A"],
            "nothink_p_B": r["p_B"], "nothink_gap": r["gap"], "nothink_gap_bound": r["X_bound"],
        }
    return rows


def think_rows(runs: list[Path]) -> dict[str, list[dict[str, Any]]]:
    per: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for r in iter_traces(run, FIELDS):
            per[r["item_id"]].append(r)
    return per


def classify(label: str, nt: dict[str, Any] | None, samples: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    n = len(samples)
    n_answered = sum(s["answer"] is not None for s in samples)
    n_correct = sum(bool(s["correct"]) for s in samples)
    n_wrong = n_answered - n_correct
    gaps = [s["gap"] for s in samples if s["gap"] is not None]
    tokens = [s["think_tokens"] for s in samples if s["think_tokens"] is not None]
    gap0 = nt["nothink_gap"] if nt else None
    n_up = sum(g > gap0 for g in gaps) if gap0 is not None else 0
    n_down = sum(g < gap0 for g in gaps) if gap0 is not None else 0
    share = lambda k: k / n if n else None  # noqa: E731
    nothink_correct = (nt["nothink_answer"] == label) if nt and nt["nothink_answer"] else None
    if nt is None:
        cls, reason = "unclassified", "no no-thinking answer record"
    elif nothink_correct is None:
        cls, reason = "unclassified", "no no-thinking answer"
    elif n == 0:
        cls, reason = "unclassified", "no thinking samples"
    elif gap0 is None and not nothink_correct and share(n_correct) < threshold:
        cls, reason = "unclassified", "no no-thinking gap"
    elif gap0 is None and nothink_correct and share(n_wrong) < threshold:
        cls, reason = "unclassified", "no no-thinking gap"
    elif not nothink_correct:
        if share(n_correct) >= threshold:
            cls, reason = "helps", "flips to correct"
        elif share(n_down) >= threshold:
            cls, reason = "hurts", "more wrong"
        else:
            cls, reason = "neutral", ""
    else:
        if share(n_wrong) >= threshold:
            cls, reason = "hurts", "flips to wrong"
        elif share(n_up) >= threshold:
            cls, reason = "helps", "more confident"
        else:
            cls, reason = "neutral", ""
    return {
        "nothink_correct": nothink_correct,
        "n_think": n, "n_answered": n_answered, "n_correct": n_correct, "n_wrong": n_wrong,
        "share_correct": share(n_correct), "share_wrong": share(n_wrong),
        "share_gap_up": share(n_up), "share_gap_down": share(n_down),
        "think_gap_median": statistics.median(gaps) if gaps else None,
        "think_gap_min": min(gaps) if gaps else None, "think_gap_max": max(gaps) if gaps else None,
        "think_tokens_median": statistics.median(tokens) if tokens else None,
        "n_truncated": sum(s["finish_reason"] == "length" for s in samples),
        "cls": cls, "reason": reason,
    }


def letter_biased_pairs(tasks: list[dict[str, Any]], nothink: dict[str, dict[str, Any]]) -> set[str]:
    """Pairs whose no-thinking answer is right in one order and wrong in the other (the same letter twice)."""
    correct: dict[str, dict[str, bool]] = defaultdict(dict)
    for it in tasks:
        nt = nothink.get(it["item_id"])
        if nt and nt["nothink_answer"]:
            correct[it["pair_id"]][it["order"]] = nt["nothink_answer"] == it["label"]
    return {pid for pid, c in correct.items() if len(c) == 2 and c["ab"] != c["ba"]}


def pair_table(prompts: pd.DataFrame) -> pd.DataFrame:
    keep = ["nothink_answer", "nothink_correct", "nothink_gap", "share_correct", "share_gap_up", "share_gap_down",
            "think_tokens_median", "cls"]
    rows = []
    for pair_id, g in prompts.groupby("pair_id", sort=True):
        first = g.iloc[0]
        row = {"pair_id": pair_id, "source_id": first["source_id"], "question_type": first["question_type"],
               "body_system": first["body_system"], "correct_text": first["correct_text"],
               "distractor_text": first["distractor_text"], "letter_biased": bool(first["letter_biased"])}
        by_order = {r["order"]: r for _, r in g.iterrows()}
        for order in ("ab", "ba"):
            r = by_order.get(order)
            for k in keep:
                row[f"{k}_{order}"] = r[k] if r is not None else None
        classes = {row["cls_ab"], row["cls_ba"]}
        if "excluded" in classes:
            row["pair_cls"] = "excluded"
        elif None in classes or "unclassified" in classes:
            row["pair_cls"] = "unclassified"
        elif len(classes) == 1:
            row["pair_cls"] = classes.pop()
        else:
            row["pair_cls"] = "mixed"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nothink", required=True, type=Path, nargs="+", help="run directories of the no-thinking pass")
    ap.add_argument("--think", required=True, type=Path, nargs="+", help="run directories of the thinking samples")
    ap.add_argument("--threshold", type=float, default=0.7, help="share of thinking samples (default 0.7)")
    ap.add_argument("--keep-biased", action="store_true", help="classify letter-biased pairs instead of excluding them")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    tasks = load_tasks()
    nothink = nothink_rows(args.nothink)
    think = think_rows(args.think)
    biased = letter_biased_pairs(tasks, nothink)
    rows = []
    for it in tasks:
        nt = nothink.get(it["item_id"])
        row = {k: it[k] for k in TASK_FIELDS}
        row["letter_biased"] = it["pair_id"] in biased
        row.update(nt or {"nothink_answer": None, "nothink_answer_source": None, "nothink_logp_A": None,
                          "nothink_logp_B": None, "nothink_p_A": None, "nothink_p_B": None, "nothink_gap": None,
                          "nothink_gap_bound": None})
        row.update(classify(it["label"], nt, think.get(it["item_id"], []), args.threshold))
        if row["letter_biased"] and not args.keep_biased:
            row["cls"], row["reason"] = "excluded", "letter bias: the same letter in both orders"
        rows.append(row)
    prompts = pd.DataFrame(rows)
    pairs = pair_table(prompts)

    args.out.mkdir(parents=True, exist_ok=True)
    prompts.to_csv(args.out / "medxpertqa_prompts.csv", index=False)
    pairs.to_csv(args.out / "medxpertqa_pairs.csv", index=False)

    def counts(s: pd.Series) -> dict[str, int]:
        return {str(k): int(v) for k, v in Counter(s.dropna()).most_common()}

    n_think = prompts["n_think"]
    kept = prompts[prompts["cls"] != "excluded"]
    summary = {
        "threshold": args.threshold,
        "runs": {"nothink": [str(p) for p in args.nothink], "think": [str(p) for p in args.think]},
        "n_prompts": int(len(prompts)), "n_pairs": int(len(pairs)),
        "thinking_samples_per_prompt": {"min": int(n_think.min()), "max": int(n_think.max())},
        "nothink_answer_source": counts(prompts["nothink_answer_source"]),
        "nothink_correct": counts(prompts["nothink_correct"].map({True: "right", False: "wrong"})),
        "nothink_accuracy": float(prompts["nothink_correct"].mean()),
        "think_accuracy": float((prompts["n_correct"].sum() / n_think.sum()) if n_think.sum() else float("nan")),
        "letter_biased_pairs": int(pairs["letter_biased"].sum()),
        "letter_biased_excluded": not args.keep_biased,
        "prompt_classes": counts(prompts["cls"]),
        "prompt_classes_by_nothink": {
            "wrong": counts(kept.loc[kept["nothink_correct"] == False, "cls"]),   # noqa: E712
            "right": counts(kept.loc[kept["nothink_correct"] == True, "cls"]),    # noqa: E712
        },
        "prompt_reasons": counts(kept.loc[kept["reason"] != "", "reason"]),
        "pair_classes": counts(pairs["pair_cls"]),
        "pairs_both_orders": {c: int((pairs["pair_cls"] == c).sum()) for c in ("helps", "hurts", "neutral")},
        "questions_with_a_helps_pair": int(pairs.loc[pairs["pair_cls"] == "helps", "source_id"].nunique()),
        "questions_with_a_hurts_pair": int(pairs.loc[pairs["pair_cls"] == "hurts", "source_id"].nunique()),
    }
    (args.out / "medxpertqa_summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    print(f"{summary['n_prompts']} prompts, {summary['n_pairs']} pairs; thinking samples per prompt "
          f"{summary['thinking_samples_per_prompt']['min']}-{summary['thinking_samples_per_prompt']['max']}")
    print(f"no-thinking accuracy {summary['nothink_accuracy']:.3f} ({summary['nothink_correct']}), "
          f"thinking accuracy {summary['think_accuracy']:.3f}")
    print(f"letter-biased pairs: {summary['letter_biased_pairs']}"
          + (" (excluded)" if summary["letter_biased_excluded"] else " (kept)"))
    print("prompt classes:", summary["prompt_classes"])
    print("  no-thinking wrong:", summary["prompt_classes_by_nothink"]["wrong"])
    print("  no-thinking right:", summary["prompt_classes_by_nothink"]["right"])
    print("  reasons:", summary["prompt_reasons"])
    print("pair classes (both orders):", summary["pair_classes"])
    print(f"-> {args.out / 'medxpertqa_prompts.csv'}, medxpertqa_pairs.csv, medxpertqa_summary.json")


if __name__ == "__main__":
    main()
