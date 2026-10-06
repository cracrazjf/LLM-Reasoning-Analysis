"""Instruction pilot: what each condition does to length, accuracy, answer letter and format.

    python src/pilot_analysis.py --run runs/medxpertqa/pilot-qwen3-8b

Every prompt (pair x order) was sampled under every condition, so each condition is compared
with the baseline WITHIN prompt: length ratio = median over prompts of (median thinking tokens
under the condition / under C0); accuracy change = mean over prompts of (accuracy under the
condition - under C0) with a bootstrap CI over prompts; letter shift = change in the share of
answers that are "B"; format = share of answers that are the bare letter. Reported per band
(hard / mid) and overall. Writes <run>/analysis/summary.json and prints the tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from generate import iter_traces, parse_answer  # noqa: E402

BASE = "C0_baseline"


def load(run: Path) -> pd.DataFrame:
    rows = []
    for r in iter_traces(run, ("prompt_id", "item_id", "pair_id", "source_id", "order", "band", "condition", "label",
                               "answer_text", "think_tokens", "n_tokens", "finish_reason", "X")):
        a, fmt = parse_answer(r["answer_text"], ("A", "B"))
        rows.append({**r, "answer": a, "fmt": fmt, "correct": (a == r["label"]) if a else None,
                     "is_B": (a == "B") if a else None})
    return pd.DataFrame(rows)


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def per_prompt(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (item, condition): accuracy, median thinking tokens, share B, format compliance."""
    g = df.groupby(["item_id", "pair_id", "source_id", "band", "order", "condition"])
    out = g.agg(n=("answer", "size"), answered=("answer", lambda a: a.notna().mean()),
                acc=("correct", lambda c: c.dropna().astype(float).mean() if c.notna().any() else np.nan),
                think=("think_tokens", "median"), think_mean=("think_tokens", "mean"),
                share_B=("is_B", lambda b: b.dropna().astype(float).mean() if b.notna().any() else np.nan),
                bare_letter=("fmt", lambda f: (f == "letter").mean()),
                truncated=("finish_reason", lambda f: (f == "length").mean())).reset_index()
    return out


def compare(pp: pd.DataFrame, cond: str, band: str | None) -> dict:
    base = pp[pp.condition == BASE].set_index("item_id")
    c = pp[pp.condition == cond].set_index("item_id")
    if band:
        base, c = base[base.band == band], c[c.band == band]
    idx = base.index.intersection(c.index)
    base, c = base.loc[idx], c.loc[idx]
    ratio = (c.think.clip(lower=1) / base.think.clip(lower=1)).values
    dacc = (c.acc - base.acc).dropna().values
    dB = (c.share_B - base.share_B).dropna().values
    lo, hi = boot_ci(dacc) if len(dacc) > 1 else (np.nan, np.nan)
    return {
        "n_prompts": int(len(idx)),
        "think_tokens_median": float(c.think.median()), "baseline_think_median": float(base.think.median()),
        "length_ratio_median": float(np.median(ratio)), "share_prompts_shorter": float((ratio < 1).mean()),
        "accuracy": float(c.acc.mean()), "baseline_accuracy": float(base.acc.mean()),
        "accuracy_change_mean": float(dacc.mean()), "accuracy_change_ci95": [lo, hi],
        "share_prompts_less_accurate": float((dacc < 0).mean()), "share_prompts_more_accurate": float((dacc > 0).mean()),
        "share_B": float(c.share_B.mean()), "share_B_change_mean": float(dB.mean()),
        "bare_letter_share": float(c.bare_letter.mean()), "answered_share": float(c.answered.mean()),
        "truncated_share": float(c.truncated.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--exclude-non-diagnosis", action="store_true",
                    help="drop pair_distance.NON_DIAGNOSIS_PAIRS (options that are tests, not diagnoses)")
    args = ap.parse_args()
    df = load(args.run)
    if df.empty:
        raise SystemExit(f"no traces in {args.run}")
    if args.exclude_non_diagnosis:
        from pair_distance import NON_DIAGNOSIS_PAIRS
        n0 = df.pair_id.nunique()
        df = df[~df.pair_id.isin(NON_DIAGNOSIS_PAIRS)]
        print(f"excluded {n0 - df.pair_id.nunique()} non-diagnosis pairs")
    pp = per_prompt(df)
    conds = [c for c in pp.condition.unique() if c != BASE]
    summary = {"run": str(args.run), "n_traces": int(len(df)), "n_prompts": int(pp.item_id.nunique()),
               "conditions": [BASE] + sorted(conds), "baseline": {}, "by_condition": {}}
    bands = sorted(pp.band.dropna().unique())
    for band in [None] + bands:
        b = pp[(pp.condition == BASE) & ((pp.band == band) if band else True)]
        summary["baseline"][band or "all"] = {"accuracy": float(b.acc.mean()), "think_tokens_median": float(b.think.median()),
                                              "share_B": float(b.share_B.mean()), "bare_letter_share": float(b.bare_letter.mean())}
    for cond in sorted(conds):
        summary["by_condition"][cond] = {band or "all": compare(pp, cond, band) for band in [None] + bands}
    out = args.run / "analysis"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    pp.to_csv(out / "per_prompt.csv", index=False)
    # tables
    print(f"{summary['n_traces']} traces, {summary['n_prompts']} prompts; baseline: "
          + ", ".join(f"{k}: acc {v['accuracy']:.2f} think {v['think_tokens_median']:.0f}" for k, v in summary["baseline"].items()))
    hdr = f"{'condition':<18}{'band':<6}{'n':>4}{'think':>7}{'ratio':>7}{'acc':>6}{'d_acc':>7}{'ci95':>16}{'dB':>7}{'bare':>6}{'trunc':>7}"
    print(hdr)
    for cond in sorted(conds):
        for band in ["all"] + bands:
            r = summary["by_condition"][cond][band]
            print(f"{cond:<18}{band:<6}{r['n_prompts']:>4}{r['think_tokens_median']:>7.0f}{r['length_ratio_median']:>7.2f}"
                  f"{r['accuracy']:>6.2f}{r['accuracy_change_mean']:>+7.2f}  [{r['accuracy_change_ci95'][0]:+.2f},{r['accuracy_change_ci95'][1]:+.2f}]"
                  f"{r['share_B_change_mean']:>+7.2f}{r['bare_letter_share']:>6.2f}{r['truncated_share']:>7.3f}")
    print(f"-> {out}/summary.json, per_prompt.csv")


if __name__ == "__main__":
    main()
