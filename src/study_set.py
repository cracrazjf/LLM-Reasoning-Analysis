"""Draw the 20-pair study set for the stopping-rule analysis.

    python src/study_set.py        # -> data/screen/medxpertqa_study20.csv (and prints it)

Rule (user's decision 2026-10-08): take pairs where thinking is decisive in both orders, either
mostly right or mostly wrong across the 30 samples; pairs near 50% are left out.

  group A  "right, first reaction wrong"  thinking accuracy >= 0.7 in both orders (the flip class; pairs
           at >= 0.8 are drawn first), no-thinking gap < -1 in both orders                 8 pairs
  group B  "right, first reaction right"  accuracy >= 0.8 in both orders, no-thinking gap > 1 in both   4 pairs
  group C  "wrong, first reaction wrong"  accuracy <= 0.2 in both orders, no-thinking gap < -1 in both  8 pairs
           (at most 4 of them osteopathic-manipulation items, which dominate this pool)

Common filters: letter-biased pairs out (src/screen.py), no truncated sample, all 30 samples answered,
median thinking length 500-2,500 tokens in both orders, not Text-164 (options are tests, not
diagnoses), one pair per source question. Within a group, pairs whose stopping time is spread
(p90/p10 of thinking tokens >= 2 in both orders) are drawn first; the rest fill up if needed. If a
group's pool runs out, group B (the largest pool) is topped up so the set has 20 pairs.
Draws are random with seed 20261008.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import ROOT, iter_traces  # noqa: E402

SCREEN = ROOT / "data/screen"
RUNS = [ROOT / "runs/medxpertqa/think-qwen3-8b-ab", ROOT / "runs/medxpertqa/think-qwen3-8b-ba"]
SEED = 20261008
OSTEO = r"torsion|strain|tender point|sacral|innominate|shear|extension|flexion|rotation|sidebending|ERLSL|RSRRL|NSRRL|inflare|outflare|dysfunction|vertebral|rib|somatic"
GROUPS = [("A", "right, first reaction wrong", 8), ("B", "right, first reaction right", 4), ("C", "wrong, first reaction wrong", 8)]
MAX_OSTEO_C = 4


def main() -> None:
    p = pd.read_csv(SCREEN / "medxpertqa_prompts.csv")
    k = p[p.cls != "excluded"].copy()
    s = pd.DataFrame([r for run in RUNS for r in iter_traces(run, ("item_id", "think_tokens"))])
    g = s.groupby("item_id").think_tokens
    k = k.join(pd.DataFrame({"tok_p10": g.quantile(0.1), "tok_p50": g.median(), "tok_p90": g.quantile(0.9)}), on="item_id")
    k["tok_ratio"] = k.tok_p90 / k.tok_p10

    vals = ["share_correct", "nothink_gap", "think_gap_median", "tok_p50", "tok_ratio", "n_truncated", "n_answered", "reason", "cls"]
    w = k.pivot(index="pair_id", columns="order", values=vals)
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    meta = k.drop_duplicates("pair_id").set_index("pair_id")[["source_id", "body_system", "question_type", "correct_text", "distractor_text"]]
    w = w.join(meta)
    w["osteo"] = w.correct_text.str.contains(OSTEO, case=False, regex=True) | w.distractor_text.str.contains(OSTEO, case=False, regex=True)
    clean = ((w.n_truncated_ab == 0) & (w.n_truncated_ba == 0) & (w.n_answered_ab == 30) & (w.n_answered_ba == 30)
             & w.tok_p50_ab.between(500, 2500) & w.tok_p50_ba.between(500, 2500) & (w.source_id != "Text-164"))
    right8 = (w.share_correct_ab >= 0.8) & (w.share_correct_ba >= 0.8)
    right7 = (w.share_correct_ab >= 0.7) & (w.share_correct_ba >= 0.7)
    wrong = (w.share_correct_ab <= 0.2) & (w.share_correct_ba <= 0.2)
    prior_wrong = (w.nothink_gap_ab < -1) & (w.nothink_gap_ba < -1)
    prior_right = (w.nothink_gap_ab > 1) & (w.nothink_gap_ba > 1)
    wide = (w.tok_ratio_ab >= 2) & (w.tok_ratio_ba >= 2)
    pools = {"A": w[clean & right7 & prior_wrong], "B": w[clean & right8 & prior_right], "C": w[clean & wrong & prior_wrong]}

    rng = np.random.default_rng(SEED)
    chosen, used_questions = [], set()

    def draw(key: str, label: str, n: int) -> None:
        pool = pools[key].copy()
        pool["wide"] = wide.loc[pool.index]
        pool["strong"] = right8.loc[pool.index] if key == "A" else True
        pool = pool.iloc[rng.permutation(len(pool))]                       # random order ...
        pool = pool.sort_values(["wide", "strong"], ascending=False, kind="stable")   # ... wide spread first, then >= 0.8
        n_osteo = sum(c["osteo"] for c in chosen if c["group"] == key)
        for pid, row in pool.iterrows():
            if len([c for c in chosen if c["group"] == key]) >= n:
                break
            if row.source_id in used_questions:
                continue
            if key == "C" and row.osteo:
                if n_osteo >= MAX_OSTEO_C:
                    continue
                n_osteo += 1
            used_questions.add(row.source_id)
            chosen.append({"group": key, "group_label": label, "pair_id": pid, **row.drop(["wide", "strong"]).to_dict()})
        print(f"group {key} ({label}): pool {len(pool)} pairs / {pool.source_id.nunique()} questions, "
              f"{int(pool.wide.sum())} with a wide stopping-time spread -> {len([c for c in chosen if c['group'] == key])} drawn")

    for key, label, n in GROUPS:
        draw(key, label, n)
    short = sum(n for _, _, n in GROUPS) - len(chosen)
    if short > 0:
        print(f"{short} short: topping up group B")
        draw("B", GROUPS[1][1], len([c for c in chosen if c["group"] == "B"]) + short)

    out = pd.DataFrame(chosen)
    cols = ["group", "group_label", "pair_id", "source_id", "body_system", "question_type", "correct_text", "distractor_text", "osteo",
            "share_correct_ab", "share_correct_ba", "nothink_gap_ab", "nothink_gap_ba", "think_gap_median_ab", "think_gap_median_ba",
            "tok_p50_ab", "tok_p50_ba", "tok_ratio_ab", "tok_ratio_ba", "reason_ab", "reason_ba"]
    out = out[cols]
    out.to_csv(SCREEN / "medxpertqa_study20.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    print(out[["group", "pair_id", "correct_text", "distractor_text", "share_correct_ab", "share_correct_ba", "nothink_gap_ab",
               "nothink_gap_ba", "tok_p50_ab", "tok_p50_ba", "tok_ratio_ab", "tok_ratio_ba", "osteo"]].round(2).to_string(index=False))
    print(f"-> {SCREEN / 'medxpertqa_study20.csv'} ({len(out)} pairs, {out.source_id.nunique()} questions, seed {SEED})")


if __name__ == "__main__":
    main()
