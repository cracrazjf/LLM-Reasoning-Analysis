"""Thinking gain per pair: accuracy with thinking minus accuracy without thinking.

    python src/thinking_gain.py --think runs/medxpertqa/screen-qwen3-8b --nothink runs/medxpertqa/screen-nothink-qwen3-8b

The thinking run is the screening (baseline condition, src/generate.py); the no-thinking run is
the same prompts under Qwen3's "/no_think" switch (selection medxpertqa_screen_nothink.json).
Accuracies are computed per prompt (pair x order) and per pair (both orders pooled, which
cancels the letter preference that the no-thinking answers show). A trace's answer is
re-derived from answer_text with generate.parse_answer.

gain = acc_think - acc_nothink per pair. Classes (thresholds in GAIN_CLASSES):
  helps     gain >= +0.3     the first reaction is wrong or uncertain and reasoning repairs it
  hurts     gain <= -0.3     the first reaction is right and reasoning talks itself out of it
  neutral   otherwise        reasoning does not change the outcome (both right, both wrong, or both coin flips)

Writes data/pairs/medxpertqa_gain.csv (one row per pair, with per-order columns), gain_summary.json
and fig_gain.png (acc_think vs acc_nothink). Pairs are nested in questions; the summary also
counts questions holding at least one pair of each class.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from generate import iter_traces, parse_answer  # noqa: E402
from medxpertqa_dataset import ROOT  # noqa: E402

PAIRS_DIR = ROOT / "data" / "pairs"
GAIN_CLASSES = {"helps": 0.3, "hurts": -0.3}


def prompt_accuracy(run: Path, condition: str | None) -> pd.DataFrame:
    """Per prompt: n answered, accuracy, share answering B, median thinking tokens."""
    rows = []
    for r in iter_traces(run, ("item_id", "pair_id", "source_id", "order", "label", "condition", "answer_text", "think_tokens")):
        if condition and r.get("condition") not in (None, condition):
            continue
        a, _ = parse_answer(r["answer_text"], ("A", "B"))
        if a is None:
            continue
        rows.append({"item_id": r["item_id"], "pair_id": r["pair_id"], "source_id": r["source_id"], "order": r["order"],
                     "correct": float(a == r["label"]), "is_B": float(a == "B"), "think": r["think_tokens"]})
    t = pd.DataFrame(rows)
    return t.groupby(["item_id", "pair_id", "source_id", "order"]).agg(n=("correct", "size"), acc=("correct", "mean"),
                                                                       share_B=("is_B", "mean"), think=("think", "median")).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--think", type=Path, default=ROOT / "runs" / "medxpertqa" / "screen-qwen3-8b")
    ap.add_argument("--nothink", type=Path, required=True)
    ap.add_argument("--think-condition", default="C0_baseline", help="condition field to keep in the thinking run, if present")
    ap.add_argument("--nothink-condition", default="C3_no_think")
    ap.add_argument("--out", type=Path, default=PAIRS_DIR)
    args = ap.parse_args()

    th = prompt_accuracy(args.think, args.think_condition)
    nt = prompt_accuracy(args.nothink, args.nothink_condition)
    m = th.merge(nt, on=["item_id", "pair_id", "source_id", "order"], suffixes=("_think", "_nothink"))
    if m.empty:
        raise SystemExit("no prompts in common between the two runs")
    m["gain_prompt"] = m.acc_think - m.acc_nothink
    # pair level: pool both orders (weights by n)
    def pooled(g: pd.DataFrame, col: str, n: str) -> float:
        return float((g[col] * g[n]).sum() / g[n].sum())
    rows = []
    for pid, g in m.groupby("pair_id"):
        row = {"pair_id": pid, "source_id": g.source_id.iloc[0], "n_orders": len(g),
               "n_think": int(g.n_think.sum()), "n_nothink": int(g.n_nothink.sum()),
               "acc_think": pooled(g, "acc_think", "n_think"), "acc_nothink": pooled(g, "acc_nothink", "n_nothink"),
               "shareB_nothink": pooled(g, "share_B_nothink", "n_nothink"), "think_tokens": float(g.think_think.median())}
        for _, r in g.iterrows():
            row[f"acc_think_{r.order}"] = r.acc_think
            row[f"acc_nothink_{r.order}"] = r.acc_nothink
        rows.append(row)
    P = pd.DataFrame(rows)
    P["gain"] = P.acc_think - P.acc_nothink
    P["cls"] = np.where(P.gain >= GAIN_CLASSES["helps"], "helps", np.where(P.gain <= GAIN_CLASSES["hurts"], "hurts", "neutral"))
    # within-order consistency: does thinking help in both orders? (guards against letter preference)
    both = P.get("acc_think_ab") - P.get("acc_nothink_ab") if "acc_think_ab" in P else None
    if "acc_think_ab" in P and "acc_think_ba" in P:
        P["gain_ab"] = P.acc_think_ab - P.acc_nothink_ab
        P["gain_ba"] = P.acc_think_ba - P.acc_nothink_ba
        P["helps_both_orders"] = (P.gain_ab >= 0.2) & (P.gain_ba >= 0.2)
        P["hurts_both_orders"] = (P.gain_ab <= -0.2) & (P.gain_ba <= -0.2)
    args.out.mkdir(parents=True, exist_ok=True)
    P.sort_values("gain", ascending=False).to_csv(args.out / "medxpertqa_gain.csv", index=False)
    summary = {
        "think_run": str(args.think), "nothink_run": str(args.nothink), "n_pairs": int(len(P)), "n_prompts": int(len(m)),
        "n_questions": int(P.source_id.nunique()),
        "acc_think_mean": float(P.acc_think.mean()), "acc_nothink_mean": float(P.acc_nothink.mean()),
        "gain_mean": float(P.gain.mean()), "gain_quantiles": {q: float(P.gain.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "classes": P.cls.value_counts().to_dict(),
        "helps_both_orders": int(P.get("helps_both_orders", pd.Series(dtype=bool)).sum()),
        "hurts_both_orders": int(P.get("hurts_both_orders", pd.Series(dtype=bool)).sum()),
        "questions_with_a_helps_pair": int(P[P.cls == "helps"].source_id.nunique()),
        "questions_with_a_hurts_pair": int(P[P.cls == "hurts"].source_id.nunique()),
        "nothink_shareB_mean": float(P.shareB_nothink.mean()),
        "nothink_prompt_acc_bimodal": {"share_prompts_acc<=0.1": float((m.acc_nothink <= 0.1).mean()),
                                       "share_prompts_acc>=0.9": float((m.acc_nothink >= 0.9).mean())},
        "spearman_think_vs_nothink_pairs": float(P.acc_think.corr(P.acc_nothink, method="spearman")),
    }
    (args.out / "gain_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    colours = P.cls.map({"helps": "#2a9d8f", "hurts": "#e76f51", "neutral": "#8d99ae"})
    ax.scatter(P.acc_nothink, P.acc_think, c=colours, s=14, alpha=0.75)
    ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls="--")
    ax.set_xlabel("accuracy without thinking (/no_think)")
    ax.set_ylabel("accuracy with thinking (screening)")
    ax.set_title(f"helps {summary['classes'].get('helps', 0)}, hurts {summary['classes'].get('hurts', 0)}, neutral {summary['classes'].get('neutral', 0)}", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "fig_gain.png", dpi=150)
    print(json.dumps(summary, indent=1))
    print(f"-> {args.out}/medxpertqa_gain.csv, gain_summary.json, fig_gain.png")


if __name__ == "__main__":
    main()
