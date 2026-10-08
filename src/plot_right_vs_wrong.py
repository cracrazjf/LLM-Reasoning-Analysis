"""Stop height of traces that end right against traces that end wrong, within the same question.

    python src/plot_right_vs_wrong.py        # -> figures/fig14_right_vs_wrong.png, data/screen/medxpertqa_study12_stop_right_vs_wrong.csv

Only the prompts with at least 10 traces ending on the wrong answer. Top row: chosen minus unchosen
at the stop (the log odds for the answer the trace gives); bottom row: the raw logit of the chosen
option at the stop. Smoothed densities, blue = ends right, orange = ends wrong, medians as dashed
lines, Mann-Whitney p in the panel. The CSV holds the medians, counts and p-values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import ROOT  # noqa: E402
from plot_traces import RUNS, STUDY, STYLE, load  # noqa: E402

matplotlib.use("Agg")

OUT = ROOT / "figures"
TABLE = ROOT / "data/screen/medxpertqa_study12_stop_right_vs_wrong.csv"
BLUE, BLUE_DARK, ORANGE, ORANGE_DARK = "#2a78d6", "#1c5cab", "#eb6834", "#b8431c"


def density(ax, x: np.ndarray, grid: np.ndarray, face: str, edge: str) -> None:
    dens = gaussian_kde(x, bw_method=0.35)(grid)
    ax.fill_between(grid, 0, dens, facecolor=face, alpha=0.3, linewidth=0)
    ax.plot(grid, dens, color=edge, linewidth=1.0)
    ax.axvline(np.median(x), color=edge, linewidth=0.8, linestyle=(0, (4, 2)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--min-wrong", type=int, default=10)
    ap.add_argument("--diff-only", action="store_true", help="one row: only chosen minus unchosen (fig16_stop_diff_right_vs_wrong.png)")
    args = ap.parse_args()
    plt.rcParams.update(STYLE)
    study = pd.read_csv(STUDY)
    src = dict(zip(study.pair_id, study.source_id))

    rows = []
    for o, run in RUNS.items():
        traces, readouts, _ = load(run)
        for r in readouts:
            t = traces[r["sample_id"]]
            if t["answer"] is None:
                continue
            i = r["kind"].index("e")
            zc = r["closed"][f"z_{t['answer']}"][i]
            zu = r["closed"]["z_B" if t["answer"] == "A" else "z_A"][i]
            rows.append({"pair_id": t["pair_id"], "order": o, "label": t["label"], "correct": bool(t["correct"]), "chosen": zc, "unchosen": zu, "diff": zc - zu})
    d = pd.DataFrame(rows)
    d["acc"] = d.groupby(["pair_id", "order"]).correct.transform("mean")
    keys = sorted({(p, o) for p, o in zip(d.pair_id, d.order) if ((d.pair_id == p) & (d.order == o) & ~d.correct).sum() >= args.min_wrong},
                  key=lambda po: d[(d.pair_id == po[0]) & (d.order == po[1])].acc.iloc[0])

    table = []
    quantities = (("diff", -4, 22, "chosen − unchosen at the stop"), ("chosen", 30, 62, "chosen raw logit at the stop"))
    if args.diff_only:
        quantities = quantities[:1]
    fig, axes = plt.subplots(len(quantities), len(keys), figsize=(1.45 * len(keys) + 0.6, 1.9 * len(quantities) + 0.6), sharey="row", squeeze=False)
    for j, (p, o) in enumerate(keys):
        g = d[(d.pair_id == p) & (d.order == o)]
        right, wrong = g[g.correct], g[~g.correct]
        label = g.label.iloc[0]
        rec = {"prompt": f"{src[p]} {label}", "pair_id": p, "correct_option": label, "accuracy": round(float(g.acc.iloc[0]), 2), "n_right": len(right), "n_wrong": len(wrong)}
        for i, (key, lo, hi, name) in enumerate(quantities):
            ax = axes[i, j]
            grid = np.linspace(lo, hi, 400)
            density(ax, right[key].to_numpy(), grid, BLUE, BLUE_DARK)
            density(ax, wrong[key].to_numpy(), grid, ORANGE, ORANGE_DARK)
            pval = mannwhitneyu(right[key], wrong[key]).pvalue
            ax.set_xlim(lo, hi)
            ax.set_yticks([])
            ax.tick_params(labelsize=5.5, length=1.5)
            ax.text(0.03, 0.96, f"right {right[key].median():.1f}", transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color=BLUE_DARK, fontweight="bold")
            ax.text(0.03, 0.85, f"wrong {wrong[key].median():.1f}", transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color=ORANGE_DARK, fontweight="bold")
            ax.text(0.03, 0.74, f"p = {pval:.0e}" if pval < 1e-3 else f"p = {pval:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=5.4, color="#333333")
            if i == 0:
                ax.set_title(f"{src[p]} {label} · acc {g.acc.iloc[0]:.2f}", loc="left", pad=2, fontsize=6.5)
            if j == 0:
                ax.set_ylabel(name, fontsize=6)
            rec[f"{key}_median_right"] = round(float(right[key].median()), 2)
            rec[f"{key}_median_wrong"] = round(float(wrong[key].median()), 2)
            rec[f"{key}_p_mannwhitney"] = float(f"{pval:.2e}")
        rec["unchosen_median_right"] = round(float(right.unchosen.median()), 2)
        rec["unchosen_median_wrong"] = round(float(wrong.unchosen.median()), 2)
        table.append(rec)
    for ax in axes[0]:
        ax.set_xlabel("logit chosen - logit unchosen at the stop", fontsize=5.8)
    if len(quantities) > 1:
        for ax in axes[1]:
            ax.set_xlabel("raw logit", fontsize=5.8)
    fig.suptitle("Stop height of traces ending right (blue) and wrong (orange), same question", x=0.01, ha="left", fontsize=8.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.9 if args.diff_only else 0.95), w_pad=0.5, h_pad=1.0)
    args.out.mkdir(parents=True, exist_ok=True)
    name = "fig16_stop_diff_right_vs_wrong.png" if args.diff_only else "fig14_right_vs_wrong.png"
    fig.savefig(args.out / name)
    if not args.diff_only:
        pd.DataFrame(table).to_csv(TABLE, index=False)
        print(pd.DataFrame(table).to_string(index=False))
    print(f"-> {args.out / name}")


if __name__ == "__main__":
    main()
