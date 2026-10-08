"""Raw logits at the stop: the chosen option, the unchosen option, and their difference.

    python src/plot_stop_logits.py        # -> figures/fig9_stop_logits.png

At the stop (the position where the model wrote </think>, closed context) every trace has a raw
logit for the option it finally chooses and one for the other option. Top row: the pooled
distributions over the 24 prompts of the chosen logit, the unchosen logit and their difference,
traces that end right against traces that end wrong. Bottom row: the same three quantities per
prompt, median and interquartile range, right (grey circles) and wrong (black squares) traces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import ROOT  # noqa: E402
from plot_traces import RUNS, STUDY, STYLE, load  # noqa: E402

matplotlib.use("Agg")

OUT = ROOT / "figures"
WHITE, LIGHT, MID, BLACK = "#ffffff", "#c8c8c8", "#8c8c8c", "#000000"
QUANTITIES = [("chosen", "Raw logit of the chosen option at the stop"),
              ("unchosen", "Raw logit of the unchosen option at the stop"),
              ("diff", "Chosen minus unchosen at the stop")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--color", action="store_true", help="colour version: blue = ends right, orange = ends wrong (file suffix _color)")
    args = ap.parse_args()
    plt.rcParams.update(STYLE)
    # colours of the two outcomes: greys for print, blue / orange (validated categorical slots 1 and 2) with --color
    C_RIGHT, C_RIGHT_EDGE, C_WRONG = ("#2a78d6", "#1c5cab", "#eb6834") if args.color else (LIGHT, BLACK, BLACK)
    suffix = "_color" if args.color else ""
    study = pd.read_csv(STUDY).sort_values(["group", "pair_id"]).reset_index(drop=True)

    rows = []
    for o, run in RUNS.items():
        traces, readouts, _ = load(run)
        for r in readouts:
            t = traces[r["sample_id"]]
            if t["answer"] is None:
                continue
            i_e = r["kind"].index("e")
            zc = r["closed"][f"z_{t['answer']}"][i_e]
            zu = r["closed"]["z_B" if t["answer"] == "A" else "z_A"][i_e]
            rows.append({"pair_id": t["pair_id"], "order": o, "correct": bool(t["correct"]), "chosen": zc, "unchosen": zu, "diff": zc - zu,
                         "vocab_mean": r["closed"]["vocab_mean"][i_e], "vocab_lse": r["closed"]["vocab_lse"][i_e]})
    d = pd.DataFrame(rows)
    right, wrong = d[d.correct], d[~d.correct]
    q = lambda s: [float(s.median()), float(s.quantile(0.25)), float(s.quantile(0.75)), float(s.std(ddof=1)) if len(s) > 1 else None]  # noqa: E731
    pooled = {k: {"right_median_q1_q3_sd": q(right[k]), "wrong_median_q1_q3_sd": q(wrong[k])} for k, _ in QUANTITIES}
    pooled["n_right"], pooled["n_wrong"] = int(len(right)), int(len(wrong))

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.2), gridspec_kw={"height_ratios": [1, 1.15]})
    for (key, label), ax in zip(QUANTITIES, axes[0]):
        lo, hi = np.floor(d[key].quantile(0.002)) - 1, np.ceil(d[key].quantile(0.998)) + 1
        bins = np.arange(lo, hi + 1, 1.0)
        ax.hist(right[key], bins=bins, density=True, facecolor=C_RIGHT, edgecolor=C_RIGHT_EDGE if args.color else BLACK, linewidth=0.5,
                alpha=0.75 if args.color else 1.0, label=f"ends right (n = {len(right)})")
        ax.hist(wrong[key], bins=bins, density=True, histtype="step", edgecolor=C_WRONG, linewidth=1.1, hatch=None if args.color else "////",
                label=f"ends wrong (n = {len(wrong)})")
        ax.axvline(right[key].median(), color=C_RIGHT_EDGE if args.color else BLACK, linewidth=0.8, linestyle=(0, (4, 2)))
        ax.axvline(wrong[key].median(), color=C_WRONG, linewidth=0.8, linestyle=(0, (1, 2)))
        ax.set_xlabel(label, fontsize=6.5)
        ax.set_xlim(lo, hi)
        ax.text(0.98, 0.97, f"medians {right[key].median():.1f} / {wrong[key].median():.1f}\nSD {right[key].std(ddof=1):.1f} / {wrong[key].std(ddof=1):.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6)
    axes[0][0].set_ylabel("Share of traces per unit")
    axes[0][0].legend(loc="upper left", fontsize=5.5)
    for ax, letter in zip(axes[0], "abc"):
        ax.set_title(letter, loc="left", fontweight="bold")

    per = []
    for _, srow in study.iterrows():
        for o in ("ab", "ba"):
            g = d[(d.pair_id == srow.pair_id) & (d.order == o)]
            rec = {"label": f"{srow.source_id} {'A' if o == 'ab' else 'B'}", "group": srow.group, "pair_id": srow.pair_id, "order": o,
                   "n_right": int(g.correct.sum()), "n_wrong": int((~g.correct).sum())}
            for key, _ in QUANTITIES:
                rec[f"{key}_right"] = q(g[g.correct][key]) if rec["n_right"] else None
                rec[f"{key}_wrong"] = q(g[~g.correct][key]) if rec["n_wrong"] else None
            per.append(rec)
    n_a = sum(p["group"] == "A" for p in per)
    for (key, label), ax in zip(QUANTITIES, axes[1]):
        for i, p in enumerate(per):
            if p[f"{key}_right"]:
                m, lo, hi, _ = p[f"{key}_right"]
                ax.errorbar(i - 0.12, m, yerr=[[m - lo], [hi - m]], fmt="o", color=C_RIGHT if args.color else MID, markeredgecolor=C_RIGHT_EDGE if args.color else BLACK,
                            markersize=2.8, elinewidth=0.5, capsize=1.2, markeredgewidth=0.4)
            if p[f"{key}_wrong"]:
                m, lo, hi, _ = p[f"{key}_wrong"]
                ax.errorbar(i + 0.12, m, yerr=[[m - lo], [hi - m]], fmt="s", color=C_WRONG, markersize=2.6, elinewidth=0.5, capsize=1.2)
        ax.axvline(n_a - 0.5, color=BLACK, linewidth=0.5)
        ax.set_xticks(np.arange(len(per)), [p["label"] for p in per], rotation=90, fontsize=5.2)
        ax.set_ylabel(label.replace(" at the stop", "") + " (median, IQR)", fontsize=6.5)
        ax.tick_params(axis="x", length=0)
    for ax, letter in zip(axes[1], "def"):
        ax.set_title(letter, loc="left", fontweight="bold")
    from matplotlib.lines import Line2D
    axes[1][0].legend(handles=[Line2D([], [], marker="o", color=C_RIGHT if args.color else MID, markeredgecolor=C_RIGHT_EDGE if args.color else BLACK, markersize=3.2, linestyle="none", label="ends right"),
                               Line2D([], [], marker="s", color=C_WRONG, markersize=3, linestyle="none", label="ends wrong")],
                      loc="upper left", fontsize=5.5)
    fig.text(0.01, 0.995, "Group A (left of the line): no-thinking answer wrong.  Group B: no-thinking answer right.", ha="left", va="top", fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=1.2, w_pad=1.0)
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png",):
        fig.savefig(args.out / f"fig9_stop_logits{suffix}.{ext}")
    print(json.dumps(pooled, indent=1))
    print(f"-> {args.out / f'fig9_stop_logits{suffix}.png'}")


if __name__ == "__main__":
    main()
