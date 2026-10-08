"""Every sample's truncation readouts along its reasoning: logit of the chosen minus logit of the unchosen option.

    python src/plot_diff_traces.py        # -> figures/fig15_diff_trajectory.png

Same layout as fig8 (one panel per question, ordered by accuracy, one line per sample, shared axes),
but y = chosen minus unchosen in the closed context, i.e. the log odds for the answer the sample
finally gives. Blue = samples that end on the correct answer, orange = samples that end on the wrong
answer, so a wrong sample whose two logits rise together would show as a flat orange line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import ROOT  # noqa: E402
from plot_logits import series  # noqa: E402
from plot_traces import RUNS, STUDY, STYLE, load  # noqa: E402

matplotlib.use("Agg")

OUT = ROOT / "figures"
BLUE, BLUE_DARK, ORANGE, ORANGE_DARK = "#2a78d6", "#1c5cab", "#eb6834", "#b8431c"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--xmax", type=int, default=250)
    ap.add_argument("--ymin", type=float, default=-15.0)
    ap.add_argument("--ymax", type=float, default=30.0)
    args = ap.parse_args()
    plt.rcParams.update(STYLE)
    study = pd.read_csv(STUDY)
    data = {o: load(run) for o, run in RUNS.items()}
    src = dict(zip(study.pair_id, study.source_id))

    panels = []
    for p in study.pair_id:
        for o in ("ab", "ba"):
            traces, readouts, start = data[o]
            iid = f"medxpertqa:{p}:{o}"
            recs = [r for r in readouts if r["item_id"] == iid]
            label = traces[recs[0]["sample_id"]]["label"]
            rows = []
            for r in recs:
                t = traces[r["sample_id"]]
                if t["answer"] is None:
                    continue
                zc, zu = series(r, start[iid], t["answer"])
                rows.append((zc - zu, bool(t["correct"])))
            acc = float(np.mean([ok for _, ok in rows]))
            panels.append((acc, f"{src[p]} {label} · acc {acc:.2f}", rows))
    panels.sort(key=lambda t: t[0])

    fig, axes = plt.subplots(6, 4, figsize=(7.4, 8.6), sharex=True, sharey=True)
    for k, (acc, title, rows) in enumerate(panels):
        ax = axes[k // 4, k % 4]
        for diff, ok in sorted(rows, key=lambda t: t[1], reverse=True):      # wrong samples drawn on top
            ax.plot(np.arange(len(diff)), diff, color=BLUE if ok else ORANGE, linewidth=0.45, alpha=0.35 if ok else 0.75)
        ax.axhline(0, color="#000000", linewidth=0.4, linestyle=(0, (3, 2)))
        ax.set_title(title, loc="left", pad=2, fontsize=6.5)
        n_wrong = sum(not ok for _, ok in rows)
        ax.text(0.98, 0.96, "ends right", transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=BLUE_DARK, fontweight="bold")
        ax.text(0.98, 0.86, f"ends wrong ({n_wrong})", transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=ORANGE_DARK, fontweight="bold")
        ax.set_xlim(0, args.xmax)
        ax.set_ylim(args.ymin, args.ymax)
        ax.tick_params(labelsize=5.5, length=1.5)
    for row in axes:
        row[0].set_ylabel("chosen − unchosen", fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("sentences of thinking (0 = before reasoning)", fontsize=6)
    fig.suptitle("Logit difference trajectory, chosen − unchosen", x=0.01, ha="left", fontsize=8.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.6, w_pad=0.5)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "fig15_diff_trajectory.png")
    print(f"-> {args.out / 'fig15_diff_trajectory.png'}")


if __name__ == "__main__":
    main()
