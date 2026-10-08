"""Every sample's truncation readouts along its reasoning: the raw logit of the chosen and the unchosen option.

    python src/plot_logits.py        # -> figures/fig8_raw_logits.png

One panel per question (correct option shown as A or as B), one line per sample (100 per panel).
x = number of sentences of thinking kept before the readout (0 = the empty-thinking readout,
before any reasoning; the last point of a line = the stop, where the model wrote </think>).
y = raw logit in the closed context (thinking so far + </think>) of the option the sample finally
chooses (blue) and of the other option (orange). Panels are ordered by accuracy, low to high.
All panels share the same x and y ranges (--xmax, --ymin, --ymax).
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
from plot_traces import RUNS, STUDY, STYLE, load  # noqa: E402

matplotlib.use("Agg")

OUT = ROOT / "figures"
BLUE, BLUE_DARK, ORANGE, ORANGE_DARK = "#2a78d6", "#1c5cab", "#eb6834", "#b8431c"


def series(r: dict, start: dict, chosen: str) -> tuple[np.ndarray, np.ndarray]:
    other = "B" if chosen == "A" else "A"
    zc, zu = [start[f"z_{chosen}"]], [start[f"z_{other}"]]
    for k, a, b in zip(r["kind"], r["closed"][f"z_{chosen}"], r["closed"][f"z_{other}"]):
        if k in ("s", "e"):
            zc.append(a)
            zu.append(b)
    return np.array(zc), np.array(zu)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--xmax", type=int, default=250)
    ap.add_argument("--ymin", type=float, default=25.0)
    ap.add_argument("--ymax", type=float, default=65.0)
    args = ap.parse_args()
    plt.rcParams.update(STYLE)
    study = pd.read_csv(STUDY).sort_values(["group", "pair_id"]).reset_index(drop=True)
    data = {o: load(run) for o, run in RUNS.items()}
    src = dict(zip(study.pair_id, study.source_id))

    panels = []
    n_cut = 0
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
                rows.append((zc, zu))
                n_cut += len(zc) - 1 > args.xmax
            acc = float(np.mean([bool(traces[r["sample_id"]]["correct"]) for r in recs if traces[r["sample_id"]]["answer"] is not None]))
            panels.append((acc, f"{src[p]} {label} · acc {acc:.2f}", rows))
    panels.sort(key=lambda t: t[0])           # accuracy, low to high

    fig, axes = plt.subplots(6, 4, figsize=(7.4, 8.6), sharex=True, sharey=True)
    for k, (acc, title, rows) in enumerate(panels):
        ax = axes[k // 4, k % 4]
        for zc, zu in rows:
            x = np.arange(len(zc))
            ax.plot(x, zu, color=ORANGE, linewidth=0.45, alpha=0.4)
            ax.plot(x, zc, color=BLUE, linewidth=0.45, alpha=0.4)
        ax.set_title(title, loc="left", pad=2, fontsize=6.5)
        ax.text(0.98, 0.96, "chosen", transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=BLUE_DARK, fontweight="bold")
        ax.text(0.98, 0.86, "unchosen", transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=ORANGE_DARK, fontweight="bold")
        ax.set_xlim(0, args.xmax)
        ax.set_ylim(args.ymin, args.ymax)
        ax.tick_params(labelsize=5.5, length=1.5)
    for row in axes:
        row[0].set_ylabel("raw logit", fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("sentences of thinking (0 = before reasoning)", fontsize=6)
    fig.suptitle("Raw logit trajectory of chosen and unchosen", x=0.01, ha="left", fontsize=8.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.6, w_pad=0.5)
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png",):
        fig.savefig(args.out / f"fig8_raw_logits.{ext}")
    print(f"{n_cut} samples run past x = {args.xmax}")
    print(f"-> {args.out / 'fig8_raw_logits.png'}")


if __name__ == "__main__":
    main()
