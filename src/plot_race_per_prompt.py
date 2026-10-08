"""Per prompt: the three stop quantities (raw logits) centred within the prompt, so their widths compare (smoothed densities).

    python src/plot_race_per_prompt.py        # -> figures/fig12_race_per_prompt.png

For each prompt, the traces that end on the correct answer give three values at the stop: the raw
logit of the chosen option, the raw logit of the unchosen option, and their difference. Each is
centred on its own median within the prompt and the three are overlaid on one axis, so the panel
shows directly which quantity is pinned at the stop (race model: the chosen logit; diffusion: the
difference). The numbers in each panel are the three standard deviations. A second block holds
the prompts with at least 10 traces that end on the wrong answer, using those traces.
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
COL = {"chosen": "#2a78d6", "unchosen": "#eb6834", "diff": "#1baf7a"}
DARK = {"chosen": "#1c5cab", "unchosen": "#b8431c", "diff": "#117a55"}
NAME = {"chosen": "chosen", "unchosen": "unchosen", "diff": "difference"}
LABEL = {"chosen": "chosen (raw logit)", "unchosen": "unchosen (raw logit)", "diff": "chosen − unchosen"}


def panel(ax, g: pd.DataFrame, title: str, bins: np.ndarray) -> dict:
    from scipy.stats import gaussian_kde
    sds = {}
    grid = np.linspace(bins[0], bins[-1], 400)
    for k in ("chosen", "unchosen", "diff"):
        x = (g[k] - g[k].median()).to_numpy()
        sds[k] = float(g[k].std(ddof=1))
        dens = gaussian_kde(x, bw_method=0.35)(grid)    # smoothed density, same bandwidth rule for the three quantities
        ax.fill_between(grid, 0, dens, facecolor=COL[k], alpha=0.28, linewidth=0)
        ax.plot(grid, dens, color=DARK[k], linewidth=0.9)
    ax.set_ylim(0, None)
    ax.set_title(title, loc="left", pad=2, fontsize=6.5)
    tight = min(sds, key=sds.get)
    for j, k in enumerate(("chosen", "unchosen", "diff")):
        ax.text(0.98, 0.96 - 0.12 * j, f"{'>' if k == tight else ' '} {NAME[k]} {sds[k]:.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=5.4, family="monospace", color=DARK[k], fontweight="bold" if k == tight else "normal")
    ax.set_xlim(bins[0], bins[-1])
    ax.tick_params(labelsize=5.5, length=1.5)
    ax.set_yticks([])
    return sds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--min-wrong", type=int, default=10)
    args = ap.parse_args()
    plt.rcParams.update(STYLE)
    study = pd.read_csv(STUDY).sort_values(["group", "pair_id"]).reset_index(drop=True)

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
    bins = np.arange(-8, 8.5, 0.5)

    wrong_groups = [(p, o) for p in study.pair_id for o in ("ab", "ba")
                    if ((d.pair_id == p) & (d.order == o) & ~d.correct).sum() >= args.min_wrong]
    n_main, n_wrong = 24, len(wrong_groups)
    n_rows = 6 + int(np.ceil(n_wrong / 4))
    fig, axes = plt.subplots(n_rows, 4, figsize=(7.4, 1.25 * n_rows + 0.7))
    summary = {}
    src = dict(zip(study.pair_id, study.source_id))
    acc_of = {(p, o): d[(d.pair_id == p) & (d.order == o)].correct.mean() for p in study.pair_id for o in ("ab", "ba")}
    order_right = sorted(acc_of, key=acc_of.get)                      # accuracy, low to high
    order_wrong = sorted(wrong_groups, key=lambda po: acc_of[po])
    k = 0
    for (p, o) in order_right:
        g = d[(d.pair_id == p) & (d.order == o) & d.correct]
        ax = axes[k // 4, k % 4]
        summary[f"{p}:{o} right"] = panel(ax, g, f"{src[p]} {g.label.iloc[0]} · acc {acc_of[(p, o)]:.2f}", bins)
        k += 1
    for (p, o) in order_wrong:
        g = d[(d.pair_id == p) & (d.order == o) & ~d.correct]
        ax = axes[k // 4, k % 4]
        summary[f"{p}:{o} wrong"] = panel(ax, g, f"{src[p]} {g.label.iloc[0]} · acc {acc_of[(p, o)]:.2f} · wrong traces", bins)
        k += 1
    while k < n_rows * 4:
        axes[k // 4, k % 4].axis("off")
        k += 1
    axes[-1][0].set_xlabel("raw logit at the stop, centred on the question median", fontsize=6)
    tight = pd.Series({g: min(v, key=v.get) for g, v in summary.items()}).value_counts().to_dict()
    fig.suptitle("Raw logits at the stop: chosen option, unchosen option and their difference, centred within each question",
                 x=0.01, ha="left", fontsize=8.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.6, w_pad=0.5)
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png",):
        fig.savefig(args.out / f"fig12_race_per_prompt.{ext}")
    print("tightest counts:", tight)
    print(f"-> {args.out / 'fig12_race_per_prompt.png'}")


if __name__ == "__main__":
    main()
