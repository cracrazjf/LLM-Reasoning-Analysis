"""Per prompt: the distributions at the stop of the chosen logit, the unchosen logit and their difference.

    python src/plot_stop_logits_per_prompt.py        # -> figures/fig10_stop_logits_per_prompt.png

One row per pair, two blocks per row (correct option shown as A, shown as B). In each block the left
panel holds the raw logits at the stop (closed context): the chosen option as a filled histogram,
the unchosen option as an outline; the right panel holds chosen minus unchosen. Blue = traces that
end on the correct answer, orange = traces that end on the wrong answer. Counts of traces, 100 per
prompt.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
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
            i_e = r["kind"].index("e")
            zc = r["closed"][f"z_{t['answer']}"][i_e]
            zu = r["closed"]["z_B" if t["answer"] == "A" else "z_A"][i_e]
            rows.append({"pair_id": t["pair_id"], "order": o, "label": t["label"], "correct": bool(t["correct"]), "chosen": zc, "unchosen": zu, "diff": zc - zu})
    d = pd.DataFrame(rows)
    zbins = np.arange(24, 66, 1.0)
    dbins = np.arange(-6, 32, 1.0)

    n = len(study)
    fig, axes = plt.subplots(n, 4, figsize=(7.4, 1.05 * n + 0.6), gridspec_kw={"width_ratios": [1.3, 1, 1.3, 1]})
    for i, srow in study.iterrows():
        for j, o in enumerate(("ab", "ba")):
            g = d[(d.pair_id == srow.pair_id) & (d.order == o)]
            right, wrong = g[g.correct], g[~g.correct]
            axz, axd = axes[i, 2 * j], axes[i, 2 * j + 1]
            # raw logits: chosen filled, unchosen outline; blue right, orange wrong
            axz.hist(right.chosen, bins=zbins, facecolor=BLUE, edgecolor=BLUE_DARK, linewidth=0.4, alpha=0.7)
            axz.hist(right.unchosen, bins=zbins, histtype="step", edgecolor=BLUE_DARK, linewidth=0.9)
            if len(wrong):
                axz.hist(wrong.chosen, bins=zbins, facecolor=ORANGE, edgecolor=ORANGE_DARK, linewidth=0.4, alpha=0.7)
                axz.hist(wrong.unchosen, bins=zbins, histtype="step", edgecolor=ORANGE_DARK, linewidth=0.9)
            axz.set_xlim(zbins[0], zbins[-1])
            axd.hist(right["diff"], bins=dbins, facecolor=BLUE, edgecolor=BLUE_DARK, linewidth=0.4, alpha=0.7)
            if len(wrong):
                axd.hist(wrong["diff"], bins=dbins, facecolor=ORANGE, edgecolor=ORANGE_DARK, linewidth=0.4, alpha=0.7)
            axd.set_xlim(dbins[0], dbins[-1])
            axd.axvline(0, color="#000000", linewidth=0.4, linestyle=(0, (3, 2)))
            label = g.label.iloc[0]
            axz.set_title(f"{srow.source_id} · correct = {label} · {len(right)} right / {len(wrong)} wrong", loc="left", pad=1.5, fontsize=6.3)
            txt = f"chosen {right.chosen.median():.1f}, unchosen {right.unchosen.median():.1f}"
            if len(wrong) >= 3:
                txt += f"\nwrong: {wrong.chosen.median():.1f} / {wrong.unchosen.median():.1f}"
            axz.text(0.02, 0.95, txt, transform=axz.transAxes, ha="left", va="top", fontsize=5.2)
            txt2 = f"diff {right['diff'].median():.1f}"
            if len(wrong) >= 3:
                txt2 += f"\nwrong {wrong['diff'].median():.1f}"
            axd.text(0.98, 0.95, txt2, transform=axd.transAxes, ha="right", va="top", fontsize=5.2)
            for ax in (axz, axd):
                ax.tick_params(labelsize=5.2, length=1.5)
                if i < n - 1:
                    ax.set_xticklabels([])
            if j == 0:
                axz.set_ylabel("traces", fontsize=5.5)
        if i == n - 1:
            for j in range(2):
                axes[i, 2 * j].set_xlabel("raw logit at the stop", fontsize=6)
                axes[i, 2 * j + 1].set_xlabel("chosen minus unchosen", fontsize=6)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend(handles=[Patch(facecolor=BLUE, edgecolor=BLUE_DARK, alpha=0.7, label="chosen, ends right"),
                        Line2D([], [], color=BLUE_DARK, linewidth=1, label="unchosen, ends right"),
                        Patch(facecolor=ORANGE, edgecolor=ORANGE_DARK, alpha=0.7, label="chosen, ends wrong"),
                        Line2D([], [], color=ORANGE_DARK, linewidth=1, label="unchosen, ends wrong")],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=6.5)
    fig.text(0.01, 0.995, "Rows 1-8: group A (no-thinking answer wrong); rows 9-12: group B (right). Left block: correct option shown as A; right block: shown as B.",
             ha="left", va="top", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.015, 1, 0.985), h_pad=0.5, w_pad=0.5)
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png",):
        fig.savefig(args.out / f"fig10_stop_logits_per_prompt.{ext}")
    print(f"-> {args.out / 'fig10_stop_logits_per_prompt.png'}")


if __name__ == "__main__":
    main()
