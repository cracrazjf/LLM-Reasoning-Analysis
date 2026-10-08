"""Per question: how the chosen and the unchosen logit move along the reasoning (summary curves).

    python src/plot_logit_curves.py        # -> figures/fig13_logit_curves_stop.png and fig13_logit_curves_start.png

One panel per question (correct option shown as A or as B), traces that end on the correct answer;
the prompts with >= 10 traces that end wrong get extra panels built from those traces. In each
panel the raw logit in the closed context (thinking so far + </think>) of the option the trace
finally chooses (blue) and of the other option (orange): the median over traces as a line, the
interquartile range as a band, at every sentence position. Two alignments:
  stop   x = sentences before the stop (0 = the position where the model wrote </think>);
  start  x = sentence index from the beginning of the thinking (0 = the empty-thinking readout,
         before any reasoning).
A position is drawn when at least 20 traces reach it (8 for the wrong-trace panels).
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


def stack(curves: list[np.ndarray], align: str, span: int) -> np.ndarray:
    """Traces x positions matrix (NaN where a trace has no value)."""
    m = np.full((len(curves), span + 1), np.nan)
    for i, c in enumerate(curves):
        if align == "stop":
            n = min(len(c), span + 1)
            m[i, span + 1 - n:] = c[-n:]
        else:
            n = min(len(c), span + 1)
            m[i, :n] = c[:n]
    return m


def panel(ax, groups: list[tuple[np.ndarray, np.ndarray]], align: str, span: int, title: str, min_n: int = 20) -> None:
    zc = stack([c for c, _ in groups], align, span)
    zu = stack([u for _, u in groups], align, span)
    x = np.arange(-span, 1) if align == "stop" else np.arange(span + 1)
    for m, col, dark in ((zu, ORANGE, ORANGE_DARK), (zc, BLUE, BLUE_DARK)):
        n = np.sum(~np.isnan(m), axis=0)
        ok = n >= min_n
        med = np.nanmedian(np.where(ok, m, np.nan), axis=0) if ok.any() else np.full(span + 1, np.nan)
        q1 = np.nanpercentile(np.where(ok, m, np.nan), 25, axis=0) if ok.any() else med
        q3 = np.nanpercentile(np.where(ok, m, np.nan), 75, axis=0) if ok.any() else med
        ax.fill_between(x, q1, q3, facecolor=col, alpha=0.25, linewidth=0)
        ax.plot(x, med, color=dark, linewidth=1.1)
    ax.set_title(title, loc="left", pad=2, fontsize=6.5)
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(labelsize=5.5, length=1.5)
    ax.text(0.02, 0.96, "chosen", transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color=BLUE_DARK, fontweight="bold")
    ax.text(0.02, 0.86, "unchosen", transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color=ORANGE_DARK, fontweight="bold")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--span", type=int, default=40, help="sentences shown")
    ap.add_argument("--min-wrong", type=int, default=10)
    args = ap.parse_args()
    plt.rcParams.update(STYLE)
    study = pd.read_csv(STUDY).sort_values(["group", "pair_id"]).reset_index(drop=True)
    data = {o: load(run) for o, run in RUNS.items()}
    src = dict(zip(study.pair_id, study.source_id))

    groups: dict[str, dict] = {}
    for p in study.pair_id:
        for o in ("ab", "ba"):
            traces, readouts, start = data[o]
            iid = f"medxpertqa:{p}:{o}"
            recs = [r for r in readouts if r["item_id"] == iid]
            right, wrong = [], []
            for r in recs:
                t = traces[r["sample_id"]]
                if t["answer"] is None:
                    continue
                (right if t["correct"] else wrong).append(series(r, start[iid], t["answer"]))
            label = traces[recs[0]["sample_id"]]["label"]
            acc = len(right) / (len(right) + len(wrong))
            groups[f"{p}:{o}"] = {"right": right, "wrong": wrong, "title": f"{src[p]} {label} · acc {acc:.2f}"}
    wrong_keys = [k for k, g in groups.items() if len(g["wrong"]) >= args.min_wrong]

    for align in ("stop", "start"):
        n_rows = 6 + int(np.ceil(len(wrong_keys) / 4))
        fig, axes = plt.subplots(n_rows, 4, figsize=(7.4, 1.3 * n_rows + 0.7), sharey=True)
        k = 0
        for key, g in groups.items():
            panel(axes[k // 4, k % 4], g["right"], align, args.span, g["title"])
            k += 1
        for key in wrong_keys:
            panel(axes[k // 4, k % 4], groups[key]["wrong"], align, args.span, groups[key]["title"] + " · wrong traces", min_n=8)
            k += 1
        while k < n_rows * 4:
            axes[k // 4, k % 4].axis("off")
            k += 1
        for row in axes:
            row[0].set_ylabel("raw logit", fontsize=6)
        axes[-1][0].set_xlabel("sentences before the stop" if align == "stop" else "sentence index from the start (0 = before reasoning)", fontsize=6)
        fig.suptitle("Chosen and unchosen logit along the reasoning (closed context), median and interquartile range over traces, "
                     + ("aligned to the stop" if align == "stop" else "aligned to the start"), x=0.01, ha="left", fontsize=8.5, fontweight="semibold")
        fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.6, w_pad=0.5)
        args.out.mkdir(parents=True, exist_ok=True)
        for ext in ("png",):
            fig.savefig(args.out / f"fig13_logit_curves_{align}.{ext}")
        plt.close(fig)
        print(f"-> {args.out / f'fig13_logit_curves_{align}.png'}")


if __name__ == "__main__":
    main()
