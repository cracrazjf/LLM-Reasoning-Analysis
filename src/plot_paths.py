"""Figures of the readout paths: one overview and one panel per prompt.

    python src/plot_paths.py --run runs/medxpertqa/main-qwen3-8b [--cond baseline] [--traces 12]

Uses <run>/analysis/paths.npz (src/paths.py). X_c is the forced-answer log-odds toward the correct
option, read after every sentence.

fig_path_structure.png
  a  X toward the letter of a trace's first verdict, from four sentences before it to six after
  b  mean X_c over the first 40 sentences of the review phase, per condition
  c  share of states whose next sentence is the first verdict, by thinking length, for low,
     middle and high |X| (thirds within the length bin)
  d  share of verdicts after which the thinking ends, by |X| after the verdict, for four ranges
     of thinking length
paths_by_prompt_<cond>_pN.png
  per prompt (pair x order): X_c against thinking tokens for a sample of traces, coloured by the
  final answer (correct / wrong); the dot marks the first verdict, the square at 0 the readout
  with empty thinking
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from path_analysis import CONDS, Paths  # noqa: E402

COND_COL = {"speed": "#e76f51", "baseline": "#264653", "careful": "#2a9d8f"}
COND_STYLE = {"speed": ":", "baseline": "-", "careful": "--"}
OUTCOME_COL = {True: "#2a9d8f", False: "#e76f51"}
RAMP = ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"]   # one hue, light to dark: ordered groups
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax: plt.Axes) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def overview(P: Paths, out: Path) -> None:
    X, t, k, n, off, tr, fv = P.X, P.t, P.k, P.n, P.off, P.tr, P.fv
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    # a
    ax = axes[0, 0]
    firsts = np.array([off[i] + fv[i] for i in range(len(n)) if 4 <= fv[i] < n[i] - 6])
    lags = np.arange(-4, 7)
    ax.plot(lags, [np.nanmean(X[firsts + j] * P.vs[firsts]) for j in lags], color=COND_COL["baseline"], lw=2, marker="o", ms=5)
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("sentences from the first verdict (0 = the verdict sentence)")
    ax.set_ylabel("X toward the verdict's letter (nats)")
    ax.set_title("a  The first verdict moves X; little sign of it before", loc="left", fontsize=10, color=INK)
    # b
    ax = axes[0, 1]
    for c in [c for c in CONDS if c in set(P.T.cond)]:
        idx = [i for i in np.where(P.T.cond.values == c)[0] if fv[i] >= 40]
        m = np.mean([X[off[i]:off[i] + 40] * P.lab[i] for i in idx], axis=0)
        ax.plot(np.arange(1, 41), m, color=COND_COL[c], ls=COND_STYLE[c], lw=2, label=f"{c} ({len(idx)} traces)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlim(1, 40)
    ax.set_xlabel("sentence of the review phase")
    ax.set_ylabel("mean X toward the correct option (nats)")
    ax.set_title("b  Before any verdict X drifts slowly", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="best")
    # c
    ax = axes[1, 0]
    rows = np.where((k < fv[tr]) & (k < n[tr] - 1) & ~np.isnan(X))[0]
    d = pd.DataFrame({"y": (k + 1 == fv[tr])[rows].astype(float), "x": np.abs(X[rows]),
                      "tb": pd.cut(t[rows], [0, 200, 400, 600, 800, 1000, 1300, 1700, 2500])})
    d["third"] = d.groupby("tb", observed=True).x.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False))
    for j, name in enumerate(("lowest third of |X|", "middle third", "highest third")):
        g = d[d.third == j].groupby("tb", observed=True).y.mean()
        mid = [iv.mid for iv in g.index]
        ax.plot(mid, g.values, color=RAMP[j + 1], lw=2, marker="o", ms=5, label=name)
    ax.set_xlabel("thinking tokens so far")
    ax.set_ylabel("share of states followed by the first verdict")
    ax.set_title("c  The first verdict follows length far more than |X|", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    # d
    ax = axes[1, 1]
    rows = np.where(P.isv & ~np.isnan(X) & P.ends_on_verdict[tr])[0]
    d = pd.DataFrame({"y": (k == P.lv[tr])[rows].astype(float), "xb": pd.cut(np.abs(X[rows]), [0, 2, 4, 6, 8, 10, 12, 30], right=False),
                      "tb": pd.cut(t[rows], [0, 800, 1300, 2000, 100000], labels=["up to 800 tokens", "800 to 1,300", "1,300 to 2,000", "over 2,000"])})
    for j, (name, g) in enumerate(d.groupby("tb", observed=True)):
        r = g.groupby("xb", observed=True).y.mean()
        ax.plot([min(iv.mid, 14) for iv in r.index], r.values, color=RAMP[j], lw=2, marker="o", ms=5, label=str(name))
    ax.set_xlabel("|X| after the verdict (nats)")
    ax.set_ylabel("share of verdicts after which the thinking ends")
    ax.set_title("d  Stopping after a verdict follows |X|, not length", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left", title="thinking length so far", title_fontsize=7)
    for ax in axes.ravel():
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_path_structure.png", dpi=150)
    plt.close(fig)


def per_prompt(P: Paths, out: Path, cond: str, n_traces: int, per_page: int = 20) -> int:
    T, X, t, off, n, fv = P.T, P.X, P.t, P.off, P.n, P.fv
    rng = np.random.default_rng(0)
    prompts = sorted(T[T.cond == cond].item_id.unique())
    pages = [prompts[i:i + per_page] for i in range(0, len(prompts), per_page)]
    for pi, page in enumerate(pages, 1):
        fig, axes = plt.subplots(4, 5, figsize=(16, 11), sharey=True)
        for ax, item in zip(axes.ravel(), page):
            g = T[(T.cond == cond) & (T.item_id == item) & T.correct.notna()]
            idx = rng.choice(g.index.values, min(n_traces, len(g)), replace=False)
            for i in idx:
                xc = np.clip(X[off[i]:off[i] + n[i]] * P.lab[i], -16, 16)
                col = OUTCOME_COL[bool(T.correct.values[i])]
                ax.plot(t[off[i]:off[i] + n[i]], xc, color=col, lw=0.7, alpha=0.75)
                if fv[i] < n[i]:
                    ax.plot(t[off[i] + fv[i]], xc[fv[i]], "o", ms=3.5, color=col, mec="white", mew=0.5)
            ax.plot(0, np.clip(float(g.X0.iloc[0]) * P.lab[g.index[0]], -16, 16), "s", ms=5, color=INK, clip_on=False)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_xlim(0, np.percentile(g.think_tokens, 95))
            ax.set_ylim(-16.5, 16.5)
            ax.set_title(f"{item.split(':', 1)[1]}   correct {g.correct.astype(float).mean():.2f}", fontsize=8, color=INK, loc="left")
            style(ax)
        for ax in axes.ravel()[len(page):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("X toward correct (nats)", fontsize=8)
        handles = [plt.Line2D([], [], color=OUTCOME_COL[True], lw=1.5, label="trace that ends correct"),
                   plt.Line2D([], [], color=OUTCOME_COL[False], lw=1.5, label="trace that ends wrong"),
                   plt.Line2D([], [], marker="o", ls="", color=MUTED, label="first verdict"),
                   plt.Line2D([], [], marker="s", ls="", color=INK, label="readout with empty thinking")]
        fig.legend(handles=handles, loc="upper right", ncol=4, fontsize=8, frameon=False)
        fig.suptitle(f"{cond}: X after every sentence against thinking tokens, {n_traces} traces per prompt (page {pi} of {len(pages)})",
                     x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"paths_by_prompt_{cond}_p{pi}.png", dpi=110)
        plt.close(fig)
    return len(pages)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cond", default="baseline", help="condition of the per-prompt pages")
    ap.add_argument("--traces", type=int, default=12, help="traces drawn per prompt")
    args = ap.parse_args()
    P = Paths(args.run)
    out = args.run / "analysis"
    overview(P, out)
    pages = per_prompt(P, out, args.cond, args.traces)
    print(f"-> {out}/fig_path_structure.png, paths_by_prompt_{args.cond}_p1..{pages}.png")


if __name__ == "__main__":
    main()
