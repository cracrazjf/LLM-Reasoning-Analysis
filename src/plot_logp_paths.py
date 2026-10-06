"""Trajectories of the chosen and of the unchosen option: log-probability or raw logit.

    python src/plot_logp_paths.py --run runs/medxpertqa/main-qwen3-8b [--value logp] [--cond baseline] [--traces 12]

Uses <run>/analysis/paths.npz (src/paths.py). After every sentence of the thinking the readout
gives the two letters' scores for the forced answer. Per trace, the chosen option is the letter the
model answered after </think>, the unchosen option is the other one.

--value  logp            log p of each letter; the two letters hold almost all the probability,
                         so the two curves mirror each other
         logit           raw logit of each letter (needs readout.py --logits); also drawn: the
                         level of the other stored candidates (log-sum-exp of their logits).
                         A logit has no fixed zero, so read differences between curves and
                         changes along a curve, not heights
         logit_vs_other  logit of each letter minus the level of the other candidates: the
                         log-odds of that letter against anything that is not A or B

fig_<value>_chosen_unchosen_<cond>.png   median and middle half (25th to 75th percentile) over traces
  a  all traces, position as a share of the trace's sentences
  b  the first 60 sentences, traces with at least 60 sentences; position 0 is the readout with
     empty thinking
  c  the last 40 sentences of the same traces, counted back from the stop
<value>_by_prompt_<cond>_pN.png   per prompt (pair x order): both curves against thinking tokens
  for a sample of traces (thin lines) and the median over all traces of the prompt per 100 tokens
  (thick lines, where at least 20 traces are still thinking)
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
from paths import load_paths  # noqa: E402

COL = {"chosen": "#2a78d6", "unchosen": "#eb6834", "other": "#8a8985"}
STYLE = {"chosen": "-", "unchosen": "--", "other": ":"}
NAME = {"chosen": "chosen option", "unchosen": "unchosen option", "other": "other candidates"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LONG, HEAD, TAIL, BIN = 60, 60, 40, 100
VALUES = {   # y label, axis limits (None: from the data), floor at which lower values are drawn
    "logp": ("log-probability of the option as the answer", (-16.0, 0.4), -16.0),
    "logit": ("raw logit", None, -np.inf),
    "logit_vs_other": ("logit minus the level of the other candidates", None, -np.inf),
}


def style(ax: plt.Axes) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


class Data:
    def __init__(self, run: Path, cond: str, value: str) -> None:
        T, A = load_paths(run)
        keep = ((T.cond == cond) & T.answer.notna()).values
        if value != "logp":
            if "z_A" not in A:
                raise SystemExit(f"no raw logits in {run}/analysis/paths.npz: run readout.py --logits, then paths.py")
            keep &= np.isfinite(A["z_A"][T.off.values])
        self.T = T[keep].reset_index(drop=True)
        self.n = T.n.values[keep]
        self.ylabel, self.ylim, self.floor = VALUES[value]
        is_a = (self.T.answer == "A").values
        sent_a = np.repeat(is_a, self.n)
        idx = np.concatenate([np.arange(o, o + m) for o, m in zip(T.off.values[keep], self.n)])
        self.off = np.r_[0, np.cumsum(self.n)[:-1]]
        self.t = A["t"][idx].astype(float)
        if value == "logp":
            a, b = A["logp_A"][idx].astype(float), A["logp_B"][idx].astype(float)
            # the readout with empty thinking: the two letters hold all the probability, so both follow from X0
            x0 = self.T.X0.astype(float).values
            a0, b0 = -np.logaddexp(0, -x0), -np.logaddexp(0, x0)
        else:
            ref = A["lse_other"][idx].astype(float) if value == "logit_vs_other" else 0.0
            ref0 = self.T.lse0_other.astype(float).values if value == "logit_vs_other" else 0.0
            a, b = A["z_A"][idx].astype(float) - ref, A["z_B"][idx].astype(float) - ref
            a0, b0 = self.T.z0_A.astype(float).values - ref0, self.T.z0_B.astype(float).values - ref0
        self.lp = {"chosen": np.where(sent_a, a, b), "unchosen": np.where(sent_a, b, a)}
        self.lp0 = {"chosen": np.where(is_a, a0, b0), "unchosen": np.where(is_a, b0, a0)}
        if value == "logit":
            self.lp["other"] = A["lse_other"][idx].astype(float)
            self.lp0["other"] = self.T.lse0_other.astype(float).values
        self.series = list(self.lp)


def band(ax: plt.Axes, x: np.ndarray, M: np.ndarray, name: str) -> None:
    lo, mid, hi = np.nanpercentile(M, [25, 50, 75], axis=0)
    ax.fill_between(x, lo, hi, color=COL[name], alpha=0.18, linewidth=0)
    ax.plot(x, mid, color=COL[name], ls=STYLE[name], lw=2, label=NAME[name])


def overview(D: Data, out: Path, cond: str, value: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    grid = np.linspace(0, 1, 51)
    long = np.where(D.n >= LONG)[0]
    for name in D.series:
        v = D.lp[name]
        rel = np.array([v[o + np.minimum((grid * (m - 1)).round().astype(int), m - 1)] for o, m in zip(D.off, D.n)])
        band(axes[0], grid * 100, rel, name)
        head = np.c_[D.lp0[name][long], np.array([v[D.off[i]:D.off[i] + HEAD] for i in long])]
        band(axes[1], np.arange(0, HEAD + 1), head, name)
        tail = np.array([v[D.off[i] + D.n[i] - TAIL:D.off[i] + D.n[i]] for i in long])
        band(axes[2], np.arange(-TAIL + 1, 1), tail, name)
    axes[0].set_title(f"a  All {len(D.n):,} traces, relative position", loc="left", fontsize=10, color=INK)
    axes[0].set_xlabel("position in the thinking (% of its sentences)")
    axes[1].set_title(f"b  From the start ({len(long):,} traces with {LONG}+ sentences)", loc="left", fontsize=10, color=INK)
    axes[1].set_xlabel("sentence (0 = empty thinking)")
    axes[2].set_title("c  The same traces, up to the stop", loc="left", fontsize=10, color=INK)
    axes[2].set_xlabel("sentences before the stop (0 = last sentence)")
    axes[0].set_ylabel(D.ylabel)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left" if value == "logp" else "best",
                   title="median, band = middle half of traces", title_fontsize=7)
    for ax in axes:
        if D.ylim:
            ax.set_ylim(*D.ylim)
        style(ax)
    fig.suptitle(f"{cond}: {D.ylabel} of the option the model ends up choosing and of the other one", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f"fig_{value}_chosen_unchosen_{cond}.png", dpi=150)
    plt.close(fig)


def per_prompt(D: Data, out: Path, cond: str, value: str, n_traces: int, per_page: int = 20) -> int:
    rng = np.random.default_rng(0)
    prompts = sorted(D.T.item_id.unique())
    pages = [prompts[i:i + per_page] for i in range(0, len(prompts), per_page)]
    for pi, page in enumerate(pages, 1):
        fig, axes = plt.subplots(4, 5, figsize=(16, 11), sharey=True)
        for ax, item in zip(axes.ravel(), page):
            rows = np.where((D.T.item_id == item).values)[0]
            xmax = np.percentile(D.T.think_tokens.values[rows], 95)
            for i in rng.choice(rows, min(n_traces, len(rows)), replace=False):
                s = slice(D.off[i], D.off[i] + D.n[i])
                for name in ("chosen", "unchosen"):
                    ax.plot(D.t[s], np.maximum(D.lp[name][s], D.floor), color=COL[name], lw=0.6, alpha=0.45)
            sent = np.concatenate([np.arange(D.off[i], D.off[i] + D.n[i]) for i in rows])
            trace = np.repeat(rows, D.n[rows])
            for name in D.series:
                d = pd.DataFrame({"trace": trace, "bin": (D.t[sent] // BIN).astype(int), "v": D.lp[name][sent]})
                last = d.groupby(["trace", "bin"]).v.last().reset_index()           # the state at the end of each 100-token bin
                g = last.groupby("bin").v.agg(["median", "size"])
                g = g[g["size"] >= 20]
                ax.plot((g.index + 1) * BIN, np.maximum(g["median"], D.floor), color=COL[name], ls=STYLE[name], lw=2.2)
                ax.plot(0, max(np.median(D.lp0[name][rows]), D.floor), "s", ms=4.5, color=COL[name], clip_on=False)
            share_b = (D.T.answer.values[rows] == "B").mean()
            ax.set_xlim(0, xmax)
            if D.ylim:
                ax.set_ylim(*D.ylim)
            ax.set_title(f"{item.split(':', 1)[1]}   correct {D.T.correct.values[rows].astype(float).mean():.2f}, chose B {share_b:.2f}",
                         fontsize=8, color=INK, loc="left")
            style(ax)
        for ax in axes.ravel()[len(page):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel(D.ylabel if value != "logit_vs_other" else "logit minus other candidates", fontsize=8)
        for ax in axes[-1, :]:
            ax.set_xlabel("thinking tokens", fontsize=8)
        handles = [plt.Line2D([], [], color=COL[k], ls=STYLE[k], lw=2.2, label=f"{NAME[k]}, median of the prompt's traces") for k in D.series]
        handles += [plt.Line2D([], [], color=MUTED, lw=0.7, label=f"single traces ({n_traces} per prompt)"),
                    plt.Line2D([], [], marker="s", ls="", color=MUTED, label="empty thinking")]
        fig.legend(handles=handles, loc="upper right", ncol=4, fontsize=8, frameon=False)
        fig.suptitle(f"{cond}: {D.ylabel} of the chosen and the unchosen option (page {pi} of {len(pages)})", x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"{value}_by_prompt_{cond}_p{pi}.png", dpi=110)
        plt.close(fig)
    return len(pages)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--value", default="logp", choices=list(VALUES))
    ap.add_argument("--cond", default="baseline")
    ap.add_argument("--traces", type=int, default=12, help="single traces drawn per prompt")
    args = ap.parse_args()
    D = Data(args.run, args.cond, args.value)
    out = args.run / "analysis"
    overview(D, out, args.cond, args.value)
    pages = per_prompt(D, out, args.cond, args.value, args.traces)
    print(f"{len(D.n)} traces -> {out}/fig_{args.value}_chosen_unchosen_{args.cond}.png, {args.value}_by_prompt_{args.cond}_p1..{pages}.png")


if __name__ == "__main__":
    main()
