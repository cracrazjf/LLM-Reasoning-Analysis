"""One trace per panel: the chosen and the unchosen option's logit after every sentence.

    python src/analysis/plot_samples.py --run runs/medxpertqa/main-qwen3-8b --out figures/trajectories \\
        --prompts Text-1074:A-B:ab Text-1338:A-G:ba --per-prompt 3

Uses <run>/analysis/paths.npz (src/paths.py). For every listed prompt (pair id and order) the
first --per-prompt traces are drawn, alternating correct and wrong answers where both exist. In a
panel: raw logit of the chosen option (the letter answered after </think>) and of the unchosen one
against thinking tokens; a dot marks a verdict sentence (one that states an answer letter) and the
last point is the stop. Writes <out>/samples_<cond>.pdf and .png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from paths import Paths  # noqa: E402
from analysis.plot_logp_paths import COL, INK, MUTED  # noqa: E402
from analysis.stop_line import PAPER_RC  # noqa: E402


def pick(P: Paths, item: str, cond: str, n: int) -> list[int]:
    T = P.T
    rows = np.where((T.item_id == f"medxpertqa:{item}").values & (T.cond == cond).values & T.answer.notna().values)[0]
    right = [i for i in rows if T.correct.values[i]]
    wrong = [i for i in rows if not T.correct.values[i]]
    out: list[int] = []
    while len(out) < n and (right or wrong):
        for pool in (right, wrong):
            if pool and len(out) < n:
                out.append(pool.pop(0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cond", default="baseline")
    ap.add_argument("--prompts", nargs="+", required=True, help="pair id and order, e.g. Text-1074:A-B:ab")
    ap.add_argument("--per-prompt", type=int, default=3)
    ap.add_argument("--name", default="samples")
    args = ap.parse_args()
    P = Paths(args.run)
    A = dict(np.load(args.run / "analysis" / "paths.npz"))
    t, za, zb, V = A["t"].astype(float), A["z_A"].astype(float), A["z_B"].astype(float), A["verdict"]
    chosen_rows = [(item, i) for item in args.prompts for i in pick(P, item, args.cond, args.per_prompt)]
    cols = args.per_prompt
    rows = int(np.ceil(len(chosen_rows) / cols))
    args.out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(rows, cols, figsize=(7.2, 1.9 * rows + 0.5), sharey=True, squeeze=False)
        for ax, (item, i) in zip(axes.ravel(), chosen_rows):
            s = slice(P.off[i], P.off[i] + P.n[i])
            ans = P.T.answer.values[i]
            c, u = (za[s], zb[s]) if ans == "A" else (zb[s], za[s])
            x = t[s]
            ax.plot(x, u, color=COL["unchosen"], lw=1.0, ls="--", label="unchosen option")
            ax.plot(x, c, color=COL["chosen"], lw=1.0, label="chosen option")
            v = V[s] > 0
            ax.plot(x[v], c[v], "o", ms=2.6, color=COL["chosen"], mec="white", mew=0.4, label="verdict sentence")
            ax.plot(x[-1], c[-1], "s", ms=4, color=INK, label="stop")
            pair, order = item.rsplit(":", 1)
            ok = "correct" if P.T.correct.values[i] else "wrong"
            k = P.T.sample_id.values[i].rsplit("#", 1)[1]
            ax.set_title(f"{pair} ({order}), sample {k}\nanswered {ans} ({ok}), {int(x[-1]):,} tokens", loc="left", pad=2, fontsize=6.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction="out")
        for ax in axes.ravel()[len(chosen_rows):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("raw logit")
        for ax in axes[-1, :]:
            ax.set_xlabel("thinking tokens")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        for ext in ("pdf", "png"):
            fig.savefig(args.out / f"{args.name}_{args.cond}.{ext}")
        plt.close(fig)
    print(f"{len(chosen_rows)} traces -> {args.out}/{args.name}_{args.cond}.pdf, .png")


if __name__ == "__main__":
    main()
