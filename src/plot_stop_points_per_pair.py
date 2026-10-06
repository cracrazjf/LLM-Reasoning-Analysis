"""One panel per pair (or per prompt): every trace's stop point and that panel's own fitted lines.

    python src/plot_stop_points_per_pair.py --run runs/medxpertqa/main-qwen3-8b --cond baseline            # 70 pairs
    python src/plot_stop_points_per_pair.py --run runs/medxpertqa/main-qwen3-8b --cond baseline --by prompt # 140 prompts

x = thinking tokens, y = evidence at the stop signed toward the correct option (correct traces
above zero, wrong ones below). Per panel and outcome (>= 15 traces): the flat line a fixed boundary
would give (dashed, the mean) and a regression of the height on log thinking length (solid).
With --by pair both option orders share a panel (circle: the correct option is listed as A;
triangle: as B); with --by prompt each order has its own panel. Panels are sorted by accuracy.
Writes <run>/analysis/stop_by_<pair|prompt>_<cond>_pN.png and stop_by_pair.csv / stop_by_prompt.csv
(per unit x condition x outcome: n, mean, sd, cv, slope on log t, correlation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from medxpertqa_dataset import ROOT  # noqa: E402
from stop_height import ORDER, load  # noqa: E402

COL = {1.0: "#2a9d8f", 0.0: "#e76f51"}


def table(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    rows = []
    for (u, cond, corr), g in df.groupby([unit, "cond", "correct"]):
        r = {unit: u, "cond": cond, "outcome": "correct" if corr == 1 else "wrong", "n": len(g),
             "mean": g.H.mean(), "sd": g.H.std(), "cv": g.H.std() / g.H.mean() if len(g) > 2 else np.nan,
             "p10": g.H.quantile(.1), "p90": g.H.quantile(.9)}
        if len(g) >= 15 and g.logt.std() > 0:
            b, a = np.polyfit(g.logt, g.H, 1)
            r.update({"slope_logt": b, "intercept": a, "corr_logt": float(np.corrcoef(g.logt, g.H)[0, 1])})
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cond", default="baseline", choices=ORDER)
    ap.add_argument("--by", default="pair", choices=["pair", "prompt"])
    ap.add_argument("--per-page", type=int, default=20)
    args = ap.parse_args()
    df = load(args.run)
    df["order"] = df.item_id.str[-2:]
    df["pair_id"] = df.item_id.str.replace("medxpertqa:", "", regex=False).str.rsplit(":", n=1).str[0]
    df["prompt"] = df.pair_id + ":" + df.order
    unit = "pair_id" if args.by == "pair" else "prompt"
    out = args.run / "analysis"
    tab = table(df, unit)
    tab.to_csv(out / f"stop_by_{args.by}.csv", index=False)
    names = pd.read_csv(ROOT / "data" / "pairs" / "medxpertqa_distance.csv").set_index("pair_id")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[df.cond == args.cond]
    acc = d.groupby(unit).correct.mean()
    pair_acc = d.groupby("pair_id").correct.mean()
    if args.by == "pair":
        units = list(acc.sort_values(ascending=False).index)
    else:  # keep the two orders of a pair next to each other, pairs sorted by accuracy
        units = [f"{p}:{o}" for p in pair_acc.sort_values(ascending=False).index for o in ("ab", "ba")]
    ncol = 5
    pages = [units[i:i + args.per_page] for i in range(0, len(units), args.per_page)]
    for pi, page in enumerate(pages, 1):
        nrow = int(np.ceil(len(page) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(17, 3.5 * nrow), squeeze=False)
        for ax in axes.flat[len(page):]:
            ax.axis("off")
        for ax, u in zip(axes.flat, page):
            g = d[d[unit] == u]
            pair = g.pair_id.iloc[0]
            for order, marker in (("ab", "o"), ("ba", "^")):  # ab: the correct option is listed as A; ba: as B
                go = g[g.order == order]
                if len(go):
                    ax.scatter(go.t, np.where(go.correct == 1, go.H, -go.H), s=11, alpha=0.45, c=go.correct.map(COL),
                               marker=marker, linewidths=0)
            notes = []
            for corr, sign in ((1.0, 1), (0.0, -1)):
                h = g[g.correct == corr]
                if len(h) < 15:
                    continue
                ts = np.linspace(max(h.t.quantile(.02), 0.2), h.t.quantile(.98), 50)
                ax.plot(ts, sign * np.full_like(ts, h.H.mean()), "--", color="black", lw=1.0)
                b, a = np.polyfit(h.logt, h.H, 1)
                ax.plot(ts, sign * (a + b * np.log(ts)), "-", color=COL[corr], lw=2.2)
                note = f"{'right' if corr == 1 else 'wrong'}: {h.H.mean():.1f}±{h.H.std():.1f}, r={np.corrcoef(h.logt, h.H)[0, 1]:+.2f}"
                if args.by == "pair":
                    m = [h[h.order == o].H.mean() for o in ("ab", "ba")]
                    note += f" (listed A/B: {m[0]:.0f}/{m[1]:.0f})"
                notes.append(note)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xlim(0, 4.5)
            ax.set_ylim(-24, 24)
            k = 24 if args.by == "pair" else 18
            nm = f"{names.loc[pair, 'correct_text'][:k]} vs {names.loc[pair, 'distractor_text'][:k]}"
            if args.by == "prompt":
                nm += f" [correct={'A' if g.order.iloc[0] == 'ab' else 'B'}]"
            ax.set_title(f"{nm}\nacc {acc[u]:.2f} | " + "\n".join(notes), fontsize=7.0)
            ax.tick_params(labelsize=7)
        fig.supxlabel("thinking tokens (thousands)", fontsize=10)
        fig.supylabel("evidence at the stop, toward the correct option (nats)", fontsize=10)
        legend = ("circle = correct option listed as A, triangle = as B; " if args.by == "pair" else "")
        fig.suptitle(f"{args.cond}: stop points per {args.by} (page {pi}/{len(pages)}; green = answered correctly, orange = wrong; "
                     f"{legend}dashed = flat line, solid = fit on log length)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"stop_by_{args.by}_{args.cond}_p{pi}.png", dpi=110)
        plt.close(fig)

    t = tab[(tab.cond == args.cond) & (tab.n >= 15)]
    for o in ("correct", "wrong"):
        x = t[t.outcome == o]
        print(f"{args.cond} by {args.by} {o:<7}: units with >=15 traces {len(x):>3} | stop level across units mean {x['mean'].mean():.1f} "
              f"(range {x['mean'].min():.1f}-{x['mean'].max():.1f}) | within-unit SD median {x['sd'].median():.2f}, CV median {x['cv'].median():.2f} | "
              f"corr with log length: median {x.corr_logt.median():+.2f}, negative in {int((x.corr_logt < 0).sum())}/{len(x)}, below -0.2 in {int((x.corr_logt < -0.2).sum())}")
    print(f"-> {len(pages)} pages: {out}/stop_by_{args.by}_{args.cond}_p*.png; table {out}/stop_by_{args.by}.csv")


if __name__ == "__main__":
    main()
