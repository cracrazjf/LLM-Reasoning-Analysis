"""Stop height of every trace of the main run, traces that end right against traces that end wrong.

    python src/plot_stop.py        # -> figures/fig7_stop_height.png

Stop height = the evidence for the answer the trace finally gives, read at the stop (the position
where the model wrote </think>, closed context): log p(chosen) - log p(other). It is positive when
the readout at the stop already favours the final answer, which is nearly always.
(a) pooled over the 24 prompts: distribution of the stop height for traces that end on the
    correct answer and for traces that end on the wrong one (densities, counts in the legend);
(b) per prompt: median and interquartile range of the stop height for right and wrong traces,
    and the level before any reasoning (empty-thinking readout, oriented toward the correct answer).
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
from generate import ROOT, iter_traces  # noqa: E402
from plot_traces import RUNS, STUDY, STYLE, load  # noqa: E402

matplotlib.use("Agg")

OUT = ROOT / "figures"
WHITE, LIGHT, MID, BLACK = "#ffffff", "#c8c8c8", "#8c8c8c", "#000000"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    plt.rcParams.update(STYLE)
    study = pd.read_csv(STUDY).sort_values(["group", "pair_id"]).reset_index(drop=True)

    rows = []
    for o, run in RUNS.items():
        traces, readouts, start = load(run)
        for r in readouts:
            t = traces[r["sample_id"]]
            if t["answer"] is None:
                continue
            i_e = r["kind"].index("e")
            x_e = r["closed"]["logp_A"][i_e] - r["closed"]["logp_B"][i_e]          # A minus B at the stop
            sign_chosen = 1.0 if t["answer"] == "A" else -1.0
            sign_correct = 1.0 if t["label"] == "A" else -1.0
            s = start[t["item_id"]]
            rows.append({"item_id": t["item_id"], "pair_id": t["pair_id"], "order": o, "correct": bool(t["correct"]),
                         "height": sign_chosen * x_e, "x_stop_correct": sign_correct * x_e,
                         "start_correct": sign_correct * (s["logp_A"] - s["logp_B"]), "n_sentences": sum(k == "s" for k in r["kind"])})
    d = pd.DataFrame(rows)
    right, wrong = d[d.correct], d[~d.correct]
    q = lambda s: (float(s.median()), float(s.quantile(0.25)), float(s.quantile(0.75)))  # noqa: E731
    stats = {"n_right": int(len(right)), "n_wrong": int(len(wrong)),
             "height_right_median_q1_q3": q(right.height), "height_wrong_median_q1_q3": q(wrong.height),
             "share_height_below_0_right": float((right.height < 0).mean()), "share_height_below_0_wrong": float((wrong.height < 0).mean()),
             "share_height_below_2_right": float((right.height < 2).mean()), "share_height_below_2_wrong": float((wrong.height < 2).mean())}

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.6]})
    bins = np.arange(-6, 31, 1.0)
    a.hist(right.height, bins=bins, density=True, facecolor=LIGHT, edgecolor=BLACK, linewidth=0.5,
           label=f"ends right (n = {len(right)})")
    a.hist(wrong.height, bins=bins, density=True, histtype="step", edgecolor=BLACK, linewidth=1.0, hatch="////",
           label=f"ends wrong (n = {len(wrong)})")
    for s, ls in ((right.height, (0, (4, 2))), (wrong.height, (0, (1, 2)))):
        a.axvline(s.median(), color=BLACK, linewidth=0.7, linestyle=ls)
    a.set_xlabel("Stop height: log odds for the chosen answer at </think>")
    a.set_ylabel("Share of traces per nat")
    a.set_xlim(-6, 30)
    a.legend(loc="upper left", fontsize=6)
    a.text(0.98, 0.97, f"medians {right.height.median():.1f} / {wrong.height.median():.1f}", transform=a.transAxes, ha="right", va="top", fontsize=6.5)
    a.set_title("a", loc="left", fontweight="bold")

    per = []
    for _, srow in study.iterrows():
        for o in ("ab", "ba"):
            g = d[(d.pair_id == srow.pair_id) & (d.order == o)]
            gr, gw = g[g.correct].height, g[~g.correct].height
            per.append({"label": f"{srow.source_id} {'A' if o == 'ab' else 'B'}", "group": srow.group, "pair_id": srow.pair_id, "order": o,
                        "start": float(g.start_correct.iloc[0]),
                        "right": q(gr) if len(gr) else None, "n_right": int(len(gr)), "wrong": q(gw) if len(gw) else None, "n_wrong": int(len(gw))})
    xs = np.arange(len(per))
    for i, p in enumerate(per):
        if p["right"]:
            m, lo, hi = p["right"]
            b.errorbar(i - 0.12, m, yerr=[[m - lo], [hi - m]], fmt="o", color=MID, markeredgecolor=BLACK, markersize=3.2, elinewidth=0.6, capsize=1.5, markeredgewidth=0.5)
        if p["wrong"]:
            m, lo, hi = p["wrong"]
            b.errorbar(i + 0.12, m, yerr=[[m - lo], [hi - m]], fmt="s", color=BLACK, markersize=3.0, elinewidth=0.6, capsize=1.5)
        b.plot(i, abs(p["start"]), marker="_", color=BLACK, markersize=7, markeredgewidth=0.8, linestyle="none")
    n_a = sum(p["group"] == "A" for p in per)
    b.axvline(n_a - 0.5, color=BLACK, linewidth=0.5)
    b.axhline(0, color=BLACK, linewidth=0.4, linestyle=(0, (3, 2)))
    b.set_xticks(xs, [p["label"] for p in per], rotation=90, fontsize=5.8)
    b.set_ylabel("Stop height (median, IQR)")
    b.set_ylim(-2, 30)
    b.text(n_a / 2 - 0.5, 29, "Group A: first reaction wrong", ha="center", va="top", fontsize=6.5)
    b.text(n_a + (len(per) - n_a) / 2 - 0.5, 29, "Group B: first reaction right", ha="center", va="top", fontsize=6.5)
    from matplotlib.lines import Line2D
    b.legend(handles=[Line2D([], [], marker="o", color=MID, markeredgecolor=BLACK, markersize=3.5, linestyle="none", label="ends right"),
                      Line2D([], [], marker="s", color=BLACK, markersize=3.2, linestyle="none", label="ends wrong"),
                      Line2D([], [], marker="_", color=BLACK, markersize=7, markeredgewidth=0.8, linestyle="none", label="|level before reasoning|")],
             loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=6)
    b.set_title("b", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.5)
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png",):
        fig.savefig(args.out / f"fig7_stop_height.{ext}")
    print(json.dumps({k: v for k, v in stats.items() if k != "per_prompt"}, indent=1))
    print(f"-> {args.out / 'fig7_stop_height.png'}")


if __name__ == "__main__":
    main()
