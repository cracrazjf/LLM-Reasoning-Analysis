"""Race model or relative evidence? Which quantity is pinned at the stop.

    python src/plot_race.py        # -> figures/fig11_race_check.png

At the stop every trace has a raw logit for the option it chooses, one for the other option, and
their difference. A race model stops when the chosen accumulator reaches its bound, so the chosen
logit should be the tight quantity and the unchosen one free to vary; a relative-evidence model
(drift diffusion) stops on the difference, so the difference should be the tight quantity.
(a) the three quantities centred on their median within each prompt (and outcome), pooled over the
    24 prompts: the widths compare directly;
(b) the within-prompt standard deviation of each quantity, one point per prompt;
(c) chosen against unchosen logit at the stop, centred per prompt: the two move together along a
    shared per-position offset (the diagonal), which is what the difference removes.
Groups with at least 10 traces are used (right-ending traces of every prompt, wrong-ending traces
of the six prompts with >= 10 errors).
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
COL = {"chosen": "#2a78d6", "unchosen": "#eb6834", "diff": "#1baf7a"}   # categorical slots 1-3 (validated all-pairs)
DARK = {"chosen": "#1c5cab", "unchosen": "#b8431c", "diff": "#117a55"}
NAME = {"chosen": "chosen option (raw logit)", "unchosen": "unchosen option (raw logit)", "diff": "chosen − unchosen"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--min-group", type=int, default=10)
    args = ap.parse_args()
    plt.rcParams.update(STYLE)
    study = pd.read_csv(STUDY).sort_values(["group", "pair_id"]).reset_index(drop=True)
    order_items = [f"medxpertqa:{p}:{o}" for p in study.pair_id for o in ("ab", "ba")]
    short = {f"medxpertqa:{p}:{o}": f"{s} {'A' if o == 'ab' else 'B'}" for p, s in zip(study.pair_id, study.source_id) for o in ("ab", "ba")}

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
            rows.append({"item": t["item_id"], "correct": bool(t["correct"]), "chosen": zc, "unchosen": zu, "diff": zc - zu})
    d = pd.DataFrame(rows)
    d["grp"] = d["item"] + np.where(d.correct, " right", " wrong")
    sizes = d.groupby("grp").size()
    d = d[d.grp.isin(sizes[sizes >= args.min_group].index)].copy()
    for k in ("chosen", "unchosen", "diff"):
        d[f"{k}_c"] = d[k] - d.groupby("grp")[k].transform("median")
    sd = d.groupby("grp")[["chosen", "unchosen", "diff"]].std(ddof=1)
    corr = d.groupby("grp").apply(lambda g: g.chosen.corr(g.unchosen), include_groups=False)
    stats = {"n_groups": int(len(sd)), "n_traces": int(len(d)),
             "within_group_sd_median": {k: float(sd[k].median()) for k in sd.columns},
             "within_group_sd_q1_q3": {k: [float(sd[k].quantile(0.25)), float(sd[k].quantile(0.75))] for k in sd.columns},
             "groups_where_unchosen_is_tightest": int((sd.unchosen <= sd[["chosen", "diff"]].min(axis=1)).sum()),
             "groups_where_chosen_is_tightest": int((sd.chosen <= sd[["unchosen", "diff"]].min(axis=1)).sum()),
             "groups_where_diff_is_tightest": int((sd["diff"] <= sd[["chosen", "unchosen"]].min(axis=1)).sum()),
             "corr_chosen_unchosen_within_group_median": float(corr.median())}

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(8.4, 3.0), gridspec_kw={"width_ratios": [1.05, 2.0, 0.95]})
    from scipy.stats import gaussian_kde
    grid = np.linspace(-10, 10, 600)
    for k in ("unchosen", "diff", "chosen"):
        dens = gaussian_kde(d[f"{k}_c"].to_numpy(), bw_method=0.2)(grid)
        a.fill_between(grid, 0, dens, facecolor=COL[k], alpha=0.3, linewidth=0,
                       label=f"{ {'chosen': 'chosen', 'unchosen': 'unchosen', 'diff': 'difference'}[k] }  {stats['within_group_sd_median'][k]:.2f}")
        a.plot(grid, dens, color=DARK[k], linewidth=1.0)
    a.set_xlim(-8, 8)
    a.set_xlabel("Value at the stop, centred on the prompt median")
    a.set_ylabel("Density")
    a.set_ylim(0, 0.6)
    a.legend(loc="upper left", fontsize=5.4, title="within-prompt SD, median", title_fontsize=5.4, handlelength=1.0, handletextpad=0.4, borderpad=0.4)
    a.set_title("a", loc="left", fontweight="bold")

    groups = [g for it in order_items for g in (it + " right", it + " wrong") if g in sd.index]
    xs = np.arange(len(groups))
    for k, off, mk in (("chosen", -0.22, "o"), ("diff", 0.0, "D"), ("unchosen", 0.22, "s")):
        b.plot(xs + off, [sd.loc[g, k] for g in groups], marker=mk, markersize=3, linestyle="none", color=COL[k], markeredgecolor=DARK[k],
               markeredgewidth=0.5, label=NAME[k])
        b.axhline(sd[k].median(), color=DARK[k], linewidth=0.6, linestyle=(0, (4, 2)))
    b.set_xticks(xs, [short[g.rsplit(" ", 1)[0]].replace("Text-", "") + (" w" if g.endswith("wrong") else "") for g in groups], rotation=90, fontsize=5.2)
    b.set_xlabel("prompt (w = traces that end wrong)", fontsize=6)
    b.set_ylabel("SD at the stop within the prompt")
    b.set_ylim(0, max(sd.max()) * 1.15)
    b.legend(loc="upper left", fontsize=5.6, ncol=3, columnspacing=0.8, handletextpad=0.3)
    b.tick_params(axis="x", length=0)
    b.text(0.02, 0.86, f"race model predicts: chosen tight, unchosen free\ndiffusion predicts: difference tight\n"
                       f"observed: unchosen tightest in {stats['groups_where_unchosen_is_tightest']} of {stats['n_groups']} groups,\n"
                       f"difference in {stats['groups_where_diff_is_tightest']}, chosen in {stats['groups_where_chosen_is_tightest']}",
           transform=b.transAxes, ha="left", va="top", fontsize=5.6, bbox={"facecolor": "#ffffff", "edgecolor": "#bbbbbb", "linewidth": 0.5, "pad": 2})
    b.set_title("b", loc="left", fontweight="bold")

    c.scatter(d.unchosen_c, d.chosen_c, s=4, color=COL["chosen"], alpha=0.25, linewidths=0)
    lim = 9
    c.plot([-lim, lim], [-lim, lim], color="#000000", linewidth=0.6, linestyle=(0, (4, 2)))
    c.set_xlim(-lim, lim)
    c.set_ylim(-lim, lim)
    c.set_aspect("equal")
    c.set_xlabel("unchosen, centred")
    c.set_ylabel("chosen, centred")
    c.text(0.03, 0.97, f"within-prompt r = {stats['corr_chosen_unchosen_within_group_median']:.2f}", transform=c.transAxes, ha="left", va="top", fontsize=6)
    c.set_title("c", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.2)
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png",):
        fig.savefig(args.out / f"fig11_race_check.{ext}")
    print(json.dumps({k: v for k, v in stats.items() if k != "per_group_sd"}, indent=1))
    print(f"-> {args.out / 'fig11_race_check.png'}")


if __name__ == "__main__":
    main()
