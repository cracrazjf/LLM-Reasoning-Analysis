"""The three paper figures of the screening (black and white), from data/screen/ and the no-thinking runs.

    python src/figures.py        # -> figures/fig1_position_bias, fig2_thinking_effect, fig3_thinking_effect_by_order (.png)

Fig. 1  Position bias of Qwen3-8B over the whole dataset (981 pairs, each shown with the correct
        option as A and as B): accuracy when the correct option is A against when it is B, without
        thinking (one greedy answer) and with thinking (mean over 30 samples). The same pairs sit
        in both bars of a group, so the difference is a paired effect; error bars are 95% CIs over
        prompts, the bracket shows the paired difference with its 95% CI.
Fig. 2  What thinking does, pooled over both orders (letter-biased pairs left out): prompts where
        thinking flips a wrong first answer to the correct one, where it makes a correct first answer
        clearly more confident (70% rule), and where it does neither (no change, or worse).
Fig. 3  The same three categories, correct option shown as A against shown as B.
Fig. 4  How thinking moves the evidence for the correct answer, per prompt: (a) gap after thinking
        (median of the 30 samples) against the gap without thinking, gap = log p(correct) - log p(wrong)
        at the answer position; above the diagonal thinking moved toward the correct answer, below it
        toward the wrong one, the quadrants are stays-right / flips / stays-wrong; (b) the change in the
        gap for prompts the no-thinking answer got right and got wrong, the latter showing the
        "more wrong" mass on the left and the flips on the right.
Fig. 5  Accuracy of the 12 study pairs in the main run (100 thinking samples per position of the
        correct option, Wilson 95% CI), grouped by the no-thinking outcome. Drawn only when
        runs/medxpertqa/main-qwen3-8b-ab and -ba exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate import ROOT

matplotlib.use("Agg")

SCREEN = ROOT / "data/screen"
OUT = ROOT / "figures"
WHITE, LIGHT, MID, DARK, BLACK = "#ffffff", "#c8c8c8", "#8c8c8c", "#4d4d4d", "#000000"
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "hatch.linewidth": 0.5, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
}
CATEGORIES = [("flip", "Flipped to\ncorrect"), ("confident", "Clearly more\nconfident"), ("none", "No\nimprovement")]


def categories(k: pd.DataFrame) -> dict[str, int]:
    """Prompt counts of the three categories plus the split of the third one."""
    r = k["reason"].fillna("")
    return {"flip": int((r == "flips to correct").sum()), "confident": int((r == "more confident").sum()),
            "neutral": int((k.cls == "neutral").sum()), "more_wrong": int((r == "more wrong").sum()),
            "flip_wrong": int((r == "flips to wrong").sum()),
            "none": int((k.cls == "neutral").sum() + (k.cls == "hurts").sum()), "n": int(len(k))}


def bar_label(ax, x, top, text, dy=0.012):
    ax.text(x, top + dy * ax.get_ylim()[1], text, ha="center", va="bottom", fontsize=7, color=BLACK)


def category_bars(ax, counts_by_series: list[tuple[str, dict[str, int], str, str | None]], width: float, show_pct=True):
    """Three categories on x; one bar per series = (label, counts, facecolor, hatch). The third bar is stacked:
    no change (plain) below, worse (dotted hatch) above."""
    x = np.arange(len(CATEGORIES))
    n_series = len(counts_by_series)
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * width
    tops = []
    for (label, c, face, hatch), off in zip(counts_by_series, offsets):
        vals = [c["flip"], c["confident"], c["neutral"]]
        ax.bar(x + off, vals, width=width * 0.92, facecolor=face, edgecolor=BLACK, linewidth=0.6, hatch=hatch, label=label)
        worse = c["more_wrong"] + c["flip_wrong"]
        ax.bar(x[2] + off, worse, bottom=c["neutral"], width=width * 0.92, facecolor=face, edgecolor=BLACK,
               linewidth=0.6, hatch=(hatch or "") + "....")
        for i, v in enumerate([c["flip"], c["confident"], c["none"]]):
            bar_label(ax, x[i] + off, v, f"{v}\n({v / c['n']:.0%})" if show_pct else f"{v}")
        tops.append(c["none"])
    ax.set_xticks(x, [lab for _, lab in CATEGORIES])
    ax.set_ylabel("Prompts")
    ax.tick_params(axis="x", length=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen", type=Path, default=SCREEN)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    plt.rcParams.update(STYLE)
    args.out.mkdir(parents=True, exist_ok=True)

    p = pd.read_csv(args.screen / "medxpertqa_prompts.csv")
    k = p[p.cls != "excluded"].copy()

    # ---------------- Fig. 1: position bias = accuracy by the position of the correct option
    p["acc_nothink"] = p.nothink_correct.astype(float)
    p["acc_think"] = p.n_correct / p.n_answered
    w = p.pivot(index="pair_id", columns="order", values=["acc_nothink", "acc_think"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]   # order ab = correct option is A, ba = correct option is B
    n_pairs = int(len(w))
    stats = {}
    for mode in ("nothink", "think"):
        a_, b_ = w[f"acc_{mode}_ab"], w[f"acc_{mode}_ba"]
        d = a_ - b_
        stats[mode] = {"acc_correct_A": float(a_.mean()), "ci_correct_A": float(1.96 * a_.std(ddof=1) / np.sqrt(len(a_))),
                       "acc_correct_B": float(b_.mean()), "ci_correct_B": float(1.96 * b_.std(ddof=1) / np.sqrt(len(b_))),
                       "paired_diff": float(d.mean()), "ci_paired_diff": float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))}

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    width = 0.34
    x = np.arange(2)
    modes = [("nothink", "No thinking\n(one greedy answer)"), ("think", "Thinking\n(mean of 30 samples)")]
    series = (("Correct option is A", "A", WHITE), ("Correct option is B", "B", MID))
    for (label, pos, face), off in zip(series, (-width / 2, width / 2)):
        vals = [stats[m][f"acc_correct_{pos}"] for m, _ in modes]
        errs = [stats[m][f"ci_correct_{pos}"] for m, _ in modes]
        ax.bar(x + off, vals, width=width * 0.92, facecolor=face, edgecolor=BLACK, linewidth=0.6, label=label,
               yerr=errs, error_kw={"elinewidth": 0.6, "capsize": 2, "capthick": 0.6, "ecolor": BLACK})
        for xi, v, e in zip(x + off, vals, errs):
            ax.text(xi, v + e + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    # paired difference brackets
    for xi, (m, _) in zip(x, modes):
        top = max(stats[m]["acc_correct_A"] + stats[m]["ci_correct_A"], stats[m]["acc_correct_B"] + stats[m]["ci_correct_B"]) + 0.06
        ax.plot([xi - width / 2, xi - width / 2, xi + width / 2, xi + width / 2], [top, top + 0.015, top + 0.015, top],
                color=BLACK, linewidth=0.6)
        d, ci = stats[m]["paired_diff"], stats[m]["ci_paired_diff"]
        ax.text(xi, top + 0.025, f"A − B = {d:+.3f}\n[{d - ci:+.3f}, {d + ci:+.3f}]", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, [lab for _, lab in modes])
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper left")
    ax.text(0.98, 0.98, f"{n_pairs} pairs, each shown\nin both positions", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
    fig.tight_layout()
    for ext in ("png",):
        fig.savefig(args.out / f"fig1_position_bias.{ext}")
    plt.close(fig)

    # ---------------- Fig. 2: what thinking does, pooled
    c_all = categories(k)
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    category_bars(ax, [("all", c_all, LIGHT, None)], width=0.6)
    ax.set_ylim(0, c_all["confident"] * 1.18)
    # legend for the dotted part of the third bar
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=LIGHT, edgecolor=BLACK, linewidth=0.6, label="No change"),
                       Patch(facecolor=LIGHT, edgecolor=BLACK, linewidth=0.6, hatch="....", label="Worse")],
              loc="upper left", title=None)
    ax.text(0.98, 0.98, f"n = {c_all['n']} prompts\n(both orders pooled)", transform=ax.transAxes, ha="right", va="top", fontsize=7)
    fig.tight_layout()
    for ext in ("png",):
        fig.savefig(args.out / f"fig2_thinking_effect.{ext}")
    plt.close(fig)

    # ---------------- Fig. 3: by order
    c_ab, c_ba = categories(k[k.order == "ab"]), categories(k[k.order == "ba"])
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    category_bars(ax, [("Correct option is A", c_ab, WHITE, None), ("Correct option is B", c_ba, MID, None)], width=0.38)
    ax.set_ylim(0, max(c_ab["confident"], c_ba["confident"]) * 1.5)
    handles = [Patch(facecolor=WHITE, edgecolor=BLACK, linewidth=0.6, label="Correct option is A"),
               Patch(facecolor=MID, edgecolor=BLACK, linewidth=0.6, label="Correct option is B"),
               Patch(facecolor=WHITE, edgecolor=BLACK, linewidth=0.6, hatch="....", label="Worse")]
    ax.legend(handles=handles, loc="upper left", ncol=1)
    ax.text(0.98, 0.98, f"n = {c_ab['n']} prompts\nper position", transform=ax.transAxes, ha="right", va="top", fontsize=7)
    fig.tight_layout()
    for ext in ("png",):
        fig.savefig(args.out / f"fig3_thinking_effect_by_order.{ext}")
    plt.close(fig)

    # ---------------- Fig. 4: gap before and after thinking
    g0, g1 = k.nothink_gap, k.think_gap_median
    delta = g1 - g0
    right, wrong = k.nothink_correct == True, k.nothink_correct == False  # noqa: E712
    quad = {"stays_right": int(((g0 > 0) & (g1 > 0)).sum()), "flips_to_correct": int(((g0 < 0) & (g1 > 0)).sum()),
            "flips_to_wrong": int(((g0 > 0) & (g1 < 0)).sum()), "stays_wrong": int(((g0 < 0) & (g1 < 0)).sum()),
            "ties_gap_0": int(((g0 == 0) | (g1 == 0)).sum())}
    above = {"stays_right_more_confident": int(((g0 > 0) & (g1 > g0)).sum()), "stays_wrong_more_wrong": int(((g0 < 0) & (g1 < g0)).sum())}
    change = {"right": {"n": int(right.sum()), "share_up": float((delta[right] > 0).mean()), "median": float(delta[right].median())},
              "wrong": {"n": int(wrong.sum()), "share_down": float((delta[wrong] < 0).mean()),
                        "share_crosses_to_correct": float((g1[wrong] > 0).mean()), "median": float(delta[wrong].median())}}

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw={"width_ratios": [1.05, 1]})
    lim = 30
    a.plot([-lim, lim], [-lim, lim], color=BLACK, linewidth=0.6, linestyle=(0, (4, 3)))
    a.axhline(0, color=BLACK, linewidth=0.4)
    a.axvline(0, color=BLACK, linewidth=0.4)
    a.scatter(g0, g1, s=4, facecolor=DARK, edgecolor="none", alpha=0.45)
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_aspect("equal")
    a.set_xlabel("Gap without thinking")
    a.set_ylabel("Gap after thinking (median of 30)")
    box = {"facecolor": WHITE, "edgecolor": "none", "alpha": 0.85, "pad": 1.5}
    a.text(1.5, lim - 0.8, f"Stays correct {quad['stays_right']}\n({above['stays_right_more_confident']} above the line:\nmore confident)",
           ha="left", va="top", fontsize=6.5, bbox=box)
    a.text(-lim + 1, lim - 0.8, f"Flips to correct\n{quad['flips_to_correct']}", ha="left", va="top", fontsize=6.5, bbox=box)
    a.text(lim - 1, -lim + 0.8, f"Flips to wrong\n{quad['flips_to_wrong']}", ha="right", va="bottom", fontsize=6.5, bbox=box)
    a.text(-lim + 1, -lim + 0.8, f"Stays wrong {quad['stays_wrong']}\n({above['stays_wrong_more_wrong']} below the line:\nmore wrong)",
           ha="left", va="bottom", fontsize=6.5, bbox=box)
    a.set_title("a", loc="left", fontweight="bold")

    bins = np.arange(-26, 27, 2)
    b.hist(delta[right], bins=bins, facecolor=WHITE, edgecolor=BLACK, linewidth=0.6, label=f"No-thinking answer right ({int(right.sum())})")
    b.hist(delta[wrong], bins=bins, facecolor=MID, edgecolor=BLACK, linewidth=0.6, alpha=0.9, label=f"No-thinking answer wrong ({int(wrong.sum())})")
    b.axvline(0, color=BLACK, linewidth=0.6, linestyle=(0, (4, 3)))
    b.set_xlabel("Change in gap after thinking")
    b.set_ylabel("Prompts")
    b.set_xlim(-26, 26)
    b.set_ylim(0, 300)
    b.legend(loc="upper left")
    b.text(0.02, 0.80, f"Right: {change['right']['share_up']:.0%} become more confident\n"
                       f"Wrong: {change['wrong']['share_down']:.0%} move further wrong,\n"
                       f"           {change['wrong']['share_crosses_to_correct']:.0%} end up correct",
           transform=b.transAxes, ha="left", va="top", fontsize=6.5)
    b.set_title("b", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=2)
    for ext in ("png",):
        fig.savefig(args.out / f"fig4_gap_change.{ext}")
    plt.close(fig)

    # ---------------- Fig. 5: accuracy of the study pairs in the main run (100 samples per order)
    study_path = SCREEN / "medxpertqa_study12.csv"
    main_runs = {o: ROOT / f"runs/medxpertqa/main-qwen3-8b-{o}" for o in ("ab", "ba")}
    if study_path.exists() and all(r.exists() for r in main_runs.values()):
        from generate import iter_traces
        study = pd.read_csv(study_path)
        acc = {}
        for o, run in main_runs.items():
            df = pd.DataFrame(iter_traces(run, ("item_id", "pair_id", "correct")))
            for pid, g in df.groupby("pair_id"):
                n, m = len(g), int(g.correct.fillna(False).sum())
                centre = (m + 1.92) / (n + 3.84)                      # Wilson 95% interval
                half = 1.96 * np.sqrt(m / n * (1 - m / n) / n + 0.96 / n ** 2) / (1 + 3.84 / n)
                acc[(pid, o)] = {"n": n, "k": m, "acc": m / n, "lo": centre - half, "hi": centre + half}
        study["mean_acc"] = [(acc[(pid, "ab")]["acc"] + acc[(pid, "ba")]["acc"]) / 2 for pid in study.pair_id]
        study = study.sort_values(["group", "mean_acc"], ascending=[True, False]).reset_index(drop=True)

        def short(s: str, n: int = 34) -> str:
            return s if len(s) <= n else s[: n - 1] + "…"

        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        height = 0.36
        ys, labels = [], []
        gap_rows, headers = [], []
        y = 0
        last_group = None
        for _, r in study.iterrows():
            if last_group is not None and r.group != last_group:
                gap_rows.append(y - 0.45)          # separator line just below the previous group
                headers.append((r.group, y + 0.05))   # header above the first bar of the new group
                y += 1.0
            elif last_group is None:
                headers.append((r.group, y - 0.95))
            last_group = r.group
            for (o, face), off in zip((("ab", WHITE), ("ba", MID)), (-height / 2, height / 2)):
                a_ = acc[(r.pair_id, o)]
                ax.barh(y + off, a_["acc"], height=height * 0.92, facecolor=face, edgecolor=BLACK, linewidth=0.6,
                        xerr=[[a_["acc"] - a_["lo"]], [a_["hi"] - a_["acc"]]],
                        error_kw={"elinewidth": 0.6, "capsize": 1.8, "capthick": 0.6, "ecolor": BLACK})
                ax.text(min(a_["hi"], 1.0) + 0.012, y + off, f"{a_['k']}", va="center", fontsize=6.5)
            ys.append(y)
            labels.append(f"{r.source_id}  {short(r.correct_text)} vs {short(r.distractor_text, 26)}")
            y += 1
        for gy in gap_rows:
            ax.axhline(gy, color=BLACK, linewidth=0.5)
        ax.set_yticks(ys, labels, fontsize=6.8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.axvline(0.5, color=BLACK, linewidth=0.5, linestyle=(0, (3, 2)))
        ax.set_xlabel("Accuracy over 100 thinking samples (Wilson 95% CI)")
        ax.tick_params(axis="y", length=0)
        names = {"A": "Group A: no-thinking answer wrong in both positions", "B": "Group B: no-thinking answer right in both positions"}
        for grp, hy in headers:
            ax.text(1.11, hy, names[grp], ha="right", va="center", fontsize=7, fontweight="semibold")
        ax.legend(handles=[Patch(facecolor=WHITE, edgecolor=BLACK, linewidth=0.6, label="Correct option is A"),
                           Patch(facecolor=MID, edgecolor=BLACK, linewidth=0.6, label="Correct option is B")],
                  loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.4)
        ax.set_ylim(ys[-1] + 0.7, ys[0] - 2.2)   # room for the legend row above the first group header
        fig.tight_layout()
        for ext in ("png",):
            fig.savefig(args.out / f"fig5_study_accuracy.{ext}")
        plt.close(fig)
        fig5 = {f"{pid}:{o}": round(v["acc"], 2) for (pid, o), v in acc.items()}
    else:
        fig5 = "main run not found"

    print(json.dumps({"fig1": stats, "fig2": c_all, "fig3": {"correct_A": c_ab, "correct_B": c_ba},
                      "fig4": {"quadrants": quad, "relative_to_diagonal": above, "change": change}, "fig5": fig5}, indent=1))
    print(f"-> {args.out}/fig1_position_bias, fig2_thinking_effect, fig3_thinking_effect_by_order, fig4_gap_change, "
          f"fig5_study_accuracy (.png)")


if __name__ == "__main__":
    main()
