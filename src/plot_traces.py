"""Evidence trajectories of every trace of the main run, one panel per prompt.

    python src/plot_traces.py        # -> figures/fig6_trajectories.png

For each of the 12 study pairs and each position of the correct option, the 100 traces are drawn
as lines: x = sentence index inside the thinking (0 = the empty-thinking readout before any
reasoning, the last point = the stop, where the model wrote </think>), y = evidence for the correct
option, log p(correct) - log p(wrong) in the closed context (the answer the model would give if it
stopped here). Traces that end on the correct answer are light grey, traces that end on the wrong
answer black; a dot marks the stop. The x-axis of each panel ends at the 95th percentile of the
stop positions of that prompt, so a few long traces run off the right edge.
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

matplotlib.use("Agg")

RUNS = {"ab": ROOT / "runs/medxpertqa/main-qwen3-8b-ab", "ba": ROOT / "runs/medxpertqa/main-qwen3-8b-ba"}
STUDY = ROOT / "data/screen/medxpertqa_study12.csv"
OUT = ROOT / "figures"
GREY, BLACK = "#9a9a9a", "#000000"
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
         "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.5,
         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42}


def load(run: Path) -> tuple[dict[str, dict], list[dict], dict[str, dict]]:
    traces = {r["sample_id"]: r for r in iter_traces(run, ("sample_id", "item_id", "pair_id", "label", "correct", "answer"))}
    readouts = [json.loads(l) for l in (run / "readouts/readouts-01.jsonl").open(encoding="utf-8")]
    start = {json.loads(l)["item_id"]: json.loads(l) for l in (run / "readouts/readouts_start.jsonl").open(encoding="utf-8")}
    return traces, readouts, start


def trajectory(r: dict, label: str, start: dict) -> np.ndarray:
    """Evidence for the correct option at x = 0 (empty thinking), each sentence start, and the stop."""
    sign = 1.0 if label == "A" else -1.0
    xs = [sign * (start["logp_A"] - start["logp_B"])]
    for k, a, b in zip(r["kind"], r["closed"]["logp_A"], r["closed"]["logp_B"]):
        if k in ("s", "e"):
            xs.append(sign * (a - b))
    return np.array(xs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--ylim", type=float, default=30.0)
    args = ap.parse_args()
    plt.rcParams.update(STYLE)
    study = pd.read_csv(STUDY).sort_values(["group", "pair_id"]).reset_index(drop=True)
    data = {o: load(run) for o, run in RUNS.items()}

    fig, axes = plt.subplots(6, 4, figsize=(7.2, 9.4), sharey=True)
    summary = {}
    for i, srow in study.iterrows():
        for j, o in enumerate(("ab", "ba")):
            ax = axes[i // 2, (i % 2) * 2 + j]
            traces, readouts, start = data[o]
            iid = f"medxpertqa:{srow.pair_id}:{o}"
            recs = [r for r in readouts if r["item_id"] == iid]
            label = traces[recs[0]["sample_id"]]["label"]
            series = [(trajectory(r, label, start[iid]), bool(traces[r["sample_id"]]["correct"])) for r in recs]
            n_sent = [len(s) - 1 for s, _ in series]
            xmax = int(np.percentile(n_sent, 95))
            for s, ok in sorted(series, key=lambda t: t[1], reverse=True):   # wrong traces drawn last, on top
                x = np.arange(len(s))
                ax.plot(x, s, color=GREY if ok else BLACK, linewidth=0.45 if ok else 0.7, alpha=0.35 if ok else 0.8)
                if len(s) - 1 <= xmax:
                    ax.plot(x[-1], s[-1], marker="o", markersize=1.6, color=GREY if ok else BLACK, alpha=0.6 if ok else 0.9, linewidth=0)
            ax.axhline(0, color=BLACK, linewidth=0.4, linestyle=(0, (3, 2)))
            ax.set_xlim(0, xmax)
            ax.set_ylim(-args.ylim, args.ylim)
            n_ok = sum(ok for _, ok in series)
            ax.set_title(f"{srow.source_id} · correct = {label} · {n_ok}/{len(series)} right", loc="left", pad=2)
            if (i % 2) * 2 + j == 0:
                ax.set_ylabel("log odds, correct vs wrong")
            if i // 2 == 5:
                ax.set_xlabel("sentence")
            summary[iid] = {"n": len(series), "right": n_ok, "sentences_median": float(np.median(n_sent)), "sentences_p95": xmax,
                            "start": float(series[0][0][0]), "stop_median_right": float(np.median([s[-1] for s, ok in series if ok])) if n_ok else None,
                            "stop_median_wrong": float(np.median([s[-1] for s, ok in series if not ok])) if n_ok < len(series) else None}
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=GREY, linewidth=1, label="ends on the correct answer"),
                        Line2D([], [], color=BLACK, linewidth=1, label="ends on the wrong answer"),
                        Line2D([], [], color=BLACK, marker="o", markersize=2.5, linewidth=0, label="stop (</think>)")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.01, 0.995, "Group A (rows 1-4): no-thinking answer wrong.  Group B (rows 5-6): no-thinking answer right.  "
             "x = 0 is the answer before any reasoning.", ha="left", va="top", fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 0.985), h_pad=0.8, w_pad=0.6)
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png",):
        fig.savefig(args.out / f"fig6_trajectories.{ext}")
    print(json.dumps(summary, indent=1))
    print(f"-> {args.out / 'fig6_trajectories.png'}")


if __name__ == "__main__":
    main()
