"""Instruction pilot, readout side: where the answer is decided and how high the stop is, per condition.

    python src/pilot_readouts.py --run runs/medxpertqa/pilot-qwen3-8b

Uses <run>/readouts/ (src/readout.py) and the traces. For every trace, X is signed toward the
correct option (X_c = X if the label is A else -X), so X_c > 0 means "leaning correct".

Per trace:
  stop_X        X_c at the stop readout (kind "e", where the model wrote </think>)
  decided_at    first readout position after the LAST sign change of X; the sign is constant
                from there to the stop. 0 = never changed (already the final sign at the first
                readout). tokens_after = think_end - decided_at.
  switched      whether the sign of X ever differs from the final sign in the thinking

Per condition x band (and within prompt vs C0): median stop |X|, median thinking tokens,
median tokens after the decision and their share of the thinking, share of traces that never
switched, and the forced-answer accuracy at fixed absolute positions (every-50-token grid):
the speed-accuracy function of the response-signal method.

Predictions. Boundary change (DDM): higher stop |X| and later decisions under C1, lower and
earlier under C2, with the same forced-answer accuracy at matched positions. Non-decision-time
inflation ("decide early, explain later"): the same stop |X| and the same decided_at, with the
extra tokens of C1 after the decision. Writes <run>/analysis/readouts_summary.json, per_trace.csv
and fig_sat.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from generate import iter_traces, parse_answer  # noqa: E402

BASE = "C0_baseline"
GRID = (50, 100, 200, 300, 400, 600, 800, 1000, 1500, 2000)


def load(run: Path) -> pd.DataFrame:
    traces = {}
    for r in iter_traces(run, ("sample_id", "prompt_id", "item_id", "pair_id", "source_id", "order", "band", "condition",
                               "label", "answer_text", "think_tokens", "X")):
        a, fmt = parse_answer(r["answer_text"], ("A", "B"))
        traces[r["sample_id"]] = {**r, "answer": a, "correct": (a == r["label"]) if a else None}
    rows = []
    for f in sorted((run / "readouts").glob("readouts-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                ro = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = traces.get(ro["sample_id"])
            if t is None or t["answer"] is None:
                continue
            sign = 1.0 if t["label"] == "A" else -1.0
            pos = np.array(ro["pos"], dtype=float)
            X = np.array([np.nan if x is None else x for x in ro["X"]], dtype=float) * sign
            kinds = ro["kind"]
            ok = ~np.isnan(X)
            if ok.sum() == 0:
                continue
            final = 1.0 if t["correct"] else -1.0  # sign of the final answer, toward correct
            stop_idx = next((i for i, k in enumerate(kinds) if "e" in k), len(kinds) - 1)
            stop_X = X[stop_idx]
            signs = np.sign(X[ok])
            p_ok = pos[ok]
            wrong = np.where(signs != final)[0]
            if len(wrong) == 0:
                decided_at = p_ok[0] if len(p_ok) else 0.0
                decided_at = 0.0
                switched = False
            else:
                last_wrong = wrong[-1]
                decided_at = p_ok[last_wrong + 1] if last_wrong + 1 < len(p_ok) else p_ok[last_wrong]
                switched = True
            start = ro["think_start"]
            rel = {}
            for g in GRID:  # forced answer at absolute thinking positions (grid readouts, kind contains "g")
                target = start + g
                j = np.where((pos == target) & ok)[0]
                rel[f"acc_at_{g}"] = float(signs[np.searchsorted(p_ok, target)] == 1.0) if len(j) else np.nan
            rows.append({
                "sample_id": ro["sample_id"], "prompt_id": t["prompt_id"], "item_id": t["item_id"], "pair_id": t["pair_id"],
                "source_id": t["source_id"], "order": t["order"], "band": t["band"], "condition": t["condition"],
                "correct": bool(t["correct"]), "think_tokens": ro["think_end"] - start,
                "stop_X": float(stop_X) if not np.isnan(stop_X) else np.nan, "stop_absX": float(abs(stop_X)) if not np.isnan(stop_X) else np.nan,
                "first_X": float(X[ok][0]), "first_sign_final": bool(signs[0] == final),
                "decided_at": float(decided_at - start) if switched else 0.0,
                "tokens_after": float(ro["think_end"] - decided_at) if switched else float(ro["think_end"] - start),
                "switched": switched, "n_readouts": int(ok.sum()), **rel,
            })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    out = {"n_traces": int(len(df)), "conditions": sorted(df.condition.unique())}
    cols = ["stop_absX", "think_tokens", "decided_at", "tokens_after", "switched", "first_sign_final", "correct"]
    for band in ("all", "hard", "mid"):
        d = df if band == "all" else df[df.band == band]
        out[band] = {}
        for cond, g in d.groupby("condition"):
            s = {
                "n": int(len(g)), "accuracy": float(g.correct.mean()),
                "stop_absX_median": float(g.stop_absX.median()),
                "think_tokens_median": float(g.think_tokens.median()),
                "share_switched": float(g.switched.mean()),
                "share_first_readout_already_final": float(g.first_sign_final.mean()),
                "decided_at_median_switched": float(g[g.switched].decided_at.median()) if g.switched.any() else np.nan,
                "decided_frac_median_switched": float((g[g.switched].decided_at / g[g.switched].think_tokens.clip(lower=1)).median()) if g.switched.any() else np.nan,
                "tokens_after_median_switched": float(g[g.switched].tokens_after.median()) if g.switched.any() else np.nan,
                "sat": {str(k): {"acc": float(g[f"acc_at_{k}"].mean()), "n": int(g[f"acc_at_{k}"].notna().sum())}
                        for k in GRID if g[f"acc_at_{k}"].notna().sum() >= 20},
            }
            out[band][cond] = s
        # within-prompt contrasts vs baseline
        pp = d.groupby(["item_id", "condition"]).agg(stop_absX=("stop_absX", "median"), think=("think_tokens", "median"),
                                                     after=("tokens_after", "median"), switched=("switched", "mean"),
                                                     acc=("correct", "mean")).reset_index()
        base = pp[pp.condition == BASE].set_index("item_id")
        out[band]["vs_baseline_within_prompt"] = {}
        for cond in sorted(df.condition.unique()):
            if cond == BASE:
                continue
            c = pp[pp.condition == cond].set_index("item_id")
            idx = base.index.intersection(c.index)
            if len(idx) == 0:
                continue
            out[band]["vs_baseline_within_prompt"][cond] = {
                "n_prompts": int(len(idx)),
                "stop_absX_diff_median": float((c.loc[idx].stop_absX - base.loc[idx].stop_absX).median()),
                "share_prompts_higher_stop_absX": float((c.loc[idx].stop_absX > base.loc[idx].stop_absX).mean()),
                "think_ratio_median": float((c.loc[idx].think.clip(lower=1) / base.loc[idx].think.clip(lower=1)).median()),
                "tokens_after_ratio_median": float((c.loc[idx].after.clip(lower=1) / base.loc[idx].after.clip(lower=1)).median()),
                "switched_diff_mean": float((c.loc[idx].switched - base.loc[idx].switched).mean()),
                "acc_diff_mean": float((c.loc[idx].acc - base.loc[idx].acc).mean()),
            }
    return out


def figure(df: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, band in zip(axes, ("hard", "mid")):
        d = df[df.band == band]
        for cond, g in d.groupby("condition"):
            xs, ys = [], []
            for k in GRID:
                col = g[f"acc_at_{k}"]
                if col.notna().sum() >= 20:
                    xs.append(k)
                    ys.append(col.mean())
            if xs:
                ax.plot(xs, ys, marker="o", ms=3, label=cond)
        ax.set_title(f"{band} pairs")
        ax.set_xlabel("thinking tokens read (forced answer)")
        ax.set_xscale("log")
    axes[0].set_ylabel("forced-answer accuracy")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_sat.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    df = load(args.run)
    if df.empty:
        raise SystemExit("no readouts matched traces")
    out = args.run / "analysis"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "per_trace.csv", index=False)
    summary = summarise(df)
    (out / "readouts_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    figure(df, out)
    print(f"{len(df)} traces with readouts")
    hdr = f"{'band':<5}{'condition':<18}{'n':>6}{'acc':>6}{'stop|X|':>8}{'think':>7}{'switch':>7}{'1st=fin':>8}{'dec@':>7}{'frac':>6}{'after':>7}"
    print(hdr)
    for band in ("hard", "mid"):
        for cond, s in summary[band].items():
            if cond == "vs_baseline_within_prompt":
                continue
            print(f"{band:<5}{cond:<18}{s['n']:>6}{s['accuracy']:>6.2f}{s['stop_absX_median']:>8.1f}{s['think_tokens_median']:>7.0f}"
                  f"{s['share_switched']:>7.2f}{s['share_first_readout_already_final']:>8.2f}{s['decided_at_median_switched']:>7.0f}"
                  f"{s['decided_frac_median_switched']:>6.2f}{s['tokens_after_median_switched']:>7.0f}")
    print(f"-> {out}/readouts_summary.json, per_trace.csv, fig_sat.png")


if __name__ == "__main__":
    main()
