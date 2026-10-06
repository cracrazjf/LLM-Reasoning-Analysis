"""Where do traces stop? Evidence height at the stop against thinking length, per condition.

    python src/stop_height.py --run runs/medxpertqa/main-qwen3-8b

Height H = |X| at the answer position (X = log p(A) - log p(B) read after the model's own
"</think>"), i.e. how far the evidence stands toward the answer the trace gives when it stops.
A fixed boundary predicts H does not depend on how long the trace thought; a collapsing boundary
predicts H falls with length. Comparisons are within cell (prompt x condition fixed effects), so
differences between questions cannot produce a slope.

Reports: (1) boundary shape: constant vs linear vs logarithmic decline (BIC), slopes with CIs from a
bootstrap over questions; (2) the profile of H over length bins; (3) the condition shift of H at
matched length and unadjusted, against the ratio of boundary separations fitted by src/ddm_fit.py;
(4) correct vs wrong traces. Writes <run>/analysis/stop_height.json and fig_stop_height.png.

Limit: this X is read after the concluding sentence and "</think>" (the saturated end readout).
The height just before the conclusion needs the truncation readouts (src/readout.py).
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
from pair_distance import NON_DIAGNOSIS_PAIRS  # noqa: E402

COND = {"C0_baseline": "baseline", "C1_caution_think": "careful", "C2_speed_think": "speed"}
ORDER = ["speed", "baseline", "careful"]


def load(run: Path) -> pd.DataFrame:
    rows = []
    for r in iter_traces(run, ("sample_id", "item_id", "pair_id", "source_id", "condition", "label", "answer_text",
                               "think_tokens", "finish_reason", "X", "X_bound")):
        if r["pair_id"] in NON_DIAGNOSIS_PAIRS or r["condition"] not in COND or r["X"] is None or not r["think_tokens"]:
            continue
        a, _ = parse_answer(r["answer_text"], ("A", "B"))
        if a is None or r["finish_reason"] == "length":
            continue
        rows.append({"item_id": r["item_id"], "source_id": r["source_id"], "cond": COND[r["condition"]],
                     "correct": float(a == r["label"]), "t": r["think_tokens"] / 1000.0, "X": r["X"],
                     "toward_answer": r["X"] if a == "A" else -r["X"], "bound": r["X_bound"] is not None})
    df = pd.DataFrame(rows).drop_duplicates()
    df["H"] = df.X.abs()
    df["logt"] = np.log(df.t)
    df["cell"] = df.item_id + "|" + df.cond
    return df


def within(df: pd.DataFrame, cols: list[str], by: str) -> pd.DataFrame:
    return df[cols] - df.groupby(by)[cols].transform("mean")


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b, float(((y - X @ b) ** 2).sum())


def boot_slopes(df: pd.DataFrame, cols: list[str], by: str, n: int = 500, seed: int = 0) -> np.ndarray:
    """Bootstrap over questions of the within-`by` OLS coefficients of H on `cols`."""
    rng = np.random.default_rng(seed)
    w = within(df, ["H"] + cols, by)
    w["q"] = df.source_id.values
    groups = {q: g[["H"] + cols].values for q, g in w.groupby("q")}
    qs = list(groups)
    out = []
    for _ in range(n):
        m = np.concatenate([groups[q] for q in rng.choice(qs, len(qs))])
        out.append(ols(m[:, 0], m[:, 1:])[0])
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    df = load(args.run)
    n, ncell = len(df), df.cell.nunique()
    S: dict = {"n_traces": n, "n_cells": ncell, "sign_of_X_matches_answer": float((df.toward_answer > 0).mean()),
               "bounded_readouts": int(df.bound.sum()),
               "H_median_by_condition": {c: float(df[df.cond == c].H.median()) for c in ORDER},
               "H_quantiles": {c: [float(df[df.cond == c].H.quantile(q)) for q in (.1, .25, .5, .75, .9)] for c in ORDER}}

    # (1) boundary shape within cell
    w = within(df, ["H", "t", "logt"], "cell")
    y = w.H.values
    tss = float((y ** 2).sum())
    shapes = {"constant": [], "linear": ["t"], "log": ["logt"], "linear+log": ["t", "logt"]}
    S["shape"] = {}
    for name, cols in shapes.items():
        if cols:
            b, rss = ols(y, w[cols].values)
        else:
            b, rss = np.array([]), tss
        k = ncell + len(cols)
        S["shape"][name] = {"coef": dict(zip(cols, map(float, b))), "rss": rss, "r2_within": 1 - rss / tss,
                            "bic": n * np.log(rss / n) + k * np.log(n)}
    best = min(S["shape"], key=lambda q: S["shape"][q]["bic"])
    for name in S["shape"]:
        S["shape"][name]["delta_bic"] = S["shape"][name]["bic"] - S["shape"][best]["bic"]
    S["shape_best"] = best
    bl = boot_slopes(df, ["t"], "cell")
    bg = boot_slopes(df, ["logt"], "cell")
    S["slope_linear_per_1000_tokens"] = {"est": S["shape"]["linear"]["coef"]["t"], "ci95": [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))]}
    S["slope_log"] = {"est": S["shape"]["log"]["coef"]["logt"], "ci95": [float(np.percentile(bg, 2.5)), float(np.percentile(bg, 97.5))]}
    S["slope_by_condition"] = {}
    for c in ORDER:
        d = df[df.cond == c]
        wc = within(d, ["H", "t", "logt"], "cell")
        S["slope_by_condition"][c] = {"linear_per_1000": float(ols(wc.H.values, wc[["t"]].values)[0][0]),
                                      "log": float(ols(wc.H.values, wc[["logt"]].values)[0][0]),
                                      "share_cells_negative": float(np.mean([np.polyfit(g.logt, g.H, 1)[0] < 0 for _, g in d.groupby("cell") if g.logt.std() > 0]))}

    # (2) profile over length bins (cell effects removed)
    edges = [0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.5, 100]
    df["H_adj"] = df.H - df.groupby("cell").H.transform("mean") + df.groupby("cond").H.transform("mean")
    df["bin"] = pd.cut(df.t, edges)
    prof = df.groupby(["cond", "bin"], observed=True).agg(n=("H", "size"), t=("t", "median"), H_raw=("H", "mean"), H_adj=("H_adj", "mean")).reset_index()
    S["profile"] = {c: [{"t_median_k": float(r.t), "n": int(r.n), "H_raw": float(r.H_raw), "H_within_cell": float(r.H_adj)}
                        for r in prof[prof.cond == c].itertuples()] for c in ORDER}

    # (3) condition shift: unadjusted within prompt, and at matched length (prompt fixed effects + log t)
    med = df.groupby(["item_id", "cond"]).H.median().unstack()
    S["condition_ratio_unadjusted"] = {"speed/baseline": float((med.speed / med.baseline).median()),
                                       "careful/baseline": float((med.careful / med.baseline).median()),
                                       "mean_H": {c: float(med[c].mean()) for c in ORDER}}
    d = df.copy()
    d["speed"] = (d.cond == "speed").astype(float)
    d["careful"] = (d.cond == "careful").astype(float)
    cols = ["speed", "careful", "logt"]
    wp = within(d, ["H"] + cols, "item_id")
    b, _ = ols(wp.H.values, wp[cols].values)
    bb = boot_slopes(d, cols, "item_id", n=400)
    S["condition_shift_at_matched_length"] = {c: {"est": float(b[i]), "ci95": [float(np.percentile(bb[:, i], 2.5)), float(np.percentile(bb[:, i], 97.5))]}
                                              for i, c in enumerate(cols)}
    b2, _ = ols(within(d, ["H", "speed", "careful"], "item_id").H.values, within(d, ["H", "speed", "careful"], "item_id")[["speed", "careful"]].values)
    bb2 = boot_slopes(d, ["speed", "careful"], "item_id", n=400)
    S["condition_shift_unadjusted"] = {c: {"est": float(b2[i]), "ci95": [float(np.percentile(bb2[:, i], 2.5)), float(np.percentile(bb2[:, i], 97.5))]}
                                       for i, c in enumerate(["speed", "careful"])}
    S["ddm_boundary_ratio_prediction"] = {"speed/baseline": 2.089 / 2.501, "careful/baseline": 2.778 / 2.501}

    # (4) correct vs wrong at matched length within cell
    wc = within(df, ["H", "correct", "logt"], "cell")
    b3, _ = ols(wc.H.values, wc[["correct", "logt"]].values)
    bb3 = boot_slopes(df, ["correct", "logt"], "cell", n=400)
    S["correct_minus_wrong_at_matched_length"] = {"est": float(b3[0]), "ci95": [float(np.percentile(bb3[:, 0], 2.5)), float(np.percentile(bb3[:, 0], 97.5))]}
    S["H_mean_correct_wrong"] = {c: {"correct": float(df[(df.cond == c) & (df.correct == 1)].H.mean()), "wrong": float(df[(df.cond == c) & (df.correct == 0)].H.mean())} for c in ORDER}

    out = args.run / "analysis"
    out.mkdir(exist_ok=True)
    (out / "stop_height.json").write_text(json.dumps(S, indent=1, default=float) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colours = {"speed": "#e76f51", "baseline": "#264653", "careful": "#2a9d8f"}
    for c in ORDER:
        p = prof[prof.cond == c]
        axes[0].plot(p.t, p.H_adj, "o-", color=colours[c], label=c)
        axes[1].plot(p.t, p.H_raw, "o-", color=colours[c], label=c)
    grand = df.H.mean()
    tt = np.linspace(0.3, 4.5, 100)
    axes[0].plot(tt, grand + S["shape"]["log"]["coef"]["logt"] * (np.log(tt) - df.logt.mean()), "k--", lw=1, label="log fit (within cell)")
    axes[0].set_title("stop height vs thinking length, within prompt x condition", fontsize=9)
    axes[1].set_title("raw means (mixes easy and hard prompts)", fontsize=9)
    for ax in axes:
        ax.set_xlabel("thinking tokens (thousands)")
        ax.set_ylabel("|X| at the stop (nats)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_stop_height.png", dpi=150)

    print(f"{n} traces, {ncell} cells; sign(X) matches the answer {S['sign_of_X_matches_answer']:.4f}; bounded readouts {S['bounded_readouts']}")
    print("median |X| at the stop:", {c: round(v, 2) for c, v in S["H_median_by_condition"].items()})
    print("\nboundary shape (within cell):")
    for name, v in S["shape"].items():
        print(f"  {name:<11} dBIC {v['delta_bic']:>8.1f}  within-R2 {v['r2_within']:.4f}  coef {({k: round(x, 3) for k, x in v['coef'].items()})}")
    print(f"  linear slope per 1,000 tokens {S['slope_linear_per_1000_tokens']['est']:+.3f} {np.round(S['slope_linear_per_1000_tokens']['ci95'], 3)}; log slope {S['slope_log']['est']:+.3f} {np.round(S['slope_log']['ci95'], 3)}")
    print("  by condition:", {c: {k: round(x, 3) for k, x in v.items()} for c, v in S["slope_by_condition"].items()})
    print("\nprofile (within cell), thinking length k tokens -> |X|:")
    for c in ORDER:
        print(f"  {c:<9}", "  ".join(f"{r['t_median_k']:.2f}:{r['H_within_cell']:.2f}" for r in S["profile"][c]))
    print("\ncondition shift of |X| (nats, vs baseline):")
    print("  unadjusted      ", {c: (round(v["est"], 3), np.round(v["ci95"], 3).tolist()) for c, v in S["condition_shift_unadjusted"].items()})
    print("  at matched length", {c: (round(v["est"], 3), np.round(v["ci95"], 3).tolist()) for c, v in S["condition_shift_at_matched_length"].items()})
    print("  median ratio per prompt:", {k: round(v, 3) for k, v in S["condition_ratio_unadjusted"].items() if k != "mean_H"}, "| DDM boundary ratio:", {k: round(v, 3) for k, v in S["ddm_boundary_ratio_prediction"].items()})
    print("\ncorrect - wrong at matched length (within cell):", round(S["correct_minus_wrong_at_matched_length"]["est"], 3), np.round(S["correct_minus_wrong_at_matched_length"]["ci95"], 3))
    print("mean |X| correct / wrong:", {c: {k: round(x, 2) for k, x in v.items()} for c, v in S["H_mean_correct_wrong"].items()})
    print(f"-> {out}/stop_height.json, fig_stop_height.png")


if __name__ == "__main__":
    main()
