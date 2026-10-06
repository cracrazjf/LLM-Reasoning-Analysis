"""What the readout paths say about deliberation and stopping.

    python src/path_analysis.py --run runs/medxpertqa/main-qwen3-8b

Uses <run>/analysis/paths.npz (src/paths.py): X, the forced-answer log-odds after every sentence,
and the verdict sentences. X_c is X signed toward the correct option. Every trace is split at its
first verdict sentence into a review phase (before it) and a verdict phase (from it to the stop).

1. shape       where X moves: change of X in verdict sentences vs all others; X around a verdict;
               variance of X(k + lag) - X(k) inside the review phase (a random walk grows in
               proportion to the lag, a level with noise stays flat); mean review path of X_c;
               review slope per pair against the fitted DDM drift (analysis/ddm_fit.json)
2. stopping    logistic models with one intercept per prompt and condition, comparing thinking
               length (log tokens) and |X| as predictors of
                 first    the first verdict comes in the next sentence (states before any verdict)
                 stop     the thinking ends after this verdict (states at verdict sentences);
                          also with |X| read one sentence before the verdict
                 final    the final verdict comes in the next sentence (every state before it)
               and, per prompt and condition, how well each predictor alone separates events
               from non-events (AUC; 0.5 = not at all)
3. conditions  per prompt (pair x order) and condition: phase lengths, review slope, X_c before
               the first verdict, accuracy of the first verdict and of the answer, stop heights;
               differences to baseline within prompt, CI by bootstrap over questions; and the
               stopping rule per condition (stop ~ |X| with condition offsets)
Writes <run>/analysis/path_analysis.json and path_traces.csv (one row per trace).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
from paths import load_paths  # noqa: E402

CONDS = ["speed", "baseline", "careful"]
LAGS = (1, 2, 4, 8, 16, 24)
X_BINS = [0, 2, 4, 6, 8, 10, 100]
T_BINS = [0, 500, 1000, 1500, 2500, 100000]


def fe_logit(y: np.ndarray, Z: np.ndarray, strata: np.ndarray, iters: int = 60) -> tuple[np.ndarray, np.ndarray, float]:
    """Logistic regression with one intercept per stratum (Newton steps on the arrowhead Hessian).

    Strata without both outcomes carry no information and are dropped. Returns coefficients,
    standard errors and the log-likelihood.
    """
    s = np.unique(strata, return_inverse=True)[1]
    ny, nn = np.bincount(s, weights=y), np.bincount(s)
    keep = ((ny > 0) & (ny < nn))[s]
    y, Z, s = y[keep].astype(float), Z[keep], np.unique(s[keep], return_inverse=True)[1]
    S, p = s.max() + 1, Z.shape[1]
    ny, nn = np.bincount(s, weights=y), np.bincount(s)
    if p == 0:
        return np.zeros(0), np.zeros(0), float(np.sum(ny * np.log(ny / nn) + (nn - ny) * np.log(1 - ny / nn)))
    beta, alpha = np.zeros(p), np.log((ny + 0.5) / (nn - ny + 0.5))
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(alpha[s] + Z @ beta)))
        w, r = mu * (1 - mu), y - mu
        Haa = np.bincount(s, weights=w, minlength=S) + 1e-12
        Hba = np.stack([np.bincount(s, weights=w * Z[:, j], minlength=S) for j in range(p)])
        schur = (Z * w[:, None]).T @ Z - (Hba / Haa) @ Hba.T
        ga = np.bincount(s, weights=r, minlength=S)
        db = np.linalg.solve(schur, Z.T @ r - (Hba / Haa) @ ga)
        da = (ga - Hba.T @ db) / Haa
        step = min(1.0, 5 / max(np.max(np.abs(db)), 1e-12))
        beta, alpha = beta + step * db, alpha + step * da
        if np.max(np.abs(db)) < 1e-8 and np.max(np.abs(da)) < 1e-6:
            break
    eta = alpha[s] + Z @ beta
    return beta, np.sqrt(np.diag(np.linalg.inv(schur))), float(np.sum(y * eta - np.logaddexp(0, eta)))


def auc_by_stratum(y: np.ndarray, x: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """AUC of x for y within each stratum that has both outcomes."""
    order = np.argsort(strata, kind="stable")
    y, x, s = y[order], x[order], strata[order]
    out = []
    for a, b in zip(np.r_[0, np.flatnonzero(np.diff(s)) + 1], np.r_[np.flatnonzero(np.diff(s)) + 1, len(s)]):
        n1 = y[a:b].sum()
        if 0 < n1 < b - a:
            out.append((rankdata(x[a:b])[y[a:b] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (b - a - n1)))
    return np.array(out)


class Paths:
    """Flat sentence arrays plus per-trace indices."""

    def __init__(self, run: Path) -> None:
        T, A = load_paths(run)
        self.T = T
        self.t, self.X, self.V = A["t"].astype(float), A["X"].astype(float), A["verdict"]
        self.n, self.off = T.n.values, T.off.values
        N = len(T)
        self.tr = np.repeat(np.arange(N), self.n)
        self.k = np.arange(len(self.t)) - self.off[self.tr]
        self.isv = self.V > 0
        self.vs = np.where(self.V == 1, 1.0, np.where(self.V == 2, -1.0, 0.0))  # +1: verdict for A
        idx = np.where(self.isv)[0]
        fv = np.full(N, 10 ** 9)
        np.minimum.at(fv, self.tr[idx], self.k[idx])
        self.fv = np.where(fv == 10 ** 9, self.n, fv)      # sentence index of the first verdict (n: none)
        self.lv = np.full(N, -1)
        np.maximum.at(self.lv, self.tr[idx], self.k[idx])  # last verdict (-1: none)
        self.lab = np.where(T.label.values == "A", 1.0, -1.0)
        self.ans = np.where(T.answer.values == "A", 1.0, np.where(T.answer.values == "B", -1.0, np.nan))
        self.end = self.off + self.n - 1
        self.ends_on_verdict = self.lv >= self.n - 2       # the stop is a verdict or one sentence after it

    def features(self) -> pd.DataFrame:
        T, t, X, off, n, fv, lv, lab = self.T.copy(), self.t, self.X, self.off, self.n, self.fv, self.lv, self.lab
        has, review = fv < n, (fv < n) & (fv > 0)
        before_first, first = np.maximum(off + fv - 1, off), off + np.minimum(fv, n - 1)
        last = off + np.maximum(lv, 0)
        T["t_total"] = t[self.end]
        T["t_review"] = np.where(review, t[before_first], np.nan)          # tokens before the first verdict sentence
        T["t_verdict_phase"] = T.t_total - T.t_review
        T["n_verdicts"] = np.add.reduceat(self.isv.astype(int), off)
        T["X0_c"] = T.X0.astype(float) * lab
        T["Xc_first_sentence"] = X[off] * lab
        T["Xc_before_first_verdict"] = np.where(review, X[before_first] * lab, np.nan)
        span = (T.t_review - t[off]).where(T.t_review - t[off] > 100)
        T["review_slope"] = (T.Xc_before_first_verdict - T.Xc_first_sentence) / span * 1000   # nats per 1,000 tokens
        T["absX_before_first_verdict"] = np.where(review, np.abs(X[before_first]), np.nan)
        T["first_verdict_correct"] = np.where(has, (self.vs[first] == lab).astype(float), np.nan)
        T["answer_correct"] = T.correct.astype(float)
        T["changed_after_first_verdict"] = np.where(has & T.correct.notna(), (T.first_verdict_correct != T.answer_correct).astype(float), np.nan)
        T["absX_before_last_verdict"] = np.where(has & (lv > 0), np.abs(X[np.maximum(last - 1, off)]), np.nan)
        T["absX_after_last_verdict"] = np.where(has, np.abs(X[last]), np.nan)
        T["absX_stop"] = np.abs(X[self.end])
        return T


def shape(P: Paths, F: pd.DataFrame, run: Path) -> dict:
    X, k, n, off, tr, fv = P.X, P.k, P.n, P.off, P.tr, P.fv
    dX = np.where(k == 0, np.nan, X - np.roll(X, 1))
    dXf = dX * P.ans[tr]
    total = np.nansum(np.where(k > 0, dXf, 0.0)) / len(n)
    in_verdicts = np.nansum(np.where(P.isv & (k > 0), dXf, 0.0)) / len(n)
    in_review = np.nansum(np.where((k < fv[tr]) & (k > 0), dXf, 0.0)) / len(n)
    out = {
        "sentences": int(len(X)), "verdict_share_of_sentences": float(P.isv.mean()),
        "median_abs_change": {"verdict": float(np.nanmedian(np.abs(dX[P.isv]))), "other": float(np.nanmedian(np.abs(dX[~P.isv])))},
        "verdict_moves_toward_its_letter": float(np.nanmean((dX * P.vs)[P.isv] > 0)),
        "toward_final_answer": {"after_first_sentence": float(np.nanmean(X[off] * P.ans)), "at_stop": float(np.nanmean(X[P.end] * P.ans)),
                                "total_change": float(total), "in_verdict_sentences": float(in_verdicts),
                                "in_other_sentences": float(total - in_verdicts), "in_review_phase": float(in_review)},
    }
    sel = np.where(P.isv & (k >= 4) & (k < n[tr] - 6))[0]
    firsts = np.array([off[i] + fv[i] for i in range(len(n)) if 4 <= fv[i] < n[i] - 6])
    lags = list(range(-4, 7))
    out["around_verdict_toward_its_letter"] = {
        "lags": lags, "all_verdicts": [float(np.nanmean(X[sel + j] * P.vs[sel])) for j in lags],
        "first_verdict": [float(np.nanmean(X[firsts + j] * P.vs[firsts])) for j in lags]}
    segs = [(off[i], off[i] + fv[i]) for i in range(len(n)) if fv[i] >= 30]
    out["review_increments"] = {
        "traces": len(segs), "lags": list(LAGS),
        "variance": [float(np.nanvar(np.concatenate([X[a + h:b] - X[a:b - h] for a, b in segs]))) for h in LAGS],
        "mean_toward_correct": [float(np.nanmean(np.concatenate([(X[a + h:b] - X[a:b - h]) * P.lab[tr[a]] for a, b in segs]))) for h in LAGS]}
    long = [i for i in range(len(n)) if fv[i] >= 40]
    M = np.array([X[off[i]:off[i] + 40] * P.lab[i] for i in long])
    out["review_mean_path_toward_correct"] = {"traces": len(long), "sentence": [1, 2, 3, 5, 10, 20, 30, 40],
                                              "X_c": [float(M[:, j - 1].mean()) for j in (1, 2, 3, 5, 10, 20, 30, 40)]}
    out["start"] = {"X0_c": float(F.X0_c.mean()), "Xc_first_sentence": float(F.Xc_first_sentence.mean()),
                    "corr_X0_first_sentence": float(np.corrcoef(F.X0.astype(float), X[off])[0, 1]),
                    "first_sentence_on_correct_side": float((F.Xc_first_sentence > 0).mean())}
    fit = run / "analysis" / "ddm_fit.json"
    drift = json.loads(fit.read_text()).get("drift_per_pair") if fit.exists() else None
    if drift:
        pp = F[F.cond == "baseline"].groupby("pair_id").agg(slope=("review_slope", "mean"), acc=("answer_correct", "mean")).dropna()
        pp["drift"] = pp.index.map(drift)
        pp = pp.dropna()
        out["review_slope_vs_ddm_drift"] = {"pairs": len(pp), "pearson": float(pp.slope.corr(pp.drift)),
                                           "spearman": float(pp.slope.corr(pp.drift, method="spearman")),
                                           "slope_vs_accuracy_spearman": float(pp.slope.corr(pp.acc, method="spearman"))}
    return out


def stopping(P: Paths) -> dict:
    X, t, k, n, tr = P.X, P.t, P.k, P.n, P.tr
    strata = pd.factorize(P.T.prompt_id)[0][tr]            # a prompt_id is one prompt in one condition
    aX, aX_before, lt = np.abs(X), np.abs(np.roll(X, 1)), np.log(np.maximum(t, 1.0))
    ok = ~np.isnan(X)

    def z(v: np.ndarray) -> np.ndarray:
        return (v - v.mean()) / v.std()

    def compare(rows: np.ndarray, y: np.ndarray, x: np.ndarray) -> dict:
        yy, s, zx, zt = y[rows].astype(float), strata[rows], z(x[rows]), z(lt[rows])
        ll0 = fe_logit(yy, np.zeros((len(yy), 0)), s)[2]
        gain = {name: fe_logit(yy, Z, s)[2] - ll0 for name, Z in
                (("time", np.c_[zt, zt ** 2]), ("absX", np.c_[zx, zx ** 2]), ("both", np.c_[zt, zt ** 2, zx, zx ** 2]))}
        b, se, _ = fe_logit(yy, np.c_[zt, zx], s)
        auc_x, auc_t = auc_by_stratum(yy, zx, s), auc_by_stratum(yy, zt, s)
        return {"decision_points": int(len(yy)), "events": int(yy.sum()),
                "auc_per_prompt_and_condition": {
                    "cells": int(len(auc_x)), "absX_median": float(np.median(auc_x)), "time_median": float(np.median(auc_t)),
                    "absX_above_half": float((auc_x > 0.5).mean()), "time_above_half": float((auc_t > 0.5).mean()),
                    "absX_beats_time": float((auc_x > auc_t).mean())},
                "loglik_gain": {**gain, "unique_to_absX": gain["both"] - gain["time"], "unique_to_time": gain["both"] - gain["absX"]},
                "linear_per_sd": {"log_time": float(b[0]), "log_time_se": float(se[0]), "absX": float(b[1]), "absX_se": float(se[1])},
                "sd": {"log_time": float(lt[rows].std()), "absX": float(x[rows].std())},
                "absX_median": {"event": float(np.median(x[rows][yy == 1])), "no_event": float(np.median(x[rows][yy == 0]))}}

    first = np.where((k < P.fv[tr]) & (k < n[tr] - 1) & ok)[0]
    at_verdict = np.where(P.isv & ok & P.ends_on_verdict[tr])[0]
    before_final = np.where((k < P.lv[tr]) & P.ends_on_verdict[tr] & ok)[0]
    out = {"first": compare(first, k + 1 == P.fv[tr], aX),
           "stop": compare(at_verdict, k == P.lv[tr], aX),
           "stop_absX_before_verdict": compare(at_verdict[k[at_verdict] > 0], k == P.lv[tr], aX_before),
           "final": compare(before_final, k + 1 == P.lv[tr], aX)}
    ev = first[(k + 1 == P.fv[tr])[first]]
    out["first"]["X_toward_first_verdict_one_sentence_before"] = {
        "median": float(np.median(X[ev] * P.vs[ev + 1])), "share_on_its_side": float(np.mean(X[ev] * P.vs[ev + 1] > 0))}
    d = pd.DataFrame({"stop": (k == P.lv[tr])[at_verdict].astype(float), "absX": pd.cut(aX[at_verdict], X_BINS, right=False),
                      "tokens": pd.cut(t[at_verdict], T_BINS), "cond": P.T.cond.values[tr[at_verdict]]})
    out["stop_rate_by_absX_and_tokens"] = {str(a): {str(c): float(v) for c, v in row.items()} for a, row in
                                           d.pivot_table(index="absX", columns="tokens", values="stop", observed=True).iterrows()}
    out["stop_rate_by_absX_and_condition"] = {str(a): {c: float(v) for c, v in row.items()} for a, row in
                                              d.pivot_table(index="absX", columns="cond", values="stop", observed=True).iterrows()}
    item = pd.factorize(P.T.item_id)[0][tr[at_verdict]]
    cd = P.T.cond.values[tr[at_verdict]]
    if set(CONDS) <= set(cd):
        b, se, _ = fe_logit(d.stop.values, np.c_[aX[at_verdict], cd == "speed", cd == "careful"].astype(float), item)
        out["stopping_rule_by_condition"] = {
            "per_nat": float(b[0]), "speed_offset": float(b[1]), "speed_offset_se": float(se[1]), "careful_offset": float(b[2]),
            "careful_offset_se": float(se[2]), "speed_shift_nats": float(-b[1] / b[0]), "careful_shift_nats": float(-b[2] / b[0])}
    return out


MEASURES = ["answer_correct", "first_verdict_correct", "changed_after_first_verdict", "t_total", "t_review", "t_verdict_phase",
            "n_verdicts", "review_slope", "X0_c", "Xc_first_sentence", "Xc_before_first_verdict", "absX_before_first_verdict",
            "absX_before_last_verdict", "absX_after_last_verdict", "absX_stop"]
MEDIANS = {"t_total", "t_review", "t_verdict_phase", "n_verdicts", "absX_before_first_verdict", "absX_before_last_verdict",
           "absX_after_last_verdict", "absX_stop"}


def conditions(F: pd.DataFrame, n_boot: int = 2000) -> dict:
    conds = [c for c in CONDS if c in set(F.cond)]
    cell = F.groupby(["item_id", "source_id", "cond"]).agg({m: ("median" if m in MEDIANS else "mean") for m in MEASURES}).unstack("cond")
    q = cell.index.get_level_values("source_id").values
    by_q = {u: np.where(q == u)[0] for u in np.unique(q)}
    rng = np.random.default_rng(0)

    def ci(v: np.ndarray) -> list[float]:
        b = [np.nanmean(np.concatenate([v[by_q[u]] for u in rng.choice(list(by_q), len(by_q))])) for _ in range(n_boot)]
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    out = {"prompts": int(len(cell)), "questions": len(by_q), "measures": {}}
    for m in MEASURES:
        e = {c: float(np.nanmean(cell[(m, c)].values)) for c in conds}
        for c in conds:
            if c != "baseline" and "baseline" in conds:
                d = cell[(m, c)].values - cell[(m, "baseline")].values
                e[f"{c}_minus_baseline"] = {"mean": float(np.nanmean(d)), "ci": ci(d)}
        out["measures"][m] = e
    out["final_given_first_verdict"] = {}
    for c in conds:
        g = F[(F.cond == c) & F.first_verdict_correct.notna() & F.answer_correct.notna()]
        out["final_given_first_verdict"][c] = {"first_correct": float(g[g.first_verdict_correct == 1].answer_correct.mean()),
                                               "first_wrong": float(g[g.first_verdict_correct == 0].answer_correct.mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    P = Paths(args.run)
    F = P.features()
    res = {"traces": int(len(F)), "shape": shape(P, F, args.run), "stopping": stopping(P), "conditions": conditions(F)}
    out = args.run / "analysis"
    F.drop(columns=["off"]).to_csv(out / "path_traces.csv", index=False)
    (out / "path_analysis.json").write_text(json.dumps(res, indent=1) + "\n")

    s, h, c = res["shape"], res["stopping"], res["conditions"]
    tf = s["toward_final_answer"]
    print(f"{res['traces']} traces, {s['sentences']} sentences, {s['verdict_share_of_sentences']:.3f} of them verdicts")
    print(f"\n1. shape. X toward the final answer: {tf['after_first_sentence']:+.2f} after sentence 1, {tf['at_stop']:+.2f} at the stop; "
          f"of the change {tf['total_change']:+.2f}: verdict sentences {tf['in_verdict_sentences']:+.2f}, others {tf['in_other_sentences']:+.2f} "
          f"(review phase {tf['in_review_phase']:+.2f})")
    print("   around a first verdict (toward its letter), lags -4..+6: " + " ".join(f"{v:+.2f}" for v in s["around_verdict_toward_its_letter"]["first_verdict"]))
    print("   review increments, variance by lag " + ", ".join(f"{lag}: {v:.2f}" for lag, v in zip(LAGS, s["review_increments"]["variance"])))
    print("   review mean X_c by sentence " + ", ".join(f"{j}: {v:+.2f}" for j, v in zip(s["review_mean_path_toward_correct"]["sentence"], s["review_mean_path_toward_correct"]["X_c"])))
    print(f"   start: X0_c {s['start']['X0_c']:+.2f}, after sentence 1 {s['start']['Xc_first_sentence']:+.2f}, correlation {s['start']['corr_X0_first_sentence']:.2f}")
    if "review_slope_vs_ddm_drift" in s:
        r = s["review_slope_vs_ddm_drift"]
        print(f"   review slope vs fitted DDM drift over {r['pairs']} pairs: Pearson {r['pearson']:.2f}, Spearman {r['spearman']:.2f}")
    print("\n2. stopping. log-likelihood gain over prompt-and-condition intercepts")
    for name in ("first", "stop", "stop_absX_before_verdict", "final"):
        g = h[name]["loglik_gain"]
        print(f"   {name:<25} events {h[name]['events']:>6} / {h[name]['decision_points']:>8}   time {g['time']:>8.0f}   |X| {g['absX']:>8.0f}   "
              f"unique to time {g['unique_to_time']:>8.0f}   unique to |X| {g['unique_to_absX']:>8.0f}")
        a = h[name]["auc_per_prompt_and_condition"]
        print(f"   {'':<25} per cell ({a['cells']}): AUC time {a['time_median']:.2f}, |X| {a['absX_median']:.2f}; |X| better than time in {a['absX_beats_time']:.2f} of cells")
    print("   stop rate after a verdict by |X| (rows) and tokens so far (columns):")
    for a, row in h["stop_rate_by_absX_and_tokens"].items():
        print(f"     {a:<12}" + " ".join(f"{v:.3f}" for v in row.values()))
    if "stopping_rule_by_condition" in h:
        r = h["stopping_rule_by_condition"]
        print(f"   stopping rule: {r['per_nat']:+.3f} per nat; at equal |X| speed needs {r['speed_shift_nats']:+.2f} nats, careful {r['careful_shift_nats']:+.2f} nats more than baseline")
    print(f"\n3. conditions ({c['prompts']} prompts, {c['questions']} questions)")
    conds = [x for x in CONDS if x in c["measures"]["t_total"]]
    print(f"   {'measure':<28}" + "".join(f"{x:>10}" for x in conds))
    for m, e in c["measures"].items():
        diffs = "".join(f"   {x} {e[x + '_minus_baseline']['mean']:+.3f} [{e[x + '_minus_baseline']['ci'][0]:+.3f}, {e[x + '_minus_baseline']['ci'][1]:+.3f}]"
                        for x in conds if x + "_minus_baseline" in e)
        print(f"   {m:<28}" + "".join(f"{e[x]:>10.3f}" for x in conds) + diffs)
    for x, e in c["final_given_first_verdict"].items():
        print(f"   {x:<9} answer correct given first verdict correct {e['first_correct']:.3f}, given first verdict wrong {e['first_wrong']:.3f}")
    print(f"-> {out}/path_analysis.json, path_traces.csv")


if __name__ == "__main__":
    main()
