"""Fit drift-diffusion models to the main run: choice = correct / wrong, time = thinking tokens.

    python src/ddm_fit.py --run runs/medxpertqa/main-qwen3-8b

Model. Evidence starts between two boundaries and drifts toward one of them with noise
(diffusion coefficient 1; time unit = 1,000 thinking tokens). Upper boundary = the correct option.
  v   drift rate, one per pair (how strongly the evidence favours the correct diagnosis)
  sv  across-trace variability of the drift (normal); gives errors that are slower than corrects
  a   boundary separation (caution): larger = slower and more accurate
  t0  non-decision time: thinking tokens that are not accumulation (set-up, wrap-up)
  d   start-point offset toward the option labelled B, in evidence units (start = a/2 +- d)
A fixed 2% of traces are treated as contaminants (uniform in time, either answer).

Density: Navarro & Fuss (2009) series for the first-passage time, with the closed-form
integral over a normal drift (as in HDDM). Fit by maximum likelihood (L-BFGS-B).

Model comparison: which parameter does the instruction change?
  M0   nothing                         Mat  a and t0
  Ma   boundary a                      Mav  a and drift scale
  Mt   non-decision time t0            Mvt  drift scale and t0
  Mv   drift (one multiplier k)        Mavt all three
Writes <run>/analysis/ddm_fit.json and fig_ddm_fit.png (observed vs predicted accuracy and
thinking-time quantiles for correct and wrong traces).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from helps_analysis import load  # noqa: E402

CONDS = ["speed", "baseline", "careful"]
P_CONTAM = 0.02
MODELS = {"M0": (), "Ma": ("a",), "Mt": ("t0",), "Mv": ("v",), "Mat": ("a", "t0"), "Mav": ("a", "v"),
          "Mvt": ("v", "t0"), "Mavt": ("a", "v", "t0")}


# ------------------------------------------------------------- density


def f01(u: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Zero-drift, unit-boundary first-passage density at the lower barrier (Navarro & Fuss 2009)."""
    u = np.maximum(u, 1e-12)
    ks = np.arange(-5, 6)[:, None]
    small = ((w + 2 * ks) * np.exp(-((w + 2 * ks) ** 2) / (2 * u))).sum(0) / np.sqrt(2 * np.pi * u ** 3)
    kl = np.arange(1, 9)[:, None]
    large = np.pi * (kl * np.exp(-(kl ** 2) * np.pi ** 2 * u / 2) * np.sin(kl * np.pi * w)).sum(0)
    return np.maximum(np.where(u < 1.0, small, large), 0.0)


def wfpt_lower(t: np.ndarray, v: np.ndarray, sv: float, a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Density of hitting the lower barrier at decision time t; drift ~ N(v, sv^2), start w*a."""
    ok = t > 0
    tt = np.where(ok, t, 1.0)
    dens = f01(tt / a ** 2, w) / a ** 2
    dens = dens * np.exp((sv ** 2 * a ** 2 * w ** 2 - 2 * a * v * w - v ** 2 * tt) / (2 * (1 + sv ** 2 * tt))) / np.sqrt(1 + sv ** 2 * tt)
    return np.where(ok, dens, 0.0)


def p_upper(v: np.ndarray, sv: float, a: np.ndarray, w: np.ndarray, n: int = 21) -> np.ndarray:
    """Probability of the upper (correct) boundary, integrating the drift distribution."""
    x, wt = np.polynomial.hermite_e.hermegauss(n)
    wt = wt / wt.sum()
    out = np.zeros_like(v, dtype=float)
    for xi, wi in zip(x, wt):
        d = v + sv * xi
        d = np.where(np.abs(d) < 1e-6, 1e-6, d)
        out += wi * (1 - np.exp(-2 * d * a * w)) / (1 - np.exp(-2 * d * a))
    return out


# --------------------------------------------------------------- model


class Data:
    def __init__(self, df: pd.DataFrame) -> None:
        d = df.dropna(subset=["correct"])
        d = d[(d.finish_reason != "length") & (d.think_tokens > 0)]
        self.pairs = sorted(d.pair_id.unique())
        self.pair = d.pair_id.map({p: i for i, p in enumerate(self.pairs)}).values
        self.cond = d.cond.map({c: i for i, c in enumerate(CONDS)}).values
        self.sign = np.where(d.order.values == "ba", 1.0, -1.0)  # +1: the correct option is B
        self.correct = d.correct.values.astype(bool)
        self.rt = d.think_tokens.values / 1000.0
        self.tmax = float(self.rt.max())
        self.n = len(d)
        self.frame = d


class Model:
    def __init__(self, data: Data, varies: tuple[str, ...]) -> None:
        self.d, self.varies = data, varies
        self.np_ = len(data.pairs)
        self.names = [f"v[{p}]" for p in data.pairs] + ["log_sv", "delta"]
        self.names += [f"log_a[{c}]" for c in CONDS] if "a" in varies else ["log_a"]
        self.names += [f"log_t0[{c}]" for c in CONDS] if "t0" in varies else ["log_t0"]
        if "v" in varies:
            self.names += ["log_k[speed]", "log_k[careful]"]
        self.k = len(self.names)

    def unpack(self, th: np.ndarray) -> dict:
        i = self.np_
        v = th[:i]
        sv, delta = np.exp(th[i]), th[i + 1]
        i += 2
        if "a" in self.varies:
            a = np.exp(th[i:i + 3]); i += 3
        else:
            a = np.repeat(np.exp(th[i]), 3); i += 1
        if "t0" in self.varies:
            t0 = np.exp(th[i:i + 3]); i += 3
        else:
            t0 = np.repeat(np.exp(th[i]), 3); i += 1
        k = np.ones(3)
        if "v" in self.varies:
            k = np.array([np.exp(th[i]), 1.0, np.exp(th[i + 1])])
        return {"v": v, "sv": sv, "delta": delta, "a": a, "t0": t0, "k": k}

    def init(self) -> np.ndarray:
        d = self.d
        acc = pd.Series(d.correct).groupby(d.pair).mean().clip(0.03, 0.97).values
        v0 = np.log(acc / (1 - acc)) / 2.0
        th = list(v0) + [np.log(0.6), 0.0]
        th += [np.log(2.0)] * (3 if "a" in self.varies else 1)
        th += [np.log(0.25)] * (3 if "t0" in self.varies else 1)
        if "v" in self.varies:
            th += [0.0, 0.0]
        return np.array(th)

    def loglik_terms(self, th: np.ndarray) -> np.ndarray:
        p, d = self.unpack(th), self.d
        a = p["a"][d.cond]
        v = p["v"][d.pair] * p["k"][d.cond]
        w = np.clip(0.5 + d.sign * p["delta"] / a, 0.02, 0.98)
        t = d.rt - p["t0"][d.cond]
        dens = np.where(d.correct, wfpt_lower(t, -v, p["sv"], a, 1 - w), wfpt_lower(t, v, p["sv"], a, w))
        return np.log((1 - P_CONTAM) * dens + P_CONTAM * 0.5 / d.tmax)

    def nll_grad(self, th: np.ndarray) -> tuple[float, np.ndarray]:
        base = self.loglik_terms(th)
        f0 = -base.sum()
        g = np.zeros_like(th)
        eps = 1e-5
        # all pair drifts at once: each trace depends on its own pair's drift only
        th2 = th.copy(); th2[:self.np_] += eps
        diff = self.loglik_terms(th2) - base
        g[:self.np_] = -np.bincount(self.d.pair, weights=diff, minlength=self.np_) / eps
        for j in range(self.np_, len(th)):
            th2 = th.copy(); th2[j] += eps
            g[j] = -(self.loglik_terms(th2).sum() - base.sum()) / eps
        return f0, g

    def fit(self, start: np.ndarray | None = None) -> dict:
        th0 = self.init() if start is None else start
        best = None
        for attempt in range(2):
            r = minimize(self.nll_grad, th0, jac=True, method="L-BFGS-B", options={"maxiter": 800, "maxfun": 4000, "ftol": 1e-10})
            if best is None or r.fun < best.fun:
                best = r
            th0 = best.x + np.random.default_rng(attempt).normal(0, 0.02, len(th0))
        return {"theta": best.x, "nll": float(best.fun), "k": self.k, "converged": bool(best.success)}


def embed(simple: Model, th: np.ndarray, target: Model) -> np.ndarray:
    """Start values for a richer model from a fitted simpler one."""
    ps = simple.unpack(th)
    out = list(ps["v"]) + [np.log(ps["sv"]), ps["delta"]]
    out += list(np.log(ps["a"])) if "a" in target.varies else [np.log(ps["a"][1])]
    out += list(np.log(ps["t0"])) if "t0" in target.varies else [np.log(ps["t0"][1])]
    if "v" in target.varies:
        out += [np.log(ps["k"][0]), np.log(ps["k"][2])]
    return np.array(out)


# ------------------------------------------------------------------ fit quality


def predictions(model: Model, th: np.ndarray) -> pd.DataFrame:
    """Per cell (pair, order, condition): predicted accuracy and time quantiles for correct and wrong traces."""
    p, d = model.unpack(th), model.d
    grid = np.linspace(0.005, min(d.tmax, 15.0), 1500)
    dt = grid[1] - grid[0]
    cells = d.frame.groupby(["pair_id", "order", "cond"]).agg(n=("correct", "size"), acc=("correct", "mean")).reset_index()
    rows = []
    pidx = {q: i for i, q in enumerate(d.pairs)}
    for r in cells.itertuples():
        c = CONDS.index(r.cond)
        a, t0 = p["a"][c], p["t0"][c]
        v = p["v"][pidx[r.pair_id]] * p["k"][c]
        s = 1.0 if r.order == "ba" else -1.0
        w = float(np.clip(0.5 + s * p["delta"] / a, 0.02, 0.98))
        t = grid - t0
        fc = wfpt_lower(t, np.full_like(t, -v), p["sv"], np.full_like(t, a), np.full_like(t, 1 - w))
        fe = wfpt_lower(t, np.full_like(t, v), p["sv"], np.full_like(t, a), np.full_like(t, w))
        rows.append({"pair_id": r.pair_id, "order": r.order, "cond": r.cond, "n": r.n, "acc_obs": r.acc,
                     "acc_pred": float(fc.sum() * dt / max((fc.sum() + fe.sum()) * dt, 1e-12)), "fc": fc * dt, "fe": fe * dt})
    return pd.DataFrame(rows), grid


def quantile_table(model: Model, th: np.ndarray, out: Path) -> dict:
    cells, grid = predictions(model, th)
    d = model.d.frame
    res = {}
    qs = (0.1, 0.3, 0.5, 0.7, 0.9)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, cond in zip(axes, CONDS):
        g = cells[cells.cond == cond]
        wts = g.n.values / g.n.sum()
        fc = (np.stack(g.fc.values) * wts[:, None]).sum(0)
        fe = (np.stack(g.fe.values) * wts[:, None]).sum(0)
        pc = fc.sum() / (fc.sum() + fe.sum())
        o = d[d.cond == cond]
        entry = {"acc_obs": float(o.correct.mean()), "acc_pred": float(pc)}
        for name, f, obs in (("correct", fc, o[o.correct == 1].think_tokens.values / 1000), ("wrong", fe, o[o.correct == 0].think_tokens.values / 1000)):
            cdf = np.cumsum(f) / f.sum()
            entry[name] = {"obs": [float(np.quantile(obs, q)) for q in qs], "pred": [float(grid[np.searchsorted(cdf, q)]) for q in qs]}
            ax.plot(entry[name]["obs"], qs, "o", color="#2a9d8f" if name == "correct" else "#e76f51", label=f"{name} observed")
            ax.plot(entry[name]["pred"], qs, "-", color="#2a9d8f" if name == "correct" else "#e76f51", label=f"{name} model")
        ax.set_title(f"{cond}: accuracy {entry['acc_obs']:.3f} observed, {entry['acc_pred']:.3f} model", fontsize=9)
        ax.set_xlabel("thinking tokens (thousands)")
        res[cond] = entry
    axes[0].set_ylabel("cumulative proportion")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_ddm_fit.png", dpi=150)
    res["pair_accuracy_correlation"] = float(np.corrcoef(cells.acc_obs, cells.acc_pred)[0, 1])
    # predicted share answering B: P(correct) when the correct option is B, P(wrong) when it is A
    cells["pB_pred"] = np.where(cells.order == "ba", cells.acc_pred, 1 - cells.acc_pred)
    cells["pB_obs"] = np.where(cells.order == "ba", cells.acc_obs, 1 - cells.acc_obs)
    res["share_B"] = {c: {"obs": float(cells[cells.cond == c].pB_obs.mean()), "pred": float(cells[cells.cond == c].pB_pred.mean())} for c in CONDS}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    data = Data(load(args.run))
    print(f"{data.n} traces, {len(data.pairs)} pairs; time unit = 1,000 thinking tokens; contaminant share fixed at {P_CONTAM}", flush=True)
    fits: dict[str, dict] = {}
    models: dict[str, Model] = {}
    order = [m for m in MODELS if m in args.models.split(",")]
    for name in order:
        t0 = time.time()
        m = Model(data, MODELS[name])
        start = None
        subs = [s for s in fits if set(MODELS[s]) < set(MODELS[name])]
        if subs:  # warm start from the best nested model already fitted
            s = min(subs, key=lambda q: fits[q]["nll"])
            start = embed(models[s], fits[s]["theta"], m)
        f = m.fit(start)
        f["aic"] = 2 * f["nll"] + 2 * f["k"]
        f["bic"] = 2 * f["nll"] + f["k"] * np.log(data.n)
        fits[name], models[name] = f, m
        p = m.unpack(f["theta"])
        print(f"{name:<5} k={f['k']:<3} -logL={f['nll']:.1f}  AIC={f['aic']:.1f}  BIC={f['bic']:.1f}  "
              f"a={np.round(p['a'], 3)} t0={np.round(p['t0'], 3)} k={np.round(p['k'], 3)} sv={p['sv']:.3f} delta={p['delta']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    best = min(fits, key=lambda q: fits[q]["bic"])
    out = args.run / "analysis"
    out.mkdir(exist_ok=True)
    gof = quantile_table(models[best], fits[best]["theta"], out)
    pbest = models[best].unpack(fits[best]["theta"])
    summary = {
        "n_traces": data.n, "n_pairs": len(data.pairs), "time_unit": "1000 thinking tokens", "contaminant_share": P_CONTAM,
        "conditions": CONDS,
        "models": {n: {"varies": list(MODELS[n]), "k": f["k"], "nll": f["nll"], "aic": f["aic"], "bic": f["bic"],
                       "delta_bic_vs_best": f["bic"] - fits[best]["bic"], "converged": f["converged"],
                       "a": models[n].unpack(f["theta"])["a"].tolist(), "t0": models[n].unpack(f["theta"])["t0"].tolist(),
                       "k_drift": models[n].unpack(f["theta"])["k"].tolist(), "sv": float(models[n].unpack(f["theta"])["sv"]),
                       "delta": float(models[n].unpack(f["theta"])["delta"])} for n, f in fits.items()},
        "best_by_bic": best,
        "best_drift_per_pair": {"mean": float(pbest["v"].mean()), "sd": float(pbest["v"].std()), "min": float(pbest["v"].min()), "max": float(pbest["v"].max())},
        "drift_per_pair": dict(zip(data.pairs, pbest["v"].tolist())),
        "fit_quality_best": gof,
    }
    (out / "ddm_fit.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"\nbest by BIC: {best}")
    for n, f in sorted(fits.items(), key=lambda kv: kv[1]["bic"]):
        print(f"  {n:<5} dBIC {f['bic'] - fits[best]['bic']:>9.1f}   dAIC {f['aic'] - min(x['aic'] for x in fits.values()):>9.1f}")
    print("\nfit quality of the best model (thousands of tokens; quantiles 0.1/0.3/0.5/0.7/0.9):")
    for c in CONDS:
        e = gof[c]
        print(f"  {c:<9} accuracy obs {e['acc_obs']:.3f} pred {e['acc_pred']:.3f}")
        for nm in ("correct", "wrong"):
            print(f"     {nm:<8} obs {np.round(e[nm]['obs'], 2)}  pred {np.round(e[nm]['pred'], 2)}")
    print("  share answering B:", {c: {k: round(v, 3) for k, v in x.items()} for c, x in gof["share_B"].items()})
    print("  correlation of observed and predicted cell accuracy:", round(gof["pair_accuracy_correlation"], 3))
    print(f"-> {out}/ddm_fit.json, fig_ddm_fit.png")


if __name__ == "__main__":
    main()
