"""Does the chosen option stop on one line? Levels of the chosen and the unchosen option at the stop.

    python src/analysis/stop_line.py --run runs/medxpertqa/main-qwen3-8b [--cond baseline]

Uses <run>/analysis/paths.npz with raw logits (readout.py --logits, then paths.py). Per trace the
chosen option is the letter answered after </think>. Three scales for each option:
  p        probability of the option as the forced answer (chosen + unchosen = 1, almost exactly)
  logit    raw logit (no fixed zero)
  rel      the logit minus a reference (--ref): raw (none; the default), mean (the mean logit over
           the whole vocabulary) or other (the level of the other stored candidates, runs read out
           before 2026-10-06 only)
read at the stop (after the last sentence) and before the final verdict sentence.

1. spread   SD of each level over traces, split into between prompts (pair x order) and within
            prompt; a common line means a small SD of the chosen level next to the unchosen one
2. length   within-prompt slope of each stop level on log thinking tokens
3. rule     at every verdict sentence: does the thinking end here? Logistic model with one
            intercept per prompt on the level of the verdict's letter and of the other letter
            (read after the verdict). A rule on the difference X gives equal and opposite
            coefficients; a rule on the stated option alone gives a zero for the other letter
Writes <run>/analysis/stop_line_<cond>.json and
  stop_line_by_prompt_<cond>_pN.png   one panel per prompt (pair x order): every trace's stop
      point, chosen level (circle; filled if the answer is correct) and unchosen level (square),
      against the trace's thinking tokens; the line is the prompt's median chosen level
  stop_hist_by_prompt_<cond>_pN.png   one panel per prompt: histograms over its traces of the
      chosen level, the unchosen level and ln(p_chosen / p_unchosen) at the stop, all in nats on
      one axis (the ratio is the difference of the two levels); a narrow histogram is what a
      fixed stopping threshold on that quantity would give
  stop_line_by_prompt_<cond>.csv      per prompt: median and SD of both stop levels, slope and
      correlation of the chosen level with log thinking tokens, and the SD that is left around
      the prompt's own straight line in log tokens
  fig_stop_line_<cond>.png            all prompts in one figure (a: median and middle half per
      prompt, ordered by the unchosen level; b: stop levels against thinking length)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from paths import Paths  # noqa: E402
from analysis.plot_logp_paths import COL, GRID, INK, MUTED, style  # noqa: E402


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


REFS = {"raw": (None, "raw logit"), "mean": ("mean", "logit minus the vocabulary mean"), "other": ("lse_other", "logit minus the other candidates")}
REF = "raw"


def reference(A: dict[str, np.ndarray], at: np.ndarray) -> np.ndarray:
    key = REFS[REF][0]
    return A[key][at].astype(float) if key else np.zeros(len(at))


def levels(P: Paths, A: dict[str, np.ndarray], rows: np.ndarray, at: np.ndarray, letter_a: np.ndarray) -> pd.DataFrame:
    """Chosen / unchosen levels of traces `rows` at flat sentence indices `at`; letter_a: the chosen letter is A."""
    za, zb, oth = A["z_A"][at].astype(float), A["z_B"][at].astype(float), reference(A, at)
    la, lb = A["logp_A"][at].astype(float), A["logp_B"][at].astype(float)
    c, u = np.where(letter_a, za, zb), np.where(letter_a, zb, za)
    return pd.DataFrame({
        "item_id": P.T.item_id.values[rows], "source_id": P.T.source_id.values[rows], "tokens": P.T.think_tokens.values[rows].astype(float),
        "correct": P.T.correct.values[rows].astype(float),
        "p_chosen": np.exp(np.where(letter_a, la, lb)), "p_unchosen": np.exp(np.where(letter_a, lb, la)),
        "logit_chosen": c, "logit_unchosen": u, "rel_chosen": c - oth, "rel_unchosen": u - oth, "reference": oth, "X": c - u})


def spread(d: pd.DataFrame) -> dict:
    out = {}
    for col in ("p_chosen", "logit_chosen", "logit_unchosen", "rel_chosen", "rel_unchosen", "reference", "X"):
        g = d.groupby("item_id")[col]
        out[col] = {"median": float(d[col].median()), "p05": float(d[col].quantile(0.05)), "p95": float(d[col].quantile(0.95)),
                    "sd": float(d[col].std()), "sd_between_prompts": float(g.mean().std()),
                    "sd_within_prompt": float(np.sqrt((g.var() * (g.size() - 1)).sum() / (g.size() - 1).sum()))}
    out["corr_chosen_unchosen"] = {"logit": float(d.logit_chosen.corr(d.logit_unchosen)), "rel": float(d.rel_chosen.corr(d.rel_unchosen))}
    w = d[["rel_chosen", "rel_unchosen"]] - d.groupby("item_id")[["rel_chosen", "rel_unchosen"]].transform("mean")
    out["corr_chosen_unchosen"]["rel_within_prompt"] = float(w.rel_chosen.corr(w.rel_unchosen))
    return out


def length_slopes(d: pd.DataFrame) -> dict:
    """Slope on log thinking tokens within prompt, CI by bootstrap over questions."""
    lt = np.log(d.tokens.clip(lower=1))
    x = lt - lt.groupby(d.item_id).transform("mean")
    rng, out = np.random.default_rng(0), {}
    qs = d.source_id.values
    idx = {q: np.where(qs == q)[0] for q in np.unique(qs)}
    for col in ("rel_chosen", "rel_unchosen", "logit_chosen", "logit_unchosen", "X"):
        y = (d[col] - d.groupby("item_id")[col].transform("mean")).values
        xv = x.values
        boot = []
        for _ in range(1000):
            i = np.concatenate([idx[q] for q in rng.choice(list(idx), len(idx))])
            boot.append((xv[i] * y[i]).sum() / (xv[i] ** 2).sum())
        out[col] = {"per_log_unit": float((xv * y).sum() / (xv ** 2).sum()), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    return out


def rule(P: Paths, A: dict[str, np.ndarray], keep: np.ndarray) -> dict:
    k, tr = P.k, P.tr
    rows = np.where(P.isv & keep[tr] & P.ends_on_verdict[tr] & np.isfinite(A["z_A"]) & np.isfinite(A["z_B"]))[0]
    y = (k[rows] == P.lv[tr[rows]]).astype(float)
    is_a = P.V[rows] == 1
    za, zb = A["z_A"][rows].astype(float), A["z_B"][rows].astype(float)
    oth = A["mean"][rows].astype(float) if np.isfinite(A["mean"][rows]).any() else A["lse_other"][rows].astype(float)
    own, other_letter = np.where(is_a, za, zb), np.where(is_a, zb, za)
    item = pd.factorize(P.T.item_id)[0][tr[rows]]
    out = {"verdicts": int(len(y)), "stops": int(y.sum())}
    for name, Z in (("rel", np.c_[own - oth, other_letter - oth]), ("logit", np.c_[own, other_letter]),
                    ("logit_with_mean", np.c_[own, other_letter, oth]), ("difference_only", np.c_[own - other_letter])):
        b, se, ll = fe_logit(y, Z, item)
        out[name] = {"coef": [float(v) for v in b], "se": [float(v) for v in se], "loglik": ll}
    return out


def figure(stop: pd.DataFrame, out: Path, cond: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    g = stop.groupby("item_id")[["rel_chosen", "rel_unchosen"]].quantile([0.25, 0.5, 0.75]).unstack()
    g = g.sort_values(("rel_unchosen", 0.5))
    x = np.arange(len(g))
    for name, off in (("unchosen", 0.0), ("chosen", 0.0)):
        col = f"rel_{name}"
        ax.vlines(x + off, g[(col, 0.25)], g[(col, 0.75)], color=COL[name], lw=1.4, alpha=0.55)
        ax.plot(x + off, g[(col, 0.5)], "o" if name == "chosen" else "s", ms=3.5, color=COL[name], mec="white", mew=0.4, label=f"{name} option")
    ax.axhline(stop.rel_chosen.median(), color=COL["chosen"], lw=0.8, ls=":")
    ax.set_xlabel(f"{len(g)} prompts (pair x order), ordered by the unchosen level")
    ax.set_ylabel(f"{REFS[REF][1]}, at the stop")
    ax.set_title("a  Stop levels per prompt: median and middle half of its traces", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax = axes[1]
    b = pd.cut(stop.tokens, [0, 500, 700, 900, 1100, 1300, 1600, 2000, 2600, 3500, 100000])
    w = stop[["rel_chosen", "rel_unchosen"]] - stop.groupby("item_id")[["rel_chosen", "rel_unchosen"]].transform("mean") + stop[["rel_chosen", "rel_unchosen"]].mean()
    for name, ls in (("chosen", "-"), ("unchosen", "--")):
        m = w[f"rel_{name}"].groupby(b, observed=True).agg(["mean", "size"])
        ax.plot([iv.mid if iv.right < 1e5 else 4200 for iv in m.index], m["mean"], color=COL[name], ls=ls, lw=2, marker="o", ms=5, label=f"{name} option")
    ax.set_xscale("log")
    ax.set_xticks([400, 700, 1000, 2000, 4000], labels=["400", "700", "1,000", "2,000", "4,000"])
    ax.minorticks_off()
    ax.set_xlabel("thinking tokens of the trace")
    ax.set_ylabel("stop level, prompt differences removed")
    ax.set_title("b  Stop levels against thinking length", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    for ax in axes:
        style(ax)
    fig.suptitle(f"{cond}: where the chosen and the unchosen option are when the thinking ends", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f"fig_stop_line_{cond}_{REF}.png", dpi=150)
    plt.close(fig)


def by_prompt(stop: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item, g in stop.groupby("item_id"):
        lt = np.log(g.tokens.clip(lower=1))
        fit = np.polyfit(lt, g.rel_chosen, 1) if lt.std() > 0 else (np.nan, np.nan)
        rows.append({"item_id": item, "source_id": g.source_id.iloc[0], "traces": len(g), "accuracy": g.correct.mean(),
                     "tokens_median": g.tokens.median(), "chosen_median": g.rel_chosen.median(), "chosen_sd": g.rel_chosen.std(),
                     "chosen_p10": g.rel_chosen.quantile(0.1), "chosen_p90": g.rel_chosen.quantile(0.9),
                     "unchosen_median": g.rel_unchosen.median(), "unchosen_sd": g.rel_unchosen.std(),
                     "ln_ratio_median": g.X.median(), "ln_ratio_sd": g.X.std(),
                     "chosen_iqr": g.rel_chosen.quantile(0.75) - g.rel_chosen.quantile(0.25),
                     "unchosen_iqr": g.rel_unchosen.quantile(0.75) - g.rel_unchosen.quantile(0.25),
                     "ln_ratio_iqr": g.X.quantile(0.75) - g.X.quantile(0.25), "p_chosen_median": g.p_chosen.median(),
                     "chosen_slope_per_log_tokens": fit[0], "chosen_sd_around_own_line": float((g.rel_chosen - np.polyval(fit, lt)).std()),
                     "chosen_corr_log_tokens": float(np.corrcoef(lt, g.rel_chosen)[0, 1]) if lt.std() > 0 else np.nan,
                     "p_chosen_above_0.99": (g.p_chosen > 0.99).mean()})
    return pd.DataFrame(rows)


def pages(stop: pd.DataFrame, table: pd.DataFrame, out: Path, cond: str, per_page: int = 20) -> int:
    items = sorted(stop.item_id.unique())
    chunks = [items[i:i + per_page] for i in range(0, len(items), per_page)]
    info = table.set_index("item_id")
    hi = np.ceil(stop.rel_chosen.quantile(0.999))
    lo = min(0.0, np.floor(stop.rel_unchosen.quantile(0.002)) - 1)
    for pi, chunk in enumerate(chunks, 1):
        fig, axes = plt.subplots(4, 5, figsize=(16, 11), sharey=True)
        for ax, item in zip(axes.ravel(), chunk):
            g = stop[stop.item_id == item]
            ax.plot(g.tokens, g.rel_unchosen, "s", ms=2.6, color=COL["unchosen"], alpha=0.55, mew=0)
            ok = g.correct.values.astype(bool)
            ax.plot(g.tokens[ok], g.rel_chosen[ok], "o", ms=3.4, color=COL["chosen"], alpha=0.8, mew=0)
            ax.plot(g.tokens[~ok], g.rel_chosen[~ok], "o", ms=3.4, mfc="none", mec=COL["chosen"], mew=0.9, alpha=0.9)
            e = info.loc[item]
            ax.axhline(e.chosen_median, color=COL["chosen"], lw=0.9, ls=":")
            ax.set_xlim(0, g.tokens.quantile(0.98) * 1.05)
            ax.set_ylim(lo, hi)
            ax.set_title(f"{item.split(':', 1)[1]}   correct {e.accuracy:.2f}\nchosen {e.chosen_median:.1f} (SD {e.chosen_sd:.1f}), unchosen {e.unchosen_median:.1f} (SD {e.unchosen_sd:.1f})",
                         fontsize=7.5, color=INK, loc="left")
            style(ax)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel(f"stop level ({REFS[REF][1]})", fontsize=8)
        for ax in axes[-1, :]:
            ax.set_xlabel("thinking tokens of the trace", fontsize=8)
        handles = [plt.Line2D([], [], marker="o", ls="", color=COL["chosen"], label="chosen option, answer correct"),
                   plt.Line2D([], [], marker="o", ls="", mfc="none", mec=COL["chosen"], label="chosen option, answer wrong"),
                   plt.Line2D([], [], marker="s", ls="", color=COL["unchosen"], label="unchosen option"),
                   plt.Line2D([], [], color=COL["chosen"], lw=0.9, ls=":", label="median chosen level of the prompt")]
        fig.legend(handles=handles, loc="upper right", ncol=4, fontsize=8, frameon=False)
        fig.suptitle(f"{cond}: where each trace stops, per prompt (page {pi} of {len(chunks)})", x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"stop_line_by_prompt_{cond}_{REF}_p{pi}.png", dpi=110)
        plt.close(fig)
    return len(chunks)


def histograms(stop: pd.DataFrame, table: pd.DataFrame, out: Path, cond: str, per_page: int = 20) -> int:
    items = sorted(stop.item_id.unique())
    chunks = [items[i:i + per_page] for i in range(0, len(items), per_page)]
    info = table.set_index("item_id")
    lo, hi = min(-4.0, float(np.floor(stop.X.quantile(0.002)))), float(np.ceil(stop.rel_chosen.quantile(0.999)))
    edges = np.arange(lo - 0.125, hi + 0.5, 0.5)    # readouts come in steps of 0.25: two steps per bin
    for pi, chunk in enumerate(chunks, 1):
        fig, axes = plt.subplots(4, 5, figsize=(16, 11), sharex=True, sharey=True)
        for ax, item in zip(axes.ravel(), chunk):
            g = stop[stop.item_id == item]
            ax.hist(g.rel_unchosen.clip(lo, hi), bins=edges, color=COL["unchosen"], alpha=0.6)
            ax.hist(g.rel_chosen.clip(lo, hi), bins=edges, color=COL["chosen"], alpha=0.6)
            ax.hist(g.X.clip(lo, hi), bins=edges, histtype="step", color=INK, lw=1.1)
            e = info.loc[item]
            ax.set_title(f"{item.split(':', 1)[1]}   correct {e.accuracy:.2f}\nSD: chosen {e.chosen_sd:.1f}, unchosen {e.unchosen_sd:.1f}, ratio {e.ln_ratio_sd:.1f}",
                         fontsize=7.5, color=INK, loc="left")
            style(ax)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("traces", fontsize=8)
        for ax in axes[-1, :]:
            ax.set_xlabel("level at the stop (nats)", fontsize=8)
        handles = [plt.Rectangle((0, 0), 1, 1, color=COL["chosen"], alpha=0.6, label=f"chosen option: {REFS[REF][1]}"),
                   plt.Rectangle((0, 0), 1, 1, color=COL["unchosen"], alpha=0.6, label=f"unchosen option: {REFS[REF][1]}"),
                   plt.Line2D([], [], color=INK, lw=1.1, label="ln(p chosen / p unchosen) = chosen - unchosen;  2.3 = ratio 10, 6.9 = 1,000, 11.5 = 100,000")]
        fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=8, frameon=False)
        fig.suptitle(f"{cond}: stop values per prompt (page {pi} of {len(chunks)})", x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"stop_hist_by_prompt_{cond}_{REF}_p{pi}.png", dpi=110)
        plt.close(fig)
    return len(chunks)


PAPER_RC = {"font.family": "sans-serif", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5, "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
            "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300}


def paper_histograms(stop: pd.DataFrame, table: pd.DataFrame, out: Path, cond: str, per_page: int = 20) -> int:
    """The stop-value histograms per prompt in a plain figure style: PDF (vector) and PNG per page."""
    out.mkdir(parents=True, exist_ok=True)
    items = sorted(stop.item_id.unique())
    chunks = [items[i:i + per_page] for i in range(0, len(items), per_page)]
    info = table.set_index("item_id")
    lo, hi = min(-4.0, float(np.floor(stop.X.quantile(0.002)))), float(np.ceil(stop.rel_chosen.quantile(0.999)))
    edges = np.arange(lo - 0.125, hi + 0.5, 0.5)
    with plt.rc_context(PAPER_RC):
        for pi, chunk in enumerate(chunks, 1):
            fig, axes = plt.subplots(5, 4, figsize=(7.2, 9.0), sharex=True, sharey=True)
            for ax, item in zip(axes.ravel(), chunk):
                g = stop[stop.item_id == item]
                e = info.loc[item]
                ax.hist(g.rel_unchosen.clip(lo, hi), bins=edges, color=COL["unchosen"], alpha=0.75, linewidth=0)
                ax.hist(g.rel_chosen.clip(lo, hi), bins=edges, color=COL["chosen"], alpha=0.75, linewidth=0)
                ax.hist(g.X.clip(lo, hi), bins=edges, histtype="step", color=INK, linewidth=0.8)
                pair, order = item.split(":", 1)[1].rsplit(":", 1)
                ax.set_title(f"{pair} ({order})   acc. {e.accuracy:.2f}\nSD {e.chosen_sd:.1f} / {e.unchosen_sd:.1f} / {e.ln_ratio_sd:.1f}", loc="left", pad=2, fontsize=6.5)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction="out")
            for ax in axes.ravel()[len(chunk):]:
                ax.axis("off")
            for ax in axes[:, 0]:
                ax.set_ylabel("traces")
            for ax in axes[-1, :]:
                ax.set_xlabel("value at the stop (nats)")
            handles = [plt.Rectangle((0, 0), 1, 1, color=COL["chosen"], alpha=0.75, label="chosen option, raw logit"),
                       plt.Rectangle((0, 0), 1, 1, color=COL["unchosen"], alpha=0.75, label="unchosen option, raw logit"),
                       plt.Line2D([], [], color=INK, lw=0.8, label=r"$\ln\,[p(\mathrm{chosen})/p(\mathrm{unchosen})]$")]
            fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
            fig.text(0.99, 0.004, f"{cond} condition, page {pi} of {len(chunks)}; SD given as chosen / unchosen / log ratio", fontsize=6, color=MUTED, ha="right", va="bottom")
            fig.tight_layout(rect=(0, 0.012, 1, 0.975))
            for ext in ("pdf", "png"):
                fig.savefig(out / f"stop_hist_{cond}_{REF}_p{pi}.{ext}")
            plt.close(fig)
    return len(chunks)


def paper_prob_histograms(stop: pd.DataFrame, table: pd.DataFrame, out: Path, cond: str, per_page: int = 20) -> int:
    """Histograms of p(chosen) and p(unchosen) at the stop per prompt, same layout as paper_histograms."""
    out.mkdir(parents=True, exist_ok=True)
    items = sorted(stop.item_id.unique())
    chunks = [items[i:i + per_page] for i in range(0, len(items), per_page)]
    info = table.set_index("item_id")
    edges = np.linspace(0, 1, 51)
    with plt.rc_context(PAPER_RC):
        for pi, chunk in enumerate(chunks, 1):
            fig, axes = plt.subplots(5, 4, figsize=(7.2, 9.0), sharex=True, sharey=True)
            for ax, item in zip(axes.ravel(), chunk):
                g = stop[stop.item_id == item]
                e = info.loc[item]
                ax.hist(g.p_unchosen, bins=edges, color=COL["unchosen"], alpha=0.75, linewidth=0)
                ax.hist(g.p_chosen, bins=edges, color=COL["chosen"], alpha=0.75, linewidth=0)
                pair, order = item.split(":", 1)[1].rsplit(":", 1)
                ax.set_title(f"{pair} ({order})   acc. {e.accuracy:.2f}\nmedian {e.p_chosen_median:.3f}, below 0.9 in {(g.p_chosen < 0.9).mean():.0%}",
                             loc="left", pad=2, fontsize=6.5)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction="out")
                ax.set_xlim(0, 1)
            for ax in axes.ravel()[len(chunk):]:
                ax.axis("off")
            for ax in axes[:, 0]:
                ax.set_ylabel("traces")
            for ax in axes[-1, :]:
                ax.set_xlabel("probability at the stop")
            handles = [plt.Rectangle((0, 0), 1, 1, color=COL["chosen"], alpha=0.75, label="p(chosen option)"),
                       plt.Rectangle((0, 0), 1, 1, color=COL["unchosen"], alpha=0.75, label="p(unchosen option)")]
            fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
            fig.text(0.99, 0.004, f"{cond} condition, page {pi} of {len(chunks)}; bins of 0.02; the two probabilities sum to one", fontsize=6, color=MUTED, ha="right", va="bottom")
            fig.tight_layout(rect=(0, 0.012, 1, 0.975))
            for ext in ("pdf", "png"):
                fig.savefig(out / f"prob_hist_{cond}_p{pi}.{ext}")
            plt.close(fig)
    return len(chunks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cond", default="baseline")
    ap.add_argument("--ref", default="raw", choices=list(REFS), help="what to subtract from the letter logits")
    ap.add_argument("--paper-dir", type=Path, help="also write the histogram pages in a plain figure style (PDF and PNG) here")
    args = ap.parse_args()
    global REF
    REF = args.ref
    P = Paths(args.run)
    A = dict(np.load(args.run / "analysis" / "paths.npz"))
    if "z_A" not in A:
        raise SystemExit("no raw logits: run readout.py --logits, then paths.py")
    keep = (P.T.cond == args.cond).values & P.T.answer.notna().values & np.isfinite(A["z_A"][P.off])
    rows = np.where(keep)[0]
    chose_a = (P.T.answer.values[rows] == "A")
    stop = levels(P, A, rows, P.end[rows], chose_a)
    has = P.lv[rows] > 0
    before = levels(P, A, rows[has], P.off[rows[has]] + P.lv[rows[has]] - 1, chose_a[has])
    res = {"cond": args.cond, "reference": REF, "traces": int(len(rows)), "prompts": int(stop.item_id.nunique()),
           "at_stop": spread(stop), "before_final_verdict": spread(before),
           "p_chosen_at_stop": {"above_0.99": float((stop.p_chosen > 0.99).mean()), "above_0.999": float((stop.p_chosen > 0.999).mean()),
                                "below_0.9": float((stop.p_chosen < 0.9).mean())},
           "length_slopes_at_stop": length_slopes(stop), "stop_rule_at_verdicts": rule(P, A, keep)}
    out = args.run / "analysis"
    table = by_prompt(stop)
    table.round(3).to_csv(out / f"stop_line_by_prompt_{args.cond}_{REF}.csv", index=False)
    res["per_prompt"] = {
        "chosen_sd": {"median": float(table.chosen_sd.median()), "min": float(table.chosen_sd.min()), "max": float(table.chosen_sd.max()),
                      "below_1": float((table.chosen_sd < 1).mean()), "below_1.5": float((table.chosen_sd < 1.5).mean())},
        "chosen_median": {"min": float(table.chosen_median.min()), "p25": float(table.chosen_median.quantile(0.25)),
                          "median": float(table.chosen_median.median()), "p75": float(table.chosen_median.quantile(0.75)), "max": float(table.chosen_median.max())},
        "chosen_sd_below_unchosen_sd": float((table.chosen_sd < table.unchosen_sd).mean()),
        "chosen_sd_around_own_line": {"median": float(table.chosen_sd_around_own_line.median()), "below_1": float((table.chosen_sd_around_own_line < 1).mean())},
        "chosen_slope_per_log_tokens": {"median": float(table.chosen_slope_per_log_tokens.median()), "p25": float(table.chosen_slope_per_log_tokens.quantile(0.25)),
                                        "p75": float(table.chosen_slope_per_log_tokens.quantile(0.75))},
        "unchosen_sd_median": float(table.unchosen_sd.median()),
        "ln_ratio_sd_median": float(table.ln_ratio_sd.median()),
        "iqr_median": {"chosen": float(table.chosen_iqr.median()), "unchosen": float(table.unchosen_iqr.median()), "ln_ratio": float(table.ln_ratio_iqr.median())},
        "chosen_sd_below_ratio_sd": float((table.chosen_sd < table.ln_ratio_sd).mean()),
        "chosen_iqr_below_ratio_iqr": float((table.chosen_iqr < table.ln_ratio_iqr).mean()),
        "unchosen_sd_below_ratio_sd": float((table.unchosen_sd < table.ln_ratio_sd).mean()),
        "chosen_falls_with_length": float((table.chosen_corr_log_tokens < 0).mean()),
        "chosen_corr_log_tokens": {"median": float(table.chosen_corr_log_tokens.median()), "below_-0.2": float((table.chosen_corr_log_tokens < -0.2).mean()),
                                   "above_0.2": float((table.chosen_corr_log_tokens > 0.2).mean())}}
    (out / f"stop_line_{args.cond}_{REF}.json").write_text(json.dumps(res, indent=1) + "\n")
    figure(stop, out, args.cond)
    n_pages = pages(stop, table, out, args.cond)
    histograms(stop, table, out, args.cond)
    if args.paper_dir:
        n_paper = paper_histograms(stop, table, args.paper_dir, args.cond)
        paper_prob_histograms(stop, table, args.paper_dir, args.cond)
        print(f"paper-style histograms: {n_paper} pages -> {args.paper_dir}/stop_hist_{args.cond}_{REF}_p1..{n_paper}.pdf/.png and prob_hist_{args.cond}_p1..{n_paper}.pdf/.png")

    print(f"{res['traces']} {args.cond} traces with raw logits, {res['prompts']} prompts")
    pc = res["p_chosen_at_stop"]
    print(f"\np(chosen) at the stop: median {res['at_stop']['p_chosen']['median']:.4f}, 5th percentile {res['at_stop']['p_chosen']['p05']:.4f}; "
          f"above 0.99 in {pc['above_0.99']:.3f}, above 0.999 in {pc['above_0.999']:.3f}, below 0.9 in {pc['below_0.9']:.4f}")
    for name, key in (("at the stop", "at_stop"), ("before the final verdict sentence", "before_final_verdict")):
        print(f"\n{name}:  {'':<16}{'median':>8}{'5%':>8}{'95%':>8}{'SD':>7}{'between prompts':>17}{'within prompt':>15}")
        for col in ("rel_chosen", "rel_unchosen", "logit_chosen", "logit_unchosen", "reference", "X"):
            e = res[key][col]
            print(f"   {col:<28}{e['median']:>8.2f}{e['p05']:>8.2f}{e['p95']:>8.2f}{e['sd']:>7.2f}{e['sd_between_prompts']:>17.2f}{e['sd_within_prompt']:>15.2f}")
        c = res[key]["corr_chosen_unchosen"]
        print(f"   correlation chosen vs unchosen: raw logit {c['logit']:+.2f}, relative {c['rel']:+.2f}, relative within prompt {c['rel_within_prompt']:+.2f}")
    print("\nstop level per log unit of thinking length, within prompt:")
    for col, e in res["length_slopes_at_stop"].items():
        print(f"   {col:<16}{e['per_log_unit']:+.2f} [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]")
    r = res["stop_rule_at_verdicts"]
    print(f"\ndoes the thinking end after a verdict? {r['stops']} stops in {r['verdicts']} verdicts; coefficients per nat (se)")
    print(f"   relative to the vocabulary mean (or the other candidates in older readouts): verdict's letter {r['rel']['coef'][0]:+.3f} ({r['rel']['se'][0]:.3f}), other letter {r['rel']['coef'][1]:+.3f} ({r['rel']['se'][1]:.3f})")
    print(f"   raw logits:                   verdict's letter {r['logit']['coef'][0]:+.3f} ({r['logit']['se'][0]:.3f}), other letter {r['logit']['coef'][1]:+.3f} ({r['logit']['se'][1]:.3f})")
    e = r["logit_with_mean"]
    print(f"   raw logits + vocabulary mean: verdict's letter {e['coef'][0]:+.3f}, other letter {e['coef'][1]:+.3f}, vocabulary mean {e['coef'][2]:+.3f}")
    print(f"   log-likelihood: difference only {r['difference_only']['loglik']:.0f}, two letters (relative) {r['rel']['loglik']:.0f}, two letters + vocabulary mean {e['loglik']:.0f}")
    pp = res["per_prompt"]
    print(f"\nper prompt ({len(table)}): median chosen stop level from {pp['chosen_median']['min']:.1f} to {pp['chosen_median']['max']:.1f} "
          f"(quartiles {pp['chosen_median']['p25']:.1f} / {pp['chosen_median']['median']:.1f} / {pp['chosen_median']['p75']:.1f})")
    print(f"   SD of the chosen stop level inside a prompt: median {pp['chosen_sd']['median']:.2f} (from {pp['chosen_sd']['min']:.2f} to {pp['chosen_sd']['max']:.2f}); "
          f"below 1 in {pp['chosen_sd']['below_1']:.2f} of prompts, below 1.5 in {pp['chosen_sd']['below_1.5']:.2f}; smaller than the unchosen SD in {pp['chosen_sd_below_unchosen_sd']:.2f}")
    c = pp["chosen_corr_log_tokens"]
    print(f"   chosen stop level vs log thinking tokens inside a prompt: correlation median {c['median']:+.2f}; negative in {pp['chosen_falls_with_length']:.2f} of prompts, "
          f"below -0.2 in {c['below_-0.2']:.2f}, above +0.2 in {c['above_0.2']:.2f}")
    sl, ln = pp["chosen_slope_per_log_tokens"], pp["chosen_sd_around_own_line"]
    print(f"   slope per log unit: median {sl['median']:+.2f} (quartiles {sl['p25']:+.2f} / {sl['p75']:+.2f}); SD left around the prompt's own line: "
          f"median {ln['median']:.2f}, below 1 in {ln['below_1']:.2f} of prompts; unchosen SD median {pp['unchosen_sd_median']:.2f}")
    print(f"   width of the stop distribution inside a prompt (median over prompts): SD chosen {pp['chosen_sd']['median']:.2f}, unchosen {pp['unchosen_sd_median']:.2f}, "
          f"ln ratio {pp['ln_ratio_sd_median']:.2f}; IQR chosen {pp['iqr_median']['chosen']:.2f}, unchosen {pp['iqr_median']['unchosen']:.2f}, ln ratio {pp['iqr_median']['ln_ratio']:.2f}")
    print(f"   chosen narrower than the ratio in {pp['chosen_sd_below_ratio_sd']:.2f} of prompts by SD, {pp['chosen_iqr_below_ratio_iqr']:.2f} by IQR; "
          f"unchosen narrower than the ratio in {pp['unchosen_sd_below_ratio_sd']:.2f}")
    print(f"-> {out}/stop_line_{args.cond}_{REF}.json, stop_line_by_prompt_{args.cond}_{REF}.csv, stop_line_by_prompt_{args.cond}_{REF}_p1..{n_pages}.png, "
          f"stop_hist_by_prompt_{args.cond}_{REF}_p1..{n_pages}.png, fig_stop_line_{args.cond}_{REF}.png")


if __name__ == "__main__":
    main()
