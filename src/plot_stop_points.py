"""Plot every trace's stop point (thinking length, evidence at the stop) with the fitted boundaries.

    python src/plot_stop_points.py --run runs/medxpertqa/main-qwen3-8b

Evidence is signed toward the correct option, so traces that answer correctly stop on the upper
side and wrong ones on the lower side. For each condition and outcome a boundary curve
H(t) = floor + drop * exp(-t / tau) is fitted to |X| with prompt effects removed (each trace's
height minus its prompt-x-condition-x-outcome mean plus the group mean), next to the flat line a
fixed boundary would give. Writes <run>/analysis/fig_stop_points.png and stop_points_fit.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent))
from stop_height import ORDER, load  # noqa: E402


def curve(t, floor, drop, tau):
    return floor + drop * np.exp(-t / tau)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    df = load(args.run)
    df["cell3"] = df.item_id + "|" + df.cond + "|" + df.correct.astype(int).astype(str)
    df["grp"] = df.cond + "|" + df.correct.astype(int).astype(str)
    df["H_adj"] = df.H - df.groupby("cell3").H.transform("mean") + df.groupby("grp").H.transform("mean")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    col = {1.0: "#2a9d8f", 0.0: "#e76f51"}
    cond_col = {"speed": "#e76f51", "baseline": "#264653", "careful": "#2a9d8f"}
    edges = np.array([0.15, 0.5, 0.65, 0.8, 0.95, 1.1, 1.3, 1.5, 1.8, 2.2, 2.7, 3.4, 6.0])
    fits = {}
    tt = np.linspace(0.2, 5.0, 200)
    for ax, cond in zip(axes.flat[:3], ORDER):
        d = df[df.cond == cond]
        s = d.sample(min(6000, len(d)), random_state=1)
        ax.scatter(s.t, np.where(s.correct == 1, s.H, -s.H), s=4, alpha=0.10, c=s.correct.map(col), linewidths=0, rasterized=True)
        for outcome, sign, name in ((1.0, 1, "correct"), (0.0, -1, "wrong")):
            g = d[d.correct == outcome]
            p, _ = curve_fit(curve, g.t.values, g.H_adj.values, p0=[7.0, 8.0, 0.4], bounds=([0, 0, 0.05], [30, 60, 5]))
            resid_curve = float(((g.H_adj - curve(g.t.values, *p)) ** 2).sum())
            resid_flat = float(((g.H_adj - g.H_adj.mean()) ** 2).sum())
            fits[f"{cond}|{name}"] = {"n": int(len(g)), "floor": float(p[0]), "drop": float(p[1]), "tau_k_tokens": float(p[2]),
                                      "height_at_400": float(curve(0.4, *p)), "height_at_1000": float(curve(1.0, *p)), "height_at_3000": float(curve(3.0, *p)),
                                      "flat_level": float(g.H_adj.mean()), "variance_explained_vs_flat": 1 - resid_curve / resid_flat}
            b = np.digitize(g.t.values, edges)
            xs = [g.t.values[b == i].mean() for i in range(1, len(edges)) if (b == i).sum() >= 30]
            ys = [g.H_adj.values[b == i].mean() for i in range(1, len(edges)) if (b == i).sum() >= 30]
            es = [g.H_adj.values[b == i].std() / np.sqrt((b == i).sum()) * 1.96 for i in range(1, len(edges)) if (b == i).sum() >= 30]
            ax.errorbar(xs, sign * np.array(ys), yerr=es, fmt="o", ms=6, color="black", mfc=col[outcome], capsize=2, zorder=5,
                        label=f"{name}: binned mean (prompt effects removed)")
            ax.plot(tt, sign * curve(tt, *p), "-", color=col[outcome], lw=2.4, zorder=4, label=f"{name}: fitted boundary")
            ax.plot(tt, sign * np.full_like(tt, g.H_adj.mean()), "--", color=col[outcome], lw=1.2, zorder=3, label=f"{name}: flat line")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlim(0, 5)
        ax.set_ylim(-26, 26)
        ax.set_title(f"{cond}: each dot is one trace's stop point (6,000 of {len(d):,} shown)", fontsize=10)
        ax.set_xlabel("thinking tokens (thousands)")
        ax.set_ylabel("evidence at the stop, toward the correct option (nats)")
        if cond == "speed":
            ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax = axes.flat[3]
    for cond in ORDER:
        for name, ls in (("correct", "-"), ("wrong", "--")):
            f = fits[f"{cond}|{name}"]
            sign = 1 if name == "correct" else -1
            ax.plot(tt, sign * curve(tt, f["floor"], f["drop"], f["tau_k_tokens"]), ls, color=cond_col[cond], lw=2.2,
                    label=f"{cond}, {name}")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlim(0, 5)
    ax.set_ylim(-16, 16)
    ax.set_title("fitted boundaries of the three conditions", fontsize=10)
    ax.set_xlabel("thinking tokens (thousands)")
    ax.set_ylabel("evidence at the stop, toward the correct option (nats)")
    ax.legend(fontsize=8, ncol=3, loc="center right")
    fig.tight_layout()
    out = args.run / "analysis"
    fig.savefig(out / "fig_stop_points.png", dpi=140)
    (out / "stop_points_fit.json").write_text(json.dumps(fits, indent=1) + "\n")
    print(f"{'group':<18}{'n':>7}{'floor':>7}{'drop':>7}{'tau':>7}{'H@400':>8}{'H@1000':>8}{'H@3000':>8}{'flat':>7}{'R2 vs flat':>11}")
    for k, f in fits.items():
        print(f"{k:<18}{f['n']:>7}{f['floor']:>7.2f}{f['drop']:>7.2f}{f['tau_k_tokens']:>7.2f}{f['height_at_400']:>8.2f}{f['height_at_1000']:>8.2f}{f['height_at_3000']:>8.2f}{f['flat_level']:>7.2f}{f['variance_explained_vs_flat']:>11.3f}")
    print(f"-> {out}/fig_stop_points.png, stop_points_fit.json")


if __name__ == "__main__":
    main()
