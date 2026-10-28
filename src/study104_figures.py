"""Figures of the 104-question study (Qwen3-8B and Qwen3-32B, 60 thinking traces per question): one command per figure, the
numbers next to each figure as CSV, captions in figures/study104/README.md.

    python src/study104_figures.py accuracy     # fig1_accuracy.png: accuracy of each model over the 104 questions, without and with thinking
    python src/study104_figures.py change       # fig2_answer_change.png: questions whose answer reasoning changes or keeps, right or wrong after
    python src/study104_figures.py steps        # fig3_conclusion_step.png: the conclusion sentence's change against the changes of the sentences before it
    python src/study104_figures.py examples     # fig4_examples.png: three traces of one question per model, the conclusion sentence marked
    python src/study104_figures.py examples_written   # fig5_examples_written.png: the same traces ended by the two written sentences
    python src/study104_figures.py perquestion  # fig6_per_question_correlation.png: distribution of the length-change correlation over questions
    python src/study104_figures.py attention    # fig7_attention_answer.png: where the token that writes the answer attends: share of the tokens vs of its attention
    python src/study104_figures.py attention_value   # fig7_attention_answer_value.png: the same with every weight multiplied by |W_O v| of its token
    python src/study104_figures.py mechanism    # fig8_length_attention_jump.png: length -> the conclusion's attention share -> the change across it
    python src/study104_figures.py masking      # fig9_masking_conclusion.png: confidence with the conclusion hidden, never written, or random sentences hidden
    python src/study104_figures.py accuracy_conclusion     # fig10_conclusion_accuracy.png: accuracy with and without the conclusion sentence
    python src/study104_figures.py confidence_conclusion   # fig11_conclusion_confidence.png: confidence with and without the conclusion sentence

Questions: data/tasks/study104.csv. Traces: runs/medxpertqa/study104-<model>-<order> (generated for the 105-question draw; the
questions not in study104.csv are left out). First reactions (one greedy answer with thinking switched off): the 8B's from its
screening run (study104.csv), the 32B's from runs/medxpertqa/study104-nothink-qwen3-32b-<order>.
Style: figures.py STYLE (Arial 8 pt, no top/right spines, 300 dpi), greys; no figure titles.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import STYLE, WHITE, LIGHT, MID, DARK, BLACK  # noqa: E402
from generate import ROOT  # noqa: E402

OUT = ROOT / "figures/study104"
RUNS = ROOT / "runs/medxpertqa"
MODELS = [("qwen3-8b", "Qwen3-8B"), ("qwen3-32b", "Qwen3-32B")]
QUESTIONS = pd.read_csv(ROOT / "data/tasks/study104.csv")


def traces(model: str) -> pd.DataFrame:
    keep = set(QUESTIONS.item_id)
    rows = [{"item_id": r["item_id"], "answered": r["answer"] in ("A", "B"), "correct": bool(r["correct"])}
            for o in ("ab", "ba") for f in sorted((RUNS / f"study104-{model}-{o}").glob("traces-*.jsonl"))
            for line in f.open(encoding="utf-8") for r in [json.loads(line)] if r["item_id"] in keep]
    return pd.DataFrame(rows)


def first_reactions(model: str) -> pd.Series:
    if model == "qwen3-8b":
        return QUESTIONS.set_index("item_id").nothink_correct.astype(bool)
    rows = {r["item_id"]: bool(r["correct"]) for o in ("ab", "ba") for f in (RUNS / f"study104-nothink-{model}-{o}").glob("traces-*.jsonl")
            for line in f.open(encoding="utf-8") for r in [json.loads(line)]}
    return pd.Series(rows).reindex(QUESTIONS.item_id)


def boot_ci(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean over questions, resampling questions."""
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def accuracy() -> None:
    per = QUESTIONS[["item_id", "pair_id", "order", "set", "correct_text", "distractor_text"]].copy()
    summary = []
    for model, label in MODELS:
        t = traces(model)
        a = t[t.answered].groupby("item_id").correct.agg(["mean", "size"])
        per[f"acc_{model}"] = per.item_id.map(a["mean"])
        per[f"n_{model}"] = per.item_id.map(a["size"]).astype(int)
        per[f"nothink_{model}"] = per.item_id.map(first_reactions(model)).astype(bool)
        for cond, col in (("no thinking (one greedy answer)", f"nothink_{model}"), ("thinking (60 samples)", f"acc_{model}")):
            x = per[col].astype(float).to_numpy()
            lo, hi = boot_ci(x)
            summary.append({"model": label, "condition": cond, "accuracy": x.mean(), "ci_low": lo, "ci_high": hi, "questions": len(x),
                            "answers": int(len(x) if "no thinking" in cond else per[f"n_{model}"].sum())})
    summary = pd.DataFrame(summary)
    per = per.sort_values(["acc_qwen3-8b", "acc_qwen3-32b"], kind="stable").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT / "fig1_accuracy.csv", index=False)
    summary.to_csv(OUT / "fig1_accuracy_summary.csv", index=False)

    plt.rcParams.update(STYLE)
    fig, bx = plt.subplots(figsize=(2.6, 2.6))
    w = 0.36
    for j, (model, label) in enumerate(MODELS):
        for k, (cond, face) in enumerate((("no thinking (one greedy answer)", WHITE), ("thinking (60 samples)", MID))):
            r = summary[(summary.model == label) & (summary.condition == cond)].iloc[0]
            xpos = j + (k - 0.5) * w
            bx.bar(xpos, r.accuracy, width=w * 0.92, facecolor=face, edgecolor=BLACK, linewidth=0.6,
                   yerr=[[r.accuracy - r.ci_low], [r.ci_high - r.accuracy]],
                   error_kw={"elinewidth": 0.6, "capsize": 2, "capthick": 0.6, "ecolor": BLACK})
            bx.text(xpos, r.ci_high + 0.02, f"{r.accuracy:.2f}", ha="center", va="bottom", fontsize=6.5)
    bx.axhline(0.5, color=MID, linewidth=0.6, linestyle=(0, (3, 2)), zorder=0)
    bx.set_xticks([0, 1], [label for _, label in MODELS])
    bx.set_xlim(-0.65, 1.65)
    bx.set_ylim(0, 1.05)
    bx.set_ylabel("accuracy (mean over 104 questions)")
    from matplotlib.patches import Patch
    bx.legend(handles=[Patch(facecolor=WHITE, edgecolor=BLACK, linewidth=0.6, label="no thinking"),
                       Patch(facecolor=MID, edgecolor=BLACK, linewidth=0.6, label="thinking")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.0, columnspacing=0.8, handletextpad=0.3)
    fig.savefig(OUT / "fig1_accuracy.png", bbox_inches="tight")
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(f"-> {OUT / 'fig1_accuracy.png'}")


def change() -> None:
    """Questions whose answer reasoning changes and questions whose answer it keeps, split by whether the answer after reasoning is
    right. Before = the first reaction; after = the answer of the majority of the 60 traces (no question has a 30-30 tie)."""
    acc = pd.read_csv(OUT / "fig1_accuracy.csv")   # written by `accuracy`
    cells = [("changed", "wrong → right", False, True), ("changed", "right → wrong", True, False),
             ("unchanged", "right → right", True, True), ("unchanged", "wrong → wrong", False, False)]
    rows = []
    for model, label in MODELS:
        before, after = acc[f"nothink_{model}"].astype(bool), acc[f"acc_{model}"] > 0.5
        assert not (acc[f"acc_{model}"] == 0.5).any()
        for cat, name, b, a in cells:
            n = int(((before == b) & (after == a)).sum())
            rows.append({"model": label, "category": cat, "transition": name, "right_after": a, "questions": n, "share": n / len(acc)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "fig2_answer_change.csv", index=False)

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.6), sharey=True, gridspec_kw={"wspace": 0.15})
    pos = [0, 1, 2.6, 3.6]
    for ax, (model, label), letter in zip(axes, MODELS, "ab"):
        d = t[t.model == label].reset_index(drop=True)
        for x, r in zip(pos, d.itertuples()):
            ax.bar(x, r.questions, width=0.8, facecolor=MID if r.right_after else WHITE, edgecolor=BLACK, linewidth=0.6)
            ax.text(x, r.questions + 1.2, str(r.questions), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(pos, [r.transition.replace(" → ", "\n→ ") for r in d.itertuples()], fontsize=7)
        for x, cat in ((np.mean(pos[:2]), "changed"), (np.mean(pos[2:]), "unchanged")):
            ax.text(x, -0.2, cat, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        n_changed = int(d[d.category == "changed"].questions.sum())
        ax.text(0.5, 1.02, f"{label}: {n_changed} changed, {len(acc) - n_changed} unchanged", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5)
        ax.set_xlim(-0.7, 4.3)
        ax.text(-0.14 if letter == "a" else -0.06, 1.1, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    axes[0].set_ylabel("questions (of 104)")
    axes[0].set_ylim(0, 75)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(facecolor=MID, edgecolor=BLACK, linewidth=0.6, label="right after reasoning"),
                            Patch(facecolor=WHITE, edgecolor=BLACK, linewidth=0.6, label="wrong after reasoning")],
                   loc="upper left", handlelength=1.0, handletextpad=0.4)
    fig.savefig(OUT / "fig2_answer_change.png", bbox_inches="tight")
    print(t.to_string(index=False))
    print(f"-> {OUT / 'fig2_answer_change.png'}")


def sentence_steps(model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per trace, from the closed readouts (readouts-*.jsonl, own close) and the conclusion onsets (conclusion-*.jsonl): the
    change of the confidence in the trace's final answer, log p(final letter) - log p(other letter), across every sentence before
    the conclusion (the first from the empty thinking block, readouts_start.jsonl) and across the conclusion (onset -> answer
    position). Returns (one row per earlier sentence, one row per trace)."""
    keep = set(QUESTIONS.item_id)
    steps, rows = [], []
    for o in ("ab", "ba"):
        rd = RUNS / f"study104-{model}-{o}" / "readouts"
        start = {r["item_id"]: r for line in (rd / "readouts_start.jsonl").open(encoding="utf-8") for r in [json.loads(line)]}
        concl = {r["sample_id"]: r for f in rd.glob("conclusion-*.jsonl") for line in f.open(encoding="utf-8") for r in [json.loads(line)]}
        for f in sorted(rd.glob("readouts-*.jsonl")):
            for line in f.open(encoding="utf-8"):
                d = json.loads(line)
                if d["item_id"] not in keep or d["sample_id"] not in concl:
                    continue
                c = concl[d["sample_id"]]
                s = 1.0 if c["letter"] == "A" else -1.0
                x = s * (np.array(d["closed"]["logp_A"]) - np.array(d["closed"]["logp_B"]))
                x0 = s * (start[d["item_id"]]["logp_A"] - start[d["item_id"]]["logp_B"])
                i_on = d["pos"].index(c["onset"])
                dx = np.diff(np.concatenate([[x0], x[: i_on + 1]]))
                jump = x[d["kind"].index("a")] - x[i_on]
                n = len(dx)
                steps += [{"sample_id": d["sample_id"], "k_from_end": k - n, "step": v} for k, v in enumerate(dx)]
                rows.append({"sample_id": d["sample_id"], "item_id": d["item_id"], "n_before": n, "jump": jump,
                             "percentile": float((np.abs(dx) < jump).mean()) if n else np.nan, "largest": bool(n and jump > np.abs(dx).max())})
    return pd.DataFrame(steps), pd.DataFrame(rows)


def steps() -> None:
    K = 15   # sentences shown before the conclusion
    looks = {"qwen3-8b": dict(marker="o", face=BLACK, edge=BLACK, line=BLACK, dx=-0.15), "qwen3-32b": dict(marker="s", face=WHITE, edge=MID, line=MID, dx=0.15)}
    prof, summ, hists = [], [], {}
    for model, label in MODELS:
        s, r = sentence_steps(model)
        g = s[s.k_from_end >= -K].groupby("k_from_end").step
        p = pd.DataFrame({"median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75), "q10": g.quantile(0.1), "q90": g.quantile(0.9),
                          "share_positive": g.apply(lambda v: (v > 0).mean()), "n": g.size()})
        p.loc[0] = [r.jump.median(), r.jump.quantile(0.25), r.jump.quantile(0.75), r.jump.quantile(0.1), r.jump.quantile(0.9), (r.jump > 0).mean(), len(r)]
        prof.append(p.assign(model=label).rename_axis("sentence_from_conclusion").reset_index())
        summ.append({"model": label, "traces": len(r), "earlier_sentences": len(s), "earlier_abs_change_median": s.step.abs().median(),
                     "earlier_change_share_positive": (s.step > 0).mean(), "conclusion_change_median": r.jump.median(),
                     "conclusion_change_share_positive": (r.jump > 0).mean(), "percentile_median": r.percentile.median(),
                     "share_percentile_ge_0.9": (r.percentile >= 0.9).mean(), "share_largest_change": r.largest.mean()})
        hists[label] = r.percentile.dropna().to_numpy()
    prof, summ = pd.concat(prof), pd.DataFrame(summ)
    prof.to_csv(OUT / "fig3_conclusion_step.csv", index=False)
    summ.to_csv(OUT / "fig3_conclusion_step_summary.csv", index=False)

    plt.rcParams.update(STYLE)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [2.1, 1], "wspace": 0.32})
    ax.axhline(0, color=MID, linewidth=0.6, zorder=0)
    for model, label in MODELS:
        lk, p = looks[model], prof[prof.model == label].set_index("sentence_from_conclusion")
        xs = p.index.to_numpy() + lk["dx"]
        ax.vlines(xs, p.q25, p.q75, color=lk["line"], linewidth=0.8, zorder=1)
        ax.scatter(xs, p["median"], s=14, marker=lk["marker"], facecolor=lk["face"], edgecolor=lk["edge"], linewidth=0.7, zorder=2, label=label)
    ax.set_xticks(list(range(-K, 0, 3)) + [0], [str(k) for k in range(-K, 0, 3)] + ["conclusion"])
    ax.set_xlim(-K - 0.8, 0.8)
    ax.set_xlabel("sentence, counted back from the conclusion sentence")
    ax.set_ylabel("change in confidence in the final answer\nacross the sentence (nats)")
    ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)

    bins = np.linspace(0, 1, 21)
    for model, label in MODELS:
        h = hists[label]
        w = np.full(len(h), 1 / len(h))
        bx.hist(h, bins=bins, weights=w, histtype="step", color=looks[model]["line"], linewidth=0.9,
                linestyle="-" if model == "qwen3-8b" else (0, (3, 1.5)), label=label)
    bx.set_xlim(0, 1)
    bx.set_xlabel("percentile of the conclusion's change\namong the trace's earlier changes")
    bx.set_ylabel("share of traces")
    bx.legend(loc="upper left", handletextpad=0.4, borderaxespad=0.2)
    for a_, letter, xo in ((ax, "a", -0.13), (bx, "b", -0.3)):
        a_.text(xo, 1.04, letter, transform=a_.transAxes, fontsize=9, fontweight="bold", va="bottom")
    fig.savefig(OUT / "fig3_conclusion_step.png", bbox_inches="tight")
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string(index=False))
    print(f"-> {OUT / 'fig3_conclusion_step.png'}")


def conclusion_jumps(model: str) -> pd.DataFrame:
    """Per trace: thinking length and the change across the conclusion sentence (conclusion-*.jsonl: own minus none), toward the
    trace's final answer."""
    keep = set(QUESTIONS.item_id)
    rows = []
    for o in ("ab", "ba"):
        run = RUNS / f"study104-{model}-{o}"
        tokens = {r["sample_id"]: r["think_tokens"] for f in run.glob("traces-*.jsonl") for line in f.open(encoding="utf-8") for r in [json.loads(line)]}
        for f in (run / "readouts").glob("conclusion-*.jsonl"):
            for line in f.open(encoding="utf-8"):
                c = json.loads(line)
                if c["item_id"] in keep:
                    s = 1.0 if c["letter"] == "A" else -1.0
                    rows.append({"item_id": c["item_id"], "sample_id": c["sample_id"], "tokens": tokens[c["sample_id"]],
                                 "jump": s * ((c["own"]["logp_A"] - c["own"]["logp_B"]) - (c["none"]["logp_A"] - c["none"]["logp_B"]))})
    return pd.DataFrame(rows)


def perquestion() -> None:
    """Within every question, the Spearman correlation between thinking length and the conclusion's change: each question with a
    95% interval (resampling its traces), against the range expected if length and change were unrelated (shuffling)."""
    from scipy.stats import rankdata
    rng = np.random.default_rng(0)
    B = 1000

    def rho(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Spearman correlation of x with each row of y (ties averaged)."""
        rx = rankdata(x); ry = np.apply_along_axis(rankdata, -1, y)
        rx, ry = rx - rx.mean(), ry - ry.mean(-1, keepdims=True)
        return (ry @ rx) / np.sqrt((ry ** 2).sum(-1) * (rx ** 2).sum())

    rows, nulls = [], {}
    for model, label in MODELS:
        t = conclusion_jumps(model)
        allnull = []
        for iid, g in t.groupby("item_id"):
            x, y = g.tokens.to_numpy(float), g.jump.to_numpy(float)
            n = len(g)
            r = float(rho(x, y[None])[0])
            idx = rng.integers(0, n, (B, n))
            boot = np.array([rho(x[i], y[i][None])[0] for i in idx])
            null = rho(x, np.array([rng.permutation(y) for _ in range(B)]))
            allnull.append(null)
            rows.append({"model": label, "item_id": iid, "traces": n, "rho": r, "ci_low": np.nanquantile(boot, 0.025), "ci_high": np.nanquantile(boot, 0.975),
                         "null_low": np.quantile(null, 0.025), "null_high": np.quantile(null, 0.975), "boot_sd": np.nanstd(boot)})
        nulls[label] = np.concatenate(allnull)
    q = pd.DataFrame(rows)
    summ = []
    for label, d in q.groupby("model", sort=False):
        noise = np.sqrt((d.boot_sd ** 2).mean())
        summ.append({"model": label, "questions": len(d), "rho_median": d.rho.median(), "negative": int((d.rho < 0).sum()),
                     "ci_below_0": int((d.ci_high < 0).sum()), "ci_above_0": int((d.ci_low > 0).sum()),
                     "sd_across_questions": d.rho.std(), "sd_sampling": noise, "sd_beyond_sampling": float(np.sqrt(max(d.rho.var() - noise ** 2, 0)))})
    summ = pd.DataFrame(summ)
    q.to_csv(OUT / "fig6_per_question_correlation.csv", index=False)
    summ.to_csv(OUT / "fig6_per_question_correlation_summary.csv", index=False)

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True, gridspec_kw={"wspace": 0.12})
    bins = np.linspace(-1, 1, 41)
    for ax, (model, label), letter in zip(axes, MODELS, "ab"):
        d = q[q.model == label]
        sm = summ[summ.model == label].iloc[0]
        ax.hist(d.rho, bins=bins, weights=np.full(len(d), 100 / len(d)), color=MID, edgecolor=WHITE, linewidth=0.3)
        ax.axvline(0, color=BLACK, linewidth=0.6)
        ax.set_xlim(-0.8, 0.8)
        ax.set_title(f"{label}: negative in {sm.negative} of {sm.questions} questions", fontsize=7.5)
        ax.set_xlabel("within-question Spearman correlation between\nthinking length and the conclusion's change")
        ax.text(-0.13 if letter == "a" else -0.04, 1.08, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    axes[0].set_ylabel("percentage of questions")
    fig.savefig(OUT / "fig6_per_question_correlation.png", bbox_inches="tight")
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string(index=False))
    print(f"-> {OUT / 'fig6_per_question_correlation.png'}")


EXAMPLE_QUESTION = "medxpertqa:Text-699:I-B:ab"   # left unilateral sacral extension (A, correct) vs left unilateral sacral flexion
EXAMPLES = {"qwen3-8b": (19, 9, 22), "qwen3-32b": (13, 0, 7)}   # sample indices of the short / medium / long trace
# Chosen for illustration (2026-10-26, on the first 30 traces of each question): of the 35 questions in which both models have a short, a medium and a long correct
# letter-verdict trace whose change across the conclusion falls with length for the own conclusion and for both written
# sentences, the one with the largest changes and the lowest readouts before the conclusion (the example question of the
# 12-question study). Within it, one trace from each third of the lengths, spaced about evenly on a log scale.
WRITTEN = {"therefore_1": "Therefore, the answer is {L}.", "between_1": "Therefore, between A and B, the answer is {L}."}


def example_traces(model: str) -> list[dict]:
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    tok = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218", local_files_only=True))
    order = EXAMPLE_QUESTION.rsplit(":", 1)[1]
    rd = RUNS / f"study104-{model}-{order}"
    want = {f"{EXAMPLE_QUESTION}#{k}": name for k, name in zip(EXAMPLES[model], ("short", "medium", "long"))}
    tr = {r["sample_id"]: r for f in rd.glob("traces-*.jsonl") for line in f.open(encoding="utf-8") for r in [json.loads(line)] if r["sample_id"] in want}
    ro = {r["sample_id"]: r for f in (rd / "readouts").glob("readouts-*.jsonl") for line in f.open(encoding="utf-8") for r in [json.loads(line)] if r["sample_id"] in want}
    co = {r["sample_id"]: r for f in (rd / "readouts").glob("conclusion-*.jsonl") for line in f.open(encoding="utf-8") for r in [json.loads(line)] if r["sample_id"] in want}
    st = next(r for line in (rd / "readouts/readouts_start.jsonl").open(encoding="utf-8") for r in [json.loads(line)] if r["item_id"] == EXAMPLE_QUESTION)
    out = []
    for sid, name in want.items():
        r, d, c = tr[sid], ro[sid], co[sid]
        s = 1.0 if c["letter"] == "A" else -1.0
        start, onset = d["think_start"], c["onset"]
        xs = [0] + [p - start for p, k in zip(d["pos"], d["kind"]) if k in ("s", "e")]
        ys = [s * (st["logp_A"] - st["logp_B"])] + [s * (a - b) for a, b, k in zip(d["closed"]["logp_A"], d["closed"]["logp_B"], d["kind"]) if k in ("s", "e")]
        val = {k: s * (c[k]["logp_A"] - c[k]["logp_B"]) for k in ("none", "own", *WRITTEN)}
        out.append({"model": model, "trace": name, "sample_id": sid, "letter": c["letter"], "tokens": d["think_end"] - start,
                    "sentences": sum(k == "s" for k in d["kind"]) + 1, "x": xs, "y": ys, "x_onset": onset - start, **val,
                    "conclusion": tok.decode(r["token_ids"][onset: d["think_end"]]).strip(),
                    "n_written": {k: len(tok(" " + v.format(L=c["letter"]), add_special_tokens=False)["input_ids"]) for k, v in WRITTEN.items()}})
    return out


def examples(written: bool = False) -> None:
    """The closed readout along three traces of one question (short, medium, long) in each model, the conclusion sentence marked.
    written=True: the same traces up to the onset of the conclusion, ended instead by the two written sentences."""
    import textwrap
    from matplotlib.lines import Line2D
    data = {m: example_traces(m) for m, _ in MODELS}
    lo = min(min(min(e["y"]), e["none"]) for es in data.values() for e in es)
    hi = max(max(max(e["y"]), *(e[k] for k in WRITTEN), e["own"]) for es in data.values() for e in es)
    rows = []
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.6), sharey=True, gridspec_kw={"wspace": 0.12, "hspace": 0.28})
    for j, (model, label) in enumerate(MODELS):
        xmax = max(e["tokens"] for e in data[model])
        for i, e in enumerate(data[model]):
            ax = axes[i, j]
            keep = [k for k, x in enumerate(e["x"]) if (x <= e["x_onset"] if written else True)]
            x, y = [e["x"][k] for k in keep], [e["y"][k] for k in keep]
            ax.axhline(0, color=MID, linewidth=0.6, linestyle=(0, (3, 2)), zorder=0)
            ax.plot(x, y, color=BLACK, linewidth=0.8, marker="o", markersize=1.8, zorder=2)
            ax.plot(e["x_onset"], e["none"], marker="o", markersize=4.5, markerfacecolor=WHITE, markeredgecolor=BLACK, markeredgewidth=0.8, zorder=4, linestyle="none")
            if not written:
                ax.plot(x[-1], e["own"], marker="o", markersize=4.5, color=BLACK, zorder=3, linestyle="none")
                right = x[-1] < 0.5 * xmax   # a short trace: the quote goes to the right of its end and the bracket to the left
                xa = e["x_onset"] - 0.03 * xmax if right else x[-1] + 0.03 * xmax
                ax.vlines(xa, e["none"], e["own"], color=BLACK, linewidth=0.7)
                ax.hlines([e["none"], e["own"]], xa - 0.008 * xmax, xa + 0.008 * xmax, color=BLACK, linewidth=0.7)
                ax.text(xa + (-0.018 if right else 0.018) * xmax, (e["none"] + e["own"]) / 2, f"+{e['own'] - e['none']:.1f}", fontsize=7, va="center",
                        ha="right" if right else "left")
                quote = "\n".join(textwrap.wrap(f'"{e["conclusion"]}"', 40))
                ax.annotate(quote, xy=(x[-1] + (0.005 if right else -0.005) * xmax, (e["none"] + e["own"]) / 2),
                            xytext=(x[-1] + (0.12 if right else -0.03) * xmax, e["own"] + 3.5 if right else e["own"] + 9), fontsize=6.3,
                            ha="left" if right else "right", va="top",
                            arrowprops={"arrowstyle": "-|>", "lw": 0.6, "color": BLACK, "shrinkB": 2})
                rows += [{"model": label, "trace": e["trace"], "sample_id": e["sample_id"], "tokens_kept": xx, "readout": yy, "point": "sentence"} for xx, yy in zip(x, y)]
            else:
                # the two written endpoints drawn apart along x (both sit at the onset plus a dozen tokens) so that they can be told apart;
                # the "Therefore" change written above its marker, the "between" change below its marker
                for key, face, marker, dx, dy, va in (("therefore_1", BLACK, "o", 0.04, 1.0, "bottom"), ("between_1", MID, "s", 0.13, -1.0, "top")):
                    xe = e["x_onset"] + dx * xmax
                    ax.plot([e["x_onset"], xe], [e["none"], e[key]], color=face, linewidth=0.8, zorder=2)
                    ax.plot(xe, e[key], marker=marker, markersize=4.5, color=face, zorder=3, linestyle="none")
                    ax.text(xe, e[key] + dy, f"+{e[key] - e['none']:.1f}", fontsize=7, va=va, ha="center", color=BLACK if key == "therefore_1" else DARK)
                    rows.append({"model": label, "trace": e["trace"], "sample_id": e["sample_id"], "tokens_kept": e["x_onset"] + e["n_written"][key],
                                 "readout": e[key], "point": key})
                rows += [{"model": label, "trace": e["trace"], "sample_id": e["sample_id"], "tokens_kept": xx, "readout": yy, "point": "sentence"} for xx, yy in zip(x, y)]
            ax.text(0.02, 0.96, f"{e['trace']}: {e['tokens']:,} tokens, {e['sentences']} sentences", transform=ax.transAxes, fontsize=7, va="top", ha="left")
            ax.set_xlim(-0.03 * xmax, 1.17 * xmax)
            ax.set_ylim(lo - 1.5, hi + (9 if not written else 6))
            if j == 0:
                ax.set_ylabel(f"log p({e['letter']}) − log p({'B' if e['letter'] == 'A' else 'A'})")
            if i == 0:
                ax.set_title(label, fontsize=8)
        axes[2, j].set_xlabel("thinking tokens kept before the cut")
    if written:
        handles = [Line2D([], [], marker="o", markersize=4.5, markerfacecolor=WHITE, markeredgecolor=BLACK, color="none", label="before the conclusion sentence"),
                   Line2D([], [], marker="o", markersize=4.5, color=BLACK, linestyle="none", label='after "Therefore, the answer is A."'),
                   Line2D([], [], marker="s", markersize=4.5, color=MID, linestyle="none", label='after "Therefore, between A and B, the answer is A."')]
    else:
        handles = [Line2D([], [], marker="o", markersize=4.5, markerfacecolor=WHITE, markeredgecolor=BLACK, color="none", label="before the conclusion sentence"),
                   Line2D([], [], marker="o", markersize=4.5, color=BLACK, linestyle="none", label="after it (the model's own conclusion)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02 if not written else -0.04), ncol=len(handles), handletextpad=0.3, columnspacing=1.2)
    stem = "fig5_examples_written" if written else "fig4_examples"
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    pd.DataFrame(rows).to_csv(OUT / f"{stem}.csv", index=False)
    for es in data.values():
        for e in es:
            print(f"{e['model']:10s} {e['trace']:6s} {e['sample_id']}: {e['tokens']} tokens, {e['sentences']} sentences, before {e['none']:.2f}, own +{e['own'] - e['none']:.2f}, "
                  f"therefore +{e['therefore_1'] - e['none']:.2f}, between +{e['between_1'] - e['none']:.2f} | {e['conclusion'][:70]!r}")
    print(f"-> {OUT / stem}.png")


# the four parts of the context, in reading order: label, share columns, token-count columns of data/screen/study104_attention_answer.csv
GROUPS = [("question", ["stem", "option_chosen", "option_other", "instruction"], ["n_stem", "n_option_chosen", "n_option_other", "n_instruction"]),
          ("reasoning before the conclusion", ["thinking"], ["n_thinking"]), ("final conclusion", ["conclusion"], ["n_conclusion"]),
          ("</think> and chat template", ["close", "template", "think_open"], ["n_close", "n_template", "n_think_open"])]


def attention(value: bool = False) -> None:
    """Where the token that writes the answer letter attends (attention_at_answer.py run104 / analyze104), with thinking: the share
    of the context's tokens that each part holds against the share of that token's attention it receives (the first token left out,
    the rest renormalised; mean over layers and heads). Shares are averaged over a question's traces, then over the 104 questions.
    value=True: every attention weight multiplied by the norm of what its token writes into the head's output, |W_O v| (the v_
    columns; Kobayashi et al. 2020)."""
    stem = "fig7_attention_answer_value" if value else "fig7_attention_answer"
    pre = "v_" if value else ""
    a = pd.read_csv(ROOT / "data/screen/study104_attention_answer.csv")
    a = a[a.item_id.isin(QUESTIONS.item_id)]
    cols = {}
    for label, share_cols, n_cols in GROUPS:
        cols[f"attention: {label}"] = a[[pre + c for c in share_cols]].sum(axis=1)
        cols[f"tokens: {label}"] = a[n_cols].sum(axis=1) / (a.n_keys - 1)
    d = pd.concat([a[["model", "mode", "item_id"]], pd.DataFrame(cols)], axis=1)
    for kind in ("attention", "tokens"):                       # each trace sums to 1 (float16 rounding leaves up to 0.5% off)
        c = [f"{kind}: {g[0]}" for g in GROUPS]
        d[c] = d[c].div(d[c].sum(axis=1), axis=0)
    q = d.groupby(["model", "mode", "item_id"]).mean().reset_index()
    q.to_csv(OUT / f"{stem}.csv", index=False)
    m = q.drop(columns="item_id").groupby(["model", "mode"]).mean()

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    faces = [LIGHT, MID, BLACK, WHITE]
    bars = [(model, kind, y) for (model, _), ys in zip(MODELS, ((3.3, 2.5), (1.1, 0.3))) for kind, y in zip(("tokens", "attention"), ys)]
    for model, kind, y in bars:
        v = [m.loc[(model, "think"), f"{kind}: {g[0]}"] for g in GROUPS]
        left = 0.0
        for k, (x, face) in enumerate(zip(v, faces)):
            ax.barh(y, x, left=left, height=0.62, facecolor=face, edgecolor=BLACK, linewidth=0.5)
            if x >= 0.045:
                ax.text(left + x / 2, y, f"{x:.0%}", ha="center", va="center", fontsize=7, color=WHITE if face in (MID, BLACK) else BLACK)
            if k == 2:                                         # the final conclusion: always labelled, outside when its slice is thin
                if x < 0.045:
                    ax.annotate(f"{x:.1%}", xy=(left + x / 2, y + 0.31), xytext=(left + x / 2 + 0.035, y + 0.5), fontsize=7, ha="left", va="center",
                                arrowprops=dict(arrowstyle="-", lw=0.5, color=BLACK))
            left += x
    ax.set_yticks([y for _, _, y in bars], ["share of the tokens", "share of the answer's attention" + (",\nweighted by |W$_O$v|" if value else "")] * 2)
    ax.tick_params(axis="y", length=0)
    for (_, label), y in zip(MODELS, (2.9, 0.7)):
        ax.text(-0.38, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.25, 3.95)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share (first token left out)")
    ax.spines["left"].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=f, edgecolor=BLACK, linewidth=0.5, label=g[0]) for f, g in zip(faces, GROUPS)], loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    print(m.round(4).T.to_string())
    print(f"-> {OUT / stem}.png")

def mechanism() -> None:
    """Longer thinking gives the final conclusion a smaller share of the answer's attention, and the conclusion raises the confidence
    in proportion to that share. Within every question its traces are split into fifths, (a) of thinking length and (b) of the
    conclusion's attention share (fig7: the first token left out, mean over layers and heads); per fifth the question's median, then
    the median and quartiles over the 104 questions. The change across the conclusion is that of fig6 (conclusion_jumps). Also the
    within-question regression of the change on log2 length, alone and with the share (both demeaned within question; intervals
    from resampling questions)."""
    a = pd.read_csv(ROOT / "data/screen/study104_attention_answer.csv", usecols=["model", "mode", "sample_id", "item_id", "conclusion"])
    a = a[(a["mode"] == "think") & a.item_id.isin(QUESTIONS.item_id)]
    rng = np.random.default_rng(0)
    rows, stats, curves = [], [], {}
    for model, label in MODELS:
        g = a[a.model == model].merge(conclusion_jumps(model), on=["sample_id", "item_id"])
        fifth = lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1                      # noqa: E731
        g["length_fifth"] = g.groupby("item_id").tokens.transform(fifth)
        g["share_fifth"] = g.groupby("item_id").conclusion.transform(fifth)
        for panel, by, y in (("a", "length_fifth", "conclusion"), ("b", "share_fifth", "jump")):
            per_q = g.groupby(["item_id", by])[y].median().unstack()
            c = pd.DataFrame({"median": per_q.median(), "q25": per_q.quantile(0.25), "q75": per_q.quantile(0.75), "questions": per_q.notna().sum()})
            curves[(label, panel)] = c
            rows.append(c.rename_axis("fifth").reset_index().assign(model=label, panel=panel, y=y))
        sp = lambda h, x, y: h[x].corr(h[y], method="spearman")                                   # noqa: E731
        r_len = g.groupby("item_id").apply(lambda h: sp(h, "tokens", "conclusion"), include_groups=False)
        r_jump = g.groupby("item_id").apply(lambda h: sp(h, "conclusion", "jump"), include_groups=False)
        z = pd.DataFrame({"loglen": np.log2(g.tokens), "share": 100 * g.conclusion, "jump": g.jump, "item_id": g.item_id})
        z[["loglen", "share", "jump"]] -= z.groupby("item_id")[["loglen", "share", "jump"]].transform("mean")
        items = z.item_id.unique()
        by_item = {i: h for i, h in z.groupby("item_id")}

        def fit(d: pd.DataFrame) -> tuple[float, float, float]:
            b1 = np.linalg.lstsq(d[["loglen"]].values, d.jump.values, rcond=None)[0][0]
            b2 = np.linalg.lstsq(d[["loglen", "share"]].values, d.jump.values, rcond=None)[0]
            return b1, b2[0], b2[1]
        est = fit(z)
        boot = np.array([fit(pd.concat([by_item[i] for i in rng.choice(items, len(items))])) for _ in range(1000)])
        lo, hi = np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)
        stats.append({"model": label, "traces": len(g), "rho_length_share_median": r_len.median(), "length_share_negative": int((r_len < 0).sum()),
                      "rho_share_jump_median": r_jump.median(), "share_jump_positive": int((r_jump > 0).sum()), "questions": len(r_len),
                      "length_alone_nats_per_doubling": est[0], "length_alone_ci": f"[{lo[0]:.2f}, {hi[0]:.2f}]",
                      "length_with_share_nats_per_doubling": est[1], "length_with_share_ci": f"[{lo[1]:.2f}, {hi[1]:.2f}]",
                      "share_nats_per_point": est[2], "share_ci": f"[{lo[2]:.2f}, {hi[2]:.2f}]"})
    pd.concat(rows).to_csv(OUT / "fig8_length_attention_jump.csv", index=False)
    stats = pd.DataFrame(stats)
    stats.to_csv(OUT / "fig8_length_attention_jump_stats.csv", index=False)

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.5), gridspec_kw={"wspace": 0.42})
    looks = {"Qwen3-8B": dict(marker="o", face=BLACK, edge=BLACK, line=BLACK, dx=-0.07), "Qwen3-32B": dict(marker="s", face=WHITE, edge=MID, line=MID, dx=0.07)}
    texts = {"a": ("thinking length\n(fifths within each question)", ["1\nshortest", "2", "3", "4", "5\nlongest"],
                   "conclusion's share of\nthe answer's attention", "rho_length_share_median", "length_share_negative", "negative"),
             "b": ("conclusion's attention share\n(fifths within each question)", ["1\nsmallest", "2", "3", "4", "5\nlargest"],
                   "change across the\nconclusion (nats)", "rho_share_jump_median", "share_jump_positive", "positive")}
    for ax, panel in zip(axes, "ab"):
        xlabel, ticks, ylabel, rcol, ncol, sign = texts[panel]
        for k, (label, lk) in enumerate(looks.items()):
            c = curves[(label, panel)]
            x = c.index.to_numpy() + lk["dx"]
            ax.vlines(x, c.q25, c.q75, color=lk["line"], lw=0.8)
            ax.plot(x, c["median"], color=lk["line"], lw=1.0, marker=lk["marker"], markersize=4, markerfacecolor=lk["face"], markeredgecolor=lk["edge"])
            s = stats[stats.model == label].iloc[0]
            rho_txt = f"{s[rcol]:+.2f}".replace("-", "−")
            ax.text(0.04, (0.15 - 0.1 * k) if panel == "a" else (0.97 - 0.1 * k), f"{label}: ρ = {rho_txt}, {sign} in {s[ncol]}/{s.questions}",
                    transform=ax.transAxes, ha="left", va="bottom" if panel == "a" else "top", fontsize=6.8, color=lk["line"])
        ax.set_xticks(range(1, 6), ticks)
        ax.set_xlim(0.6, 5.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(-0.3, 1.04, panel, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylim(0.04, 0.135)
    axes[1].set_ylim(0, 7.4)
    fig.savefig(OUT / "fig8_length_attention_jump.png", bbox_inches="tight")
    for (label, panel), c in curves.items():
        print(label, panel, c["median"].round(3).tolist())
    print(stats.round(3).to_string(index=False))
    print(f"-> {OUT / 'fig8_length_attention_jump.png'}")


def masking_traces(model: str) -> pd.DataFrame:
    """Per trace (ablate_conclusion.py analyze104 and the conclusion readout): confidence = log p(final letter) - log p(other letter)
    at the answer-writing position, the final letter being the trace's answer, with the full reasoning and with the conclusion
    sentence or random earlier sentences hidden from </think> and the answer-writing position (one masking pass), and with the
    thinking closed before the conclusion sentence or after it (conclusion-*.jsonl "none" / "own", one readout pass)."""
    m = pd.read_csv(ROOT / "data/screen/study104_mask_traces.csv")
    g = m[(m["mode"] == "think") & (m.model == model) & m.item_id.isin(QUESTIONS.item_id)]
    t = g.drop_duplicates("sample_id").set_index("sample_id")[["item_id", "answer", "y_base"]].rename(columns={"y_base": "full"})
    for v in ("conclusion", "rand_for_conclusion"):
        t[v] = g[g.variant == v].set_index("sample_id").y
    read = {}
    for o in ("ab", "ba"):
        for f in (RUNS / f"study104-{model}-{o}" / "readouts").glob("conclusion-*.jsonl"):
            for line in f.open(encoding="utf-8"):
                c = json.loads(line)
                read[c["sample_id"]] = (c["own"]["logp_A"] - c["own"]["logp_B"], c["none"]["logp_A"] - c["none"]["logp_B"])
    s = np.where(t.answer == "A", 1.0, -1.0)
    t["own"] = np.array([read.get(i, (np.nan, np.nan))[0] for i in t.index]) * s
    t["none"] = np.array([read.get(i, (np.nan, np.nan))[1] for i in t.index]) * s
    return t


def masking() -> None:
    """Hiding the conclusion sentence from the answer gives the confidence of a trace whose conclusion was never written; hiding
    random earlier sentences holding at least as many tokens leaves it unchanged (masking_traces). Per question the median over its
    traces, then the median and quartiles over the 104 questions."""
    conds = [("full reasoning", "full"), ("random earlier sentences hidden\n(at least as many tokens)", "rand_for_conclusion"),
             ("conclusion sentence hidden", "conclusion"), ("reasoning ended before\nthe conclusion sentence", "none")]
    per_q, summ = [], []
    for model, label in MODELS:
        t = masking_traces(model)
        q = t.groupby("item_id")[[c for _, c in conds]].median()
        per_q.append(q.reset_index().assign(model=label))
        for name, c in conds:
            flips = float(((t[c] > 0) != (t.full > 0))[t[c].notna()].mean())
            summ.append({"model": label, "condition": name.replace("\n", " "), "median": q[c].median(), "q25": q[c].quantile(0.25),
                         "q75": q[c].quantile(0.75), "answers_flipped": flips, "traces": int(t[c].notna().sum())})
        d = q.conclusion - q.none
        summ.append({"model": label, "condition": "conclusion hidden minus ended before it (per question)", "median": d.median(),
                     "q25": d.quantile(0.25), "q75": d.quantile(0.75), "answers_flipped": np.nan, "traces": int((d.abs() < 1).sum())})
    pd.concat(per_q).to_csv(OUT / "fig9_masking_conclusion.csv", index=False)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "fig9_masking_conclusion_summary.csv", index=False)

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.1), sharey=True, sharex=True, gridspec_kw={"wspace": 0.08})
    for ax, (model, label), letter in zip(axes, MODELS, "ab"):
        s = summ[summ.model == label].set_index("condition")
        for k, (name, _) in enumerate(conds):
            r = s.loc[name.replace("\n", " ")]
            y = len(conds) - 1 - k
            ax.hlines(y, r.q25, r.q75, color=BLACK, lw=0.9)
            ax.plot(r["median"], y, "o", color=BLACK, markersize=4.5)
            ax.text(r.q75 + 0.5, y, f"{r['median']:.1f}", va="center", ha="left", fontsize=7)
        ax.axhline(1.5, color=LIGHT, lw=0.6)
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("confidence in the final answer (nats)")
        ax.tick_params(axis="y", length=0)
        ax.text(-0.73 if letter == "a" else -0.03, 1.08, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    axes[0].set_yticks(range(len(conds)), [n for n, _ in conds][::-1])
    axes[0].set_ylim(-0.6, len(conds) - 0.4)
    axes[0].set_xlim(0, 20)
    fig.savefig(OUT / "fig9_masking_conclusion.png", bbox_inches="tight")
    print(summ.round(3).to_string(index=False))
    print(f"-> {OUT / 'fig9_masking_conclusion.png'}")


def conclusion_bars(kind: str) -> None:
    """Accuracy (kind="accuracy") or confidence (kind="confidence") with the conclusion sentence and with it hidden from the answer
    (masking_traces). Accuracy: the share of a question's traces whose answer is correct, without the conclusion the letter the
    masked readout prefers; mean over the 104 questions, 95% intervals from resampling questions. Confidence: log p(final letter) -
    log p(other letter), the median over a question's traces, then the median and quartiles over the questions."""
    m = pd.read_csv(ROOT / "data/screen/study104_mask_traces.csv")
    rows, per = [], []
    for model, label in MODELS:
        t = masking_traces(model)
        t["correct"] = m[(m.model == model) & (m["mode"] == "think")].drop_duplicates("sample_id").set_index("sample_id").correct.reindex(t.index).astype(bool)
        if kind == "accuracy":
            t["with"] = t.correct.astype(float)
            t["without"] = np.where(t.conclusion > 0, t.correct, ~t.correct).astype(float)
            q = t.groupby("item_id")[["with", "without"]].mean()
            for c in ("with", "without"):
                lo, hi = boot_ci(q[c].to_numpy())
                rows.append({"model": label, "condition": c, "value": q[c].mean(), "low": lo, "high": hi})
            lo, hi = boot_ci((q.without - q["with"]).to_numpy())
            rows.append({"model": label, "condition": "without minus with", "value": (q.without - q["with"]).mean(), "low": lo, "high": hi})
        else:
            q = t.groupby("item_id")[["full", "conclusion"]].median().rename(columns={"full": "with", "conclusion": "without"})
            for c in ("with", "without"):
                rows.append({"model": label, "condition": c, "value": q[c].median(), "low": q[c].quantile(0.25), "high": q[c].quantile(0.75)})
            d = q.without - q["with"]
            rows.append({"model": label, "condition": "without minus with", "value": d.median(), "low": d.quantile(0.25), "high": d.quantile(0.75)})
        per.append(q.reset_index().assign(model=label))
    stem = "fig10_conclusion_accuracy" if kind == "accuracy" else "fig11_conclusion_confidence"
    pd.concat(per).to_csv(OUT / f"{stem}.csv", index=False)
    s = pd.DataFrame(rows)
    s.to_csv(OUT / f"{stem}_summary.csv", index=False)

    plt.rcParams.update(STYLE)
    fig, bx = plt.subplots(figsize=(2.6, 2.6))
    w = 0.36
    top = 1.05 if kind == "accuracy" else 19
    for j, (model, label) in enumerate(MODELS):
        for k, (cond, face) in enumerate((("with", MID), ("without", WHITE))):
            r = s[(s.model == label) & (s.condition == cond)].iloc[0]
            xpos = j + (k - 0.5) * w
            bx.bar(xpos, r.value, width=w * 0.92, facecolor=face, edgecolor=BLACK, linewidth=0.6, yerr=[[r.value - r.low], [r.high - r.value]],
                   error_kw={"elinewidth": 0.6, "capsize": 2, "capthick": 0.6, "ecolor": BLACK})
            bx.text(xpos, r.high + 0.02 * top, f"{r.value:.2f}" if kind == "accuracy" else f"{r.value:.1f}", ha="center", va="bottom", fontsize=6.5)
    if kind == "accuracy":
        bx.axhline(0.5, color=MID, linewidth=0.6, linestyle=(0, (3, 2)), zorder=0)
        bx.set_ylabel("accuracy (mean over 104 questions)")
    else:
        bx.set_ylabel("confidence in the final answer\n(nats, median over 104 questions)")
        bx.set_yticks(range(0, 20, 5))
    bx.set_xticks([0, 1], [label for _, label in MODELS])
    bx.set_xlim(-0.65, 1.65)
    bx.set_ylim(0, top)
    from matplotlib.patches import Patch
    bx.legend(handles=[Patch(facecolor=MID, edgecolor=BLACK, linewidth=0.6, label="with the conclusion sentence"),
                       Patch(facecolor=WHITE, edgecolor=BLACK, linewidth=0.6, label="conclusion sentence hidden")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, handlelength=1.0, handletextpad=0.3)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    print(s.round(4).to_string(index=False))
    print(f"-> {OUT / stem}.png")

if __name__ == "__main__":
    {"accuracy": accuracy, "change": change, "steps": steps, "examples": examples,
     "examples_written": lambda: examples(written=True), "perquestion": perquestion, "attention": attention,
     "attention_value": lambda: attention(value=True), "mechanism": mechanism,
     "masking": masking, "accuracy_conclusion": lambda: conclusion_bars("accuracy"),
     "confidence_conclusion": lambda: conclusion_bars("confidence")}[sys.argv[1]]()
