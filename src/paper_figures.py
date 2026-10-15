"""Paper figures and tables of the "confidence inflation in LLMs" study, one per finding -> figures/confidence_inflation_in_llms/.

    python src/paper_figures.py positions        # table1_positions.csv: where every readout is taken, with the tokens of one example trace
    python src/paper_figures.py examples         # fig0_example_trajectories.png: the closed readout along three traces of one question
    python src/paper_figures.py jump             # fig_jump_vs_length.png: the jump across the conclusion sentence against length, natural and written sentence
    python src/paper_figures.py jumphist         # fig_jump_length_hist.png: histogram over questions of the correlation between length and the jump
    python src/paper_figures.py fig1             # fig1_answer_attention.png: what the answer-writing position attends to
    python src/paper_figures.py fig2             # fig2_attention_vs_length.png: how that attention spreads as the thinking gets longer
    python src/paper_figures.py fig3             # fig3_jump_vs_attention.png: the conclusion's jump follows the attention it receives
    python src/paper_figures.py fig4             # fig4_attention_within_trace.png: where the running state looks as the trace unfolds
    python src/paper_figures.py ablation_table   # table2_ablation.csv: the masked-attention variants, with the example trace
    python src/paper_figures.py fig5             # fig5_ablation_conclusion.png: hiding the conclusion removes its jump; read directly
    python src/paper_figures.py fig6             # fig6_ablation_layers.png: which layers carry the conclusion to the answer
    python src/paper_figures.py fig8             # fig8_ablation_thinking.png: hiding the earlier thinking raises the lean (dilution and content)
    python src/paper_figures.py fig9             # fig9_effect_vs_length.png: the conclusion's causal effect falls with length and follows its attention share

Style: figures.py STYLE (Arial 8 pt, no top/right spines, 300 dpi), greys; no figure titles, captions live in the folder's README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import STYLE, LIGHT, MID, DARK, BLACK  # noqa: E402
from generate import ROOT  # noqa: E402

OUT = ROOT / "figures/confidence_inflation_in_llms"
SCREEN = ROOT / "data/screen"
RUNS = {o: ROOT / f"runs/medxpertqa/main-qwen3-8b-{o}" for o in ("ab", "ba")}
EXAMPLE = "medxpertqa:Text-699:I-B:ab#40"
PARTS = [("sink", "sink (first token)"), ("format", "format tokens"), ("stem", "question stem"), ("options", "option lines"),
         ("thinking", "own reasoning"), ("conclusion", "conclusion sentence")]
# format tokens = chat template, the instruction, "<think>\n", "</think>" and the answer-writing position itself


def attention_meta():
    for run in RUNS.values():
        for mp in sorted((run / "readouts").glob("attention-*.jsonl")):
            for line in mp.open(encoding="utf-8"):
                yield json.loads(line)


def token_counts() -> pd.DataFrame:
    """Tokens in every part of the context, per trace (from the attention shards' meta lines)."""
    rows = []
    for m in attention_meta():
        seg = m["segments"]
        onset = m["sentence_starts"][m["onset_row"]]
        n_stem = seg["stem"][1] - seg["stem"][0]
        n_opt = (seg["option_A"][1] - seg["option_A"][0]) + (seg["option_B"][1] - seg["option_B"][0])
        # format = every prompt token that is not the sink, the stem or an option line (the instruction included), plus "<think>\n",
        # "</think>" and the answer-writing position
        rows.append({"sample_id": m["sample_id"], "n_sink": 1,
                     "n_format": m["n_prompt"] - 1 - n_stem - n_opt + m["think_start"] + (m["answer_pos"] - m["think_end"]),
                     "n_stem": n_stem, "n_options": n_opt, "n_thinking": onset - m["think_start"], "n_conclusion": m["think_end"] - onset, "n_total": m["n_total"]})
    return pd.DataFrame(rows)


def answer_masses() -> pd.DataFrame:
    a = pd.read_csv(SCREEN / "medxpertqa_attention_answer.csv")
    a = a[a.row == "a"].copy()
    a["mass_options"] = a.mass_option_A + a.mass_option_B
    a["mass_format"] = a.mass_template + a.mass_instruction + a.mass_think_open + a.mass_close
    return a.merge(token_counts(), on="sample_id")


# ----------------------------------------------------------------------------------------------------------------- table 1

def positions_table() -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from attention_at_answer import prompt_segments
    from generate import MODEL, MODEL_REVISION, TASKS, iter_traces, load_tasks

    tok = AutoTokenizer.from_pretrained(snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True))
    items = {it["item_id"]: it for it in load_tasks(TASKS)}
    m = next(x for x in attention_meta() if x["sample_id"] == EXAMPLE)
    order = EXAMPLE.rsplit("#", 1)[0].rsplit(":", 1)[1]
    tr = next(r for r in iter_traces(RUNS[order], ("sample_id", "token_ids", "answer")) if r["sample_id"] == EXAMPLE)
    p_ids, _ = prompt_segments(tok, items[m["item_id"]])
    ids = p_ids + tr["token_ids"][: m["answer_pos"] + 1]
    P, starts = m["n_prompt"], m["sentence_starts"]
    seg = m["segments"]

    def tail(pos: int, k: int = 5) -> str:
        return tok.decode(ids[max(0, pos - k): pos]) + "[[" + tok.decode([ids[pos]]) + "]]"

    rows = [
        {"name": "sink", "what": "the first token of the sequence (<|im_start|>); most heads put a large weight here regardless of content", "position_rule": "token 0",
         "visible_context": "nothing before it", "next_token": "", "example_position": 0, "example_tokens": tail(0), "used_in": "attention: reported separately, never blocked"},
        {"name": "stem", "what": "the case text of the question", "position_rule": f"prompt tokens {seg['stem'][0]}-{seg['stem'][1] - 1}", "visible_context": "", "next_token": "",
         "example_position": seg["stem"][0], "example_tokens": tok.decode(ids[seg["stem"][0]: seg["stem"][0] + 8]) + " ...", "used_in": "attention segment"},
        {"name": "option lines", "what": "the two option lines", "position_rule": f"prompt tokens {seg['option_A'][0]}-{seg['option_B'][1] - 1}", "visible_context": "", "next_token": "",
         "example_position": seg["option_A"][0], "example_tokens": tok.decode(ids[seg["option_A"][0]: seg["option_B"][1]]).replace("\n", " | "), "used_in": "attention segment"},
        {"name": "instruction", "what": "\"Answer with only A or B. No other words.\" and the template tokens after it", "position_rule": f"prompt tokens {seg['instruction'][0]}-{P - 1}",
         "visible_context": "", "next_token": "", "example_position": seg["instruction"][0], "example_tokens": tok.decode(ids[seg["instruction"][0]: P]).replace("\n", "\\n"), "used_in": "attention segment"},
        {"name": "0 (empty thinking)", "what": "the state before the first thinking sentence", "position_rule": "the newline after <think>", "visible_context": "prompt + <think>\\n",
         "next_token": "the first word of the thinking", "example_position": m["rows"][0], "example_tokens": tail(m["rows"][0]), "used_in": "truncation readouts (t = 0), attention row 0"},
        {"name": "s_k (sentence k done)", "what": "the state after sentence k is complete, before sentence k + 1 starts", "position_rule": "the last token of sentence k (its punctuation or trailing whitespace)",
         "visible_context": "prompt + sentences 1..k", "next_token": "the first word of sentence k + 1", "example_position": m["rows"][1], "example_tokens": tail(m["rows"][1]) + f"  (k = 1 of {m['n_sent']})",
         "used_in": "all truncation readouts (closed / forced / between / based / stop), stop hazard, attention rows s"},
        {"name": "onset of the conclusion", "what": "the state before the conclusion unit (the verdict sentence, or the last two when the last is an option-name fragment)",
         "position_rule": "the s_k row of the sentence before the conclusion unit", "visible_context": "prompt + thinking without its conclusion", "next_token": "the first word of the conclusion",
         "example_position": m["rows"][m["onset_row"]], "example_tokens": tail(m["rows"][m["onset_row"]]), "used_in": "jump across the conclusion (before), probes' final point, letter-sentence and ablation experiments"},
        {"name": "e (stop)", "what": "the last thinking token; its output is </think>", "position_rule": "the token before </think>", "visible_context": "prompt + whole thinking",
         "next_token": "</think>", "example_position": m["rows"][-2], "example_tokens": tail(m["rows"][-2]), "used_in": "stop hazard P(</think>), attention row e"},
        {"name": "a (answer)", "what": "the token whose output is the answer letter", "position_rule": "the newline after </think>", "visible_context": "prompt + whole thinking + </think>\\n\\n",
         "next_token": f"the letter ({tr['answer']} here)", "example_position": m["rows"][-1], "example_tokens": tail(m["rows"][-1]), "used_in": "answer readout (gap, confidence), jump across the conclusion (after), attention row a, ablation"},
    ]
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "table1_positions.csv", index=False)
    print(f"example {EXAMPLE}: prompt {P} tokens, thinking {m['think_end'] - m['think_start']} tokens in {m['n_sent']} sentences, sequence {len(ids)} tokens")
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(t[["name", "position_rule", "example_position", "example_tokens", "next_token"]].to_string(index=False))
    print(f"-> {OUT / 'table1_positions.csv'}")


# ------------------------------------------------------------------------------------------------------------------- fig 1

def fig1() -> None:
    a = answer_masses()
    names = [p for p, _ in PARTS]
    share = pd.DataFrame({p: a[f"mass_{p}"] for p in names})
    per_token = pd.DataFrame({p: a[f"mass_{p}"] / a[f"n_{p}"].clip(lower=1) for p in names})
    counts = pd.DataFrame({p: a[f"n_{p}"] for p in names})
    summary = pd.DataFrame({"part": [lab for _, lab in PARTS], "tokens_median": counts.median().values,
                            "share_median": share.median().values, "share_q25": share.quantile(0.25).values, "share_q75": share.quantile(0.75).values,
                            "share_per_token_median": per_token.median().values, "share_per_token_q25": per_token.quantile(0.25).values, "share_per_token_q75": per_token.quantile(0.75).values})
    summary.to_csv(OUT / "fig1_answer_attention.csv", index=False)
    tk = pd.read_csv(SCREEN / "medxpertqa_tokens_conclusion.csv")
    tk = tk[tk.row == "a"]
    TYPES = [("letter (own)", "answer letter"), ('"answer"', '"answer"'), ("other word", "other words"), ("option-name word", "option names"), ("punctuation / newline", "punctuation")]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.25, 0.9, 1.0]})
        y = np.arange(len(PARTS))[::-1]
        for ax, df, xlabel, log in ((axes[0], share, "attention share of the answer-writing position", False), (axes[1], per_token, "share per token", True)):
            med, lo, hi = df.median().values, df.quantile(0.25).values, df.quantile(0.75).values
            ax.barh(y, med, color=LIGHT, edgecolor=BLACK, linewidth=0.6, height=0.65)
            ax.errorbar(med, y, xerr=[med - lo, hi - med], fmt="none", ecolor=BLACK, elinewidth=0.6, capsize=2)
            ax.set_yticks(y)
            ax.set_yticklabels([lab for _, lab in PARTS])
            ax.set_xlabel(xlabel)
            if log:
                ax.set_xscale("log")
                ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax = axes[2]
        y3 = np.arange(len(TYPES))[::-1]
        for yi, (col, lab) in zip(y3, TYPES):
            v = tk[f"{col}_all"]
            ax.barh(yi, v.median(), color=LIGHT, edgecolor=BLACK, linewidth=0.6, height=0.65)
            ax.errorbar(v.median(), yi, xerr=[[v.median() - v.quantile(0.25)], [v.quantile(0.75) - v.median()]], fmt="none", ecolor=BLACK, elinewidth=0.6, capsize=2)
        ax.set_yticks(y3); ax.set_yticklabels([lab for _, lab in TYPES])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("share within the\nconclusion sentence")
        ax.set_xticks([0, 0.01, 0.02]); ax.set_xticklabels(["0", "0.01", "0.02"])
        for ax, letter, x in zip(axes, "abc", (-0.75, -0.05, -0.6)):
            ax.text(x, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
        fig.tight_layout(w_pad=2.0)
        fig.savefig(OUT / "fig1_answer_attention.png")
        plt.close(fig)
    tk_summary = pd.DataFrame({"token_type": [lab for _, lab in TYPES], "share_median": [tk[f"{c}_all"].median() for c, _ in TYPES],
                               "share_q25": [tk[f"{c}_all"].quantile(0.25) for c, _ in TYPES], "share_q75": [tk[f"{c}_all"].quantile(0.75) for c, _ in TYPES],
                               "share_late_layers_median": [tk[f"{c}_late"].median() for c, _ in TYPES]})
    tk_summary.to_csv(OUT / "fig1_answer_attention_conclusion_tokens.csv", index=False)
    print(tk_summary.round(4).to_string(index=False))
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print(f"-> {OUT / 'fig1_answer_attention.png'}, {OUT / 'fig1_answer_attention.csv'}")


# ------------------------------------------------------------------------------------------------------------------- fig 2

FIG2_PARTS = [("sink", "attention sink", BLACK, "-", "o"), ("format", "format tokens", MID, "--", "s"), ("thinking", "own reasoning", DARK, "-", "^"),
              ("stem", "question stem", BLACK, ":", "D"), ("conclusion", "conclusion sentence", BLACK, "-.", "v")]


def wrho(d: pd.DataFrame, x: str, y: str) -> float:
    from scipy.stats import spearmanr
    return float(np.nanmedian([spearmanr(g[x], g[y]).correlation for _, g in d.groupby("item_id") if len(g) > 10]))


def fig2() -> None:
    a = answer_masses()
    a["len_bin"] = pd.qcut(a.rel, 8, labels=False)
    a["ctx_bin"] = pd.qcut(a.n_keys, 10, labels=False)
    rows = []
    for part, lab, *_ in FIG2_PARTS:
        g = a.groupby("len_bin")
        for b, h in g:
            rows.append({"panel": "a", "series": lab, "bin": b, "x_median": h.rel.median(), "x_tokens_median": h.think_tokens.median(),
                         "median": h[f"mass_{part}"].median(), "q25": h[f"mass_{part}"].quantile(0.25), "q75": h[f"mass_{part}"].quantile(0.75), "n": len(h)})
    for col, lab in (("eff_tokens_mean", "effective attended tokens"), ("ent_norm_mean", "normalised entropy")):
        for b, h in a.groupby("ctx_bin"):
            rows.append({"panel": "b" if col.startswith("eff") else "c", "series": lab, "bin": b, "x_median": h.n_keys.median(), "x_tokens_median": h.think_tokens.median(),
                         "median": h[col].median(), "q25": h[col].quantile(0.25), "q75": h[col].quantile(0.75), "n": len(h)})
    stats = [{"stat": f"within-prompt Spearman rho of {lab} with thinking tokens (median over 24 prompts)", "value": wrho(a, "think_tokens", f"mass_{part}")} for part, lab, *_ in FIG2_PARTS]
    stats += [{"stat": "within-prompt Spearman rho of the effective attended tokens with context length", "value": wrho(a, "n_keys", "eff_tokens_mean")},
              {"stat": "within-prompt Spearman rho of the normalised entropy with context length", "value": wrho(a, "n_keys", "ent_norm_mean")}]
    slope = np.polyfit(np.log(a.n_keys), np.log(a.eff_tokens_mean), 1)[0]
    stats.append({"stat": "log-log slope of the effective attended tokens against context length (pooled OLS)", "value": slope})
    pd.DataFrame(rows).to_csv(OUT / "fig2_attention_vs_length.csv", index=False)
    pd.DataFrame(stats).to_csv(OUT / "fig2_attention_vs_length_stats.csv", index=False)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1.25, 1, 1]})
        ax = axes[0]
        nudge = {"stem": 1.18, "conclusion": 0.86, "format": 1.22, "thinking": 0.84}   # keep neighbouring labels apart on the log axis
        for part, lab, color, ls, mk in FIG2_PARTS:
            g = a.groupby("len_bin").agg(x=("rel", "median"), y=(f"mass_{part}", "median"))
            ax.plot(2 ** g.x, g.y, color=color, ls=ls, marker=mk, ms=3, lw=0.9, markerfacecolor="white" if mk in ("D", "v") else color, markeredgewidth=0.7)
            ax.text(2 ** g.x.iloc[-1] * 1.08, g.y.iloc[-1] * nudge.get(part, 1.0), lab, fontsize=6, va="center", ha="left", color=color)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(0.5, 4.6)
        ax.set_ylim(0.03, 0.6)
        ax.set_xticks([0.5, 1, 2]); ax.set_xticklabels(["0.5", "1", "2"])
        ax.set_yticks([0.05, 0.1, 0.2, 0.5]); ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5"])
        ax.set_xlabel("thinking length / prompt median")
        ax.set_ylabel("attention share")
        ax = axes[1]
        ax.scatter(a.n_keys, a.eff_tokens_mean, s=2, color=LIGHT, edgecolors="none", rasterized=True)
        g = a.groupby("ctx_bin").agg(x=("n_keys", "median"), y=("eff_tokens_mean", "median"))
        ax.plot(g.x, g.y, color=BLACK, marker="o", ms=3, lw=0.9, label="median per decile")
        x0, y0 = g.x.iloc[4], g.y.iloc[4]
        xs = np.array([a.n_keys.min(), a.n_keys.max()])
        ax.plot(xs, y0 * xs / x0, color=MID, ls="--", lw=0.8, label="proportional to length")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_yticks([20, 50, 100]); ax.set_yticklabels(["20", "50", "100"])
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("effective tokens attended")
        ax.legend(fontsize=6, loc="upper left")
        ax = axes[2]
        ax.scatter(a.n_keys, a.ent_norm_mean, s=2, color=LIGHT, edgecolors="none", rasterized=True)
        g = a.groupby("ctx_bin").agg(x=("n_keys", "median"), y=("ent_norm_mean", "median"))
        ax.plot(g.x, g.y, color=BLACK, marker="o", ms=3, lw=0.9)
        ax.set_xscale("log")
        ax.set_ylim(0.2, 0.5)
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("entropy / log(context length)")
        for ax, letter in zip(axes, "abc"):
            ax.text(-0.3, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
        fig.tight_layout(w_pad=1.2)
        fig.savefig(OUT / "fig2_attention_vs_length.png")
        plt.close(fig)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(stats).round(3).to_string(index=False))
    print(pd.DataFrame(rows)[lambda d: d.panel == "a"].pivot(index="bin", columns="series", values="median").round(3).to_string())
    print(f"-> {OUT / 'fig2_attention_vs_length.png'}, {OUT / 'fig2_attention_vs_length.csv'}, {OUT / 'fig2_attention_vs_length_stats.csv'}")


# ------------------------------------------------------------------------------------------------------------------- fig 3

def jumps() -> pd.DataFrame:
    """Per trace: the conclusion's attention share at the answer-writing position and the closed jump across the conclusion unit,
    both for the model's own sentence and for the written "Therefore, the answer is L.", toward the trace's final answer."""
    a = answer_masses()
    sign = np.where(a.correct, 1.0, -1.0)
    a["jump_own"] = a.jump_closed * sign             # last_sentence_main.csv is signed toward the correct letter
    if "d_therefore_own" not in a:
        a = a.merge(pd.read_csv(SCREEN / "medxpertqa_letter_sentence.csv", usecols=["sample_id", "d_therefore_own"]), on="sample_id", how="left")
    a["jump_written"] = a.d_therefore_own            # letter_sentence.csv is already signed toward the trace's own answer
    for c in ("mass_conclusion", "jump_own", "jump_written"):
        a[f"{c}_dev"] = a[c] - a.groupby("item_id")[c].transform("median")
    return a


def layer_profile(a: pd.DataFrame) -> pd.DataFrame:
    """Within-prompt Spearman of the two jumps with the conclusion's share in every layer (row a)."""
    recs = []
    for run in RUNS.values():
        for mp in sorted((run / "readouts").glob("attention-*.jsonl")):
            npz = np.load(mp.with_suffix(".npz"))
            for line in mp.open(encoding="utf-8"):
                m = json.loads(line)
                ri = m["kinds"].index("a")
                mass = npz[f"{m['i']}/mass_mean"][ri].astype(np.float32)       # [L, S]; segment 8 = the conclusion unit
                recs.append({"sample_id": m["sample_id"], **{f"L{layer}": float(mass[layer, 8]) for layer in range(mass.shape[0])}})
    L = pd.DataFrame(recs).merge(a[["sample_id", "item_id", "jump_own", "jump_written"]], on="sample_id")
    rows = []
    for layer in range(36):
        rows.append({"layer": layer, "share_median": L[f"L{layer}"].median(), "rho_own": wrho(L, f"L{layer}", "jump_own"), "rho_written": wrho(L, f"L{layer}", "jump_written")})
    return pd.DataFrame(rows)


def fig3() -> None:
    a = jumps()
    prof = layer_profile(a)
    stats = [{"stat": "within-prompt Spearman rho, own conclusion sentence (median over 24 prompts)", "value": wrho(a, "mass_conclusion", "jump_own")},
             {"stat": "within-prompt Spearman rho, written 'Therefore, the answer is L.'", "value": wrho(a, "mass_conclusion", "jump_written")},
             {"stat": "within-prompt Spearman rho of the jump (own) with thinking tokens", "value": wrho(a, "think_tokens", "jump_own")},
             {"stat": "within-prompt Spearman rho of the jump (written) with thinking tokens", "value": wrho(a, "think_tokens", "jump_written")}]
    from length_accuracy import ols_demeaned
    a["log2_tokens"] = np.log2(a.think_tokens)
    fit = []
    for y in ("jump_own", "jump_written"):
        for name, terms in (("tokens only", ["log2_tokens"]), ("tokens + conclusion share", ["log2_tokens", "mass_conclusion"])):
            fit += ols_demeaned(a, y, terms, name)
    bins = []
    for col, lab in (("jump_own", "own"), ("jump_written", "written")):
        d = a.dropna(subset=[col])
        d = d.assign(b=pd.qcut(d.mass_conclusion_dev, 10, labels=False))
        for b, h in d.groupby("b"):
            bins.append({"series": lab, "bin": b, "x_median": h.mass_conclusion_dev.median(), "y_median": h[f"{col}_dev"].median(), "y_q25": h[f"{col}_dev"].quantile(0.25), "y_q75": h[f"{col}_dev"].quantile(0.75), "n": len(h)})
    pd.DataFrame(bins).to_csv(OUT / "fig3_jump_vs_attention.csv", index=False)
    prof.to_csv(OUT / "fig3_jump_vs_attention_layers.csv", index=False)
    pd.concat([pd.DataFrame(stats), pd.DataFrame(fit).rename(columns={"coef": "value"})]).to_csv(OUT / "fig3_jump_vs_attention_stats.csv", index=False)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5))
        for ax, col, lab in ((axes[0], "jump_own", "model's own conclusion sentence"), (axes[1], "jump_written", 'written "Therefore, the answer is L."')):
            d = a.dropna(subset=[col])
            ax.scatter(d.mass_conclusion_dev, d[f"{col}_dev"], s=2, color=LIGHT, edgecolors="none", rasterized=True)
            b = pd.DataFrame(bins)
            b = b[b.series == ("own" if col == "jump_own" else "written")]
            ax.errorbar(b.x_median, b.y_median, yerr=[b.y_median - b.y_q25, b.y_q75 - b.y_median], fmt="o-", color=BLACK, ms=3, lw=0.9, elinewidth=0.6, capsize=1.5)
            ax.axhline(0, color=MID, lw=0.5); ax.axvline(0, color=MID, lw=0.5)
            ax.set_xlim(-0.02, 0.03); ax.set_ylim(-8, 10)
            ax.set_xlabel("conclusion's attention share\n(deviation from the prompt median)")
            ax.set_ylabel("jump across the conclusion (nats,\ndeviation from the prompt median)" if col == "jump_own" else "")
            ax.set_title(lab, fontsize=7.5)
        ax = axes[2]
        ax.plot(prof.layer, prof.rho_own, color=BLACK, lw=0.9, marker="o", ms=2.5, label="own sentence")
        ax.plot(prof.layer, prof.rho_written, color=MID, lw=0.9, marker="s", ms=2.5, ls="--", label="written sentence")
        ax.set_ylim(0, 0.8)
        ax.set_xlabel("layer")
        ax.set_ylabel("Spearman rho within prompt")
        ax.legend(fontsize=6, loc="lower right")
        for ax, letter in zip(axes, "abc"):
            ax.text(-0.3, 1.08, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
        fig.tight_layout(w_pad=1.2)
        fig.savefig(OUT / "fig3_jump_vs_attention.png")
        plt.close(fig)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(stats).round(3).to_string(index=False))
    print(pd.DataFrame(fit).round(3).to_string(index=False))
    print(prof.round(3).to_string(index=False))
    print(f"-> {OUT / 'fig3_jump_vs_attention.png'} (+ .csv, _layers.csv, _stats.csv)")


# ------------------------------------------------------------------------------------------------------------------- fig 4

FIG4_PARTS = [("sink", "attention sink", BLACK, "-", "o"), ("format", "format tokens", MID, "--", "s"), ("stem_options", "stem + options", BLACK, ":", "D"),
              ("older", "own sentences, older", DARK, "-", "^"), ("last3", "last 3 own sentences", BLACK, "-.", "v")]
CLASSES = [("restate", "restated finding", BLACK, "-", "o"), ("inference", "inference", DARK, "--", "s"), ("reconsider", "reconsideration", MID, "-.", "^"), ("doubt", "doubt", BLACK, ":", "D")]


def fig4() -> None:
    w = pd.read_csv(SCREEN / "medxpertqa_attention_within.csv")
    w["mass_format"] = w.mass_template + w.mass_instruction + w.mass_think_open + w.mass_close
    w["mass_stem_options"] = w.mass_stem + w.mass_option_A + w.mass_option_B
    counts = pd.read_csv(SCREEN / "medxpertqa_inflation_traces.csv", usecols=["sample_id", "t", "restate", "inference", "reconsider", "doubt"])
    w = w.merge(counts, on=["sample_id", "t"], how="left")
    for c, *_ in CLASSES:
        w[f"per_{c}"] = np.where(w[c] > 0, w[f"mass_sent_{c}"] / w[c].replace(0, np.nan), np.nan)     # share per sentence of that kind, when any exists
    med = w.groupby("t").agg(**{f"mass_{p}": (f"mass_{p}", "median") for p, *_ in FIG4_PARTS}, ent_norm=("ent_norm_mean", "median"),
                             ent_q25=("ent_norm_mean", lambda s_: s_.quantile(0.25)), ent_q75=("ent_norm_mean", lambda s_: s_.quantile(0.75)),
                             **{f"per_{c}": (f"per_{c}", "median") for c, *_ in CLASSES}, **{f"n_{c}": (f"per_{c}", "count") for c, *_ in CLASSES}, n=("sample_id", "size")).reset_index()
    med.to_csv(OUT / "fig4_attention_within_trace.csv", index=False)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5))
        ax = axes[0]
        body = med[med.t < 1]
        stop = med[med.t == 1]
        for p, lab, color, ls, mk in FIG4_PARTS:
            ax.plot(body.t, body[f"mass_{p}"], color=color, ls=ls, lw=0.9, label=lab)
            ax.plot(stop.t, stop[f"mass_{p}"], marker=mk, ms=3.5, color=color, ls="none", markerfacecolor="white" if mk in ("D", "v") else color, markeredgewidth=0.7)
        ax.set_xlabel("proportion of the thinking kept")
        ax.set_ylabel("attention share of the last token kept")
        ax.set_ylim(0, 0.64)
        ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.6, 1.0), handlelength=2.2)
        ax = axes[1]
        ax.fill_between(body.t, body.ent_q25, body.ent_q75, color=LIGHT, lw=0)
        ax.plot(body.t, body.ent_norm, color=BLACK, lw=0.9)
        ax.errorbar(stop.t, stop.ent_norm, yerr=[stop.ent_norm - stop.ent_q25, stop.ent_q75 - stop.ent_norm], fmt="o", color=BLACK, ms=3.5, elinewidth=0.6, capsize=1.5)
        ax.annotate("stop\n(writes </think>)", (1.0, float(stop.ent_norm.iloc[0])), xytext=(-30, -22), textcoords="offset points", fontsize=6, ha="center", arrowprops={"arrowstyle": "-", "lw": 0.5})
        ax.set_xlabel("proportion of the thinking kept")
        ax.set_ylabel("entropy / log(context length)")
        ax.set_ylim(0.25, 0.5)
        ax = axes[2]
        for j, (c, lab, color, ls, mk) in enumerate(CLASSES):
            d = med[(med.t < 1) & (med[f"n_{c}"] >= 200)]
            ax.plot(d.t, d[f"per_{c}"], color=color, ls=ls, lw=0.9, label=lab)
            s1 = stop[stop[f"n_{c}"] >= 200]
            ax.plot(s1.t + 0.012 * j, s1[f"per_{c}"], marker=mk, ms=3.5, color=color, ls="none", markerfacecolor="white" if mk == "D" else color, markeredgewidth=0.7)
        ax.set_xlabel("proportion of the thinking kept")
        ax.set_ylabel("attention share per sentence")
        ax.set_yscale("log")
        ax.legend(fontsize=5.5, loc="upper right", handlelength=2.2)
        for ax, letter in zip(axes, "abc"):
            ax.text(-0.3, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
        fig.tight_layout(w_pad=1.2)
        fig.savefig(OUT / "fig4_attention_within_trace.png")
        plt.close(fig)
    pd.set_option("display.width", 250)
    show = med[med.t.isin([0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])]
    print(show[["t"] + [f"mass_{p}" for p, *_ in FIG4_PARTS] + ["ent_norm"] + [f"per_{c}" for c, *_ in CLASSES]].round(3).to_string(index=False))
    print(f"-> {OUT / 'fig4_attention_within_trace.png'}, {OUT / 'fig4_attention_within_trace.csv'}")


# --------------------------------------------------------------------------------------------------------- table 2 and fig 5

VARIANT_ROWS = [
    ("base", "nothing", "", "", "reproduces the stored letter log-odds"),
    ("concl_all", "the conclusion unit", "</think> and every later position", "all 36", "the conclusion's whole effect on the answer"),
    ("concl_answer", "the conclusion unit", "the answer-writing position only", "all 36", "direct read vs relay through </think>"),
    ("concl_early", "the conclusion unit", "</think> and every later position", "0-11", "which layers carry it"),
    ("concl_mid", "the conclusion unit", "</think> and every later position", "12-23", "which layers carry it"),
    ("concl_late", "the conclusion unit", "</think> and every later position", "24-35", "which layers carry it"),
    ("prev_all", "the sentence before the conclusion unit", "</think> and every later position", "all 36", "control: a neighbouring sentence"),
    ("rand_all", "one earlier sentence (seeded by the trace id)", "</think> and every later position", "all 36", "control: an arbitrary sentence"),
    ("think_all", "all thinking before the conclusion unit", "</think> and every later position", "all 36", "the conclusion read without competition"),
]


def ablation_table() -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from attention_at_answer import prompt_segments
    from generate import MODEL, MODEL_REVISION, TASKS, iter_traces, load_tasks

    tok = AutoTokenizer.from_pretrained(snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True))
    items = {it["item_id"]: it for it in load_tasks(TASKS)}
    order = EXAMPLE.rsplit("#", 1)[0].rsplit(":", 1)[1]
    m = next(json.loads(l) for l in (RUNS[order] / "readouts/ablation-01.jsonl").open(encoding="utf-8") if json.loads(l)["sample_id"] == EXAMPLE)
    tr = next(r for r in iter_traces(RUNS[order], ("sample_id", "token_ids", "answer")) if r["sample_id"] == EXAMPLE)
    p_ids, _ = prompt_segments(tok, items[m["item_id"]])
    ids = p_ids + tr["token_ids"][: m["answer_pos"]]
    blocks = m["blocks"]
    key = {"concl_all": "concl", "concl_answer": "concl", "concl_early": "concl", "concl_mid": "concl", "concl_late": "concl", "prev_all": "prev", "rand_all": "rand", "think_all": "think"}
    rows = []
    for v, hidden, queries, layers, purpose in VARIANT_ROWS:
        b = blocks.get(key.get(v, ""))
        text = tok.decode(ids[b[0]: b[1]]) if b else ""
        if len(text) > 90:
            text = text[:60].rstrip() + " ... " + text[-25:].lstrip()
        rows.append({"variant": v, "hidden_tokens": hidden, "hidden_from_queries": queries, "layers": layers, "purpose": purpose,
                     "example_hidden_tokens": f"{b[0]}-{b[1] - 1}" if b else "", "example_hidden_text": text.replace("\n", "\\n"),
                     "example_logodds_A_minus_B": round(m[v]["logp_A"] - m[v]["logp_B"], 2), "example_letter_written": m[v]["top1"]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table2_ablation.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(f"example {EXAMPLE}: sequence {m['n_total']} tokens, prompt {m['n_prompt']}, conclusion unit {blocks['concl']}, answer-writing position {m['n_total'] - 1}, answer {tr['answer']}")
    print(t[["variant", "hidden_tokens", "hidden_from_queries", "layers", "example_hidden_tokens", "example_logodds_A_minus_B", "example_letter_written"]].to_string(index=False))
    print(f"-> {OUT / 'table2_ablation.csv'}")


def fig5() -> None:
    t = pd.read_csv(SCREEN / "medxpertqa_ablation_conclusion.csv")
    conds = [("y_onset", "thinking cut before\nthe conclusion"), ("y_base", "nothing hidden"), ("y_concl_all", "conclusion hidden"), ("y_concl_answer", "conclusion hidden from\nthe answer position only"),
             ("y_prev_all", "previous sentence\nhidden"), ("y_rand_all", "a random earlier\nsentence hidden")]
    summ = pd.DataFrame([{"condition": lab, "median": t[c].median(), "q25": t[c].quantile(0.25), "q75": t[c].quantile(0.75),
                          "flip_share": float((np.sign(t[c]) != np.sign(t.y_base)).mean()) if c != "y_base" else 0.0, "n": int(t[c].notna().sum())} for c, lab in conds])
    big = t[t.jump.abs() > 2]
    stats = pd.DataFrame([{"stat": "effect of hiding the conclusion (nats removed), median", "value": t.effect_concl_all.median()},
                          {"stat": "effect of hiding it from the answer position only, median", "value": t.effect_concl_answer.median()},
                          {"stat": "ratio effect / truncation jump, median over traces with |jump| > 2 nats", "value": big.removed_share_concl_all.median()},
                          {"stat": "within-prompt Spearman rho of the effect with the truncation jump", "value": wrho(t, "jump", "effect_concl_all")},
                          {"stat": "ratio answer-position-only / all-positions effect, median over traces with |effect| > 1 nat", "value": t[t.effect_concl_all.abs() > 1].direct_share.median()},
                          {"stat": "share of traces whose letter flips when the conclusion is hidden", "value": float((np.sign(t.y_concl_all) != np.sign(t.y_base)).mean())}])
    summ.to_csv(OUT / "fig5_ablation_conclusion.csv", index=False)
    stats.to_csv(OUT / "fig5_ablation_conclusion_stats.csv", index=False)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.7), gridspec_kw={"width_ratios": [1.25, 1, 1]})
        ax = axes[0]
        y = np.arange(len(conds))[::-1]
        colors = [MID, BLACK, BLACK, BLACK, LIGHT, LIGHT]
        for yi, (c, lab), col in zip(y, conds, colors):
            ax.barh(yi, t[c].median(), color=col if col != BLACK else LIGHT, edgecolor=BLACK, linewidth=0.6, height=0.65, hatch="////" if c == "y_onset" else None)
            ax.errorbar(t[c].median(), yi, xerr=[[t[c].median() - t[c].quantile(0.25)], [t[c].quantile(0.75) - t[c].median()]], fmt="none", ecolor=BLACK, elinewidth=0.6, capsize=2)
        ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in conds], fontsize=6.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("answer lean toward the final answer (nats)")
        ax.set_xlim(0, 13)
        for ax_, xcol, ycol, xlabel, ylabel in ((axes[1], "jump", "effect_concl_all", "jump across the conclusion\nby truncation (nats)", "removed by hiding\nthe conclusion (nats)"),
                                              (axes[2], "effect_concl_all", "effect_concl_answer", "removed, hidden from\nall later positions (nats)", "removed, hidden from the\nanswer position only (nats)")):
            d = t.dropna(subset=[xcol, ycol])
            ax_.scatter(d[xcol], d[ycol], s=2, color=LIGHT, edgecolors="none", rasterized=True)
            b = d.assign(bin=pd.qcut(d[xcol], 10, labels=False)).groupby("bin").agg(x=(xcol, "median"), y=(ycol, "median"), lo=(ycol, lambda s_: s_.quantile(0.25)), hi=(ycol, lambda s_: s_.quantile(0.75)))
            ax_.errorbar(b.x, b.y, yerr=[b.y - b.lo, b.hi - b.y], fmt="o", color=BLACK, ms=3, elinewidth=0.6, capsize=1.5)
            lim = (-4, 16)
            ax_.plot(lim, lim, color=MID, lw=0.6, ls="--")
            ax_.set_xlim(lim); ax_.set_ylim(lim)
            ax_.set_xlabel(xlabel); ax_.set_ylabel(ylabel)
            ax_.set_xticks([0, 5, 10, 15]); ax_.set_yticks([0, 5, 10, 15])
        for ax_, letter, x in zip(axes, "abc", (-0.75, -0.38, -0.38)):
            ax_.text(x, 1.04, letter, transform=ax_.transAxes, fontsize=9, fontweight="bold", va="bottom")
        fig.tight_layout(w_pad=1.0)
        fig.savefig(OUT / "fig5_ablation_conclusion.png")
        plt.close(fig)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False)); print(stats.round(3).to_string(index=False))
    print(f"-> {OUT / 'fig5_ablation_conclusion.png'} (+ .csv, _stats.csv)")


# ------------------------------------------------------------------------------------------------------------------- fig 6

def fig6() -> None:
    t = pd.read_csv(SCREEN / "medxpertqa_ablation_conclusion.csv")
    t["effect_sum"] = t.effect_concl_early + t.effect_concl_mid + t.effect_concl_late
    conds = [("effect_concl_early", "layers 0-11"), ("effect_concl_mid", "layers 12-23"), ("effect_concl_late", "layers 24-35"), ("effect_sum", "sum of the three"), ("effect_concl_all", "all layers at once")]
    summ = pd.DataFrame([{"condition": lab, "median": t[c].median(), "q25": t[c].quantile(0.25), "q75": t[c].quantile(0.75),
                          "flip_share": float((np.sign(t.y_base - t[c]) != np.sign(t.y_base)).mean()) if c != "effect_sum" else np.nan} for c, lab in conds])
    big = t[t.effect_concl_all.abs() > 1]
    stats = pd.DataFrame([{"stat": "ratio (early + mid + late) / all-layer effect, median over traces with |all| > 1 nat", "value": (big.effect_sum / big.effect_concl_all).median()},
                          {"stat": "within-prompt Spearman rho of the sum with the all-layer effect", "value": wrho(t, "effect_concl_all", "effect_sum")},
                          {"stat": "share of the sum carried by layers 12-23, median", "value": (big.effect_concl_mid / big.effect_sum).median()},
                          {"stat": "share of the sum carried by layers 24-35, median", "value": (big.effect_concl_late / big.effect_sum).median()}])
    summ.to_csv(OUT / "fig6_ablation_layers.csv", index=False)
    stats.to_csv(OUT / "fig6_ablation_layers_stats.csv", index=False)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5), gridspec_kw={"width_ratios": [1.3, 1]})
        ax = axes[0]
        y = np.arange(len(conds))[::-1]
        for yi, (c, lab) in zip(y, conds):
            ax.barh(yi, t[c].median(), color=LIGHT if c.startswith("effect_concl_") and c != "effect_concl_all" else MID, edgecolor=BLACK, linewidth=0.6, height=0.65,
                    hatch="////" if c == "effect_sum" else None)
            ax.errorbar(t[c].median(), yi, xerr=[[t[c].median() - t[c].quantile(0.25)], [t[c].quantile(0.75) - t[c].median()]], fmt="none", ecolor=BLACK, elinewidth=0.6, capsize=2)
        ax.axvline(0, color=BLACK, lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in conds])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("removed from the answer lean by hiding\nthe conclusion in those layers (nats)")
        ax = axes[1]
        d = t.dropna(subset=["effect_sum", "effect_concl_all"])
        ax.scatter(d.effect_concl_all, d.effect_sum, s=2, color=LIGHT, edgecolors="none", rasterized=True)
        b = d.assign(bin=pd.qcut(d.effect_concl_all, 10, labels=False)).groupby("bin").agg(x=("effect_concl_all", "median"), y=("effect_sum", "median"), lo=("effect_sum", lambda s_: s_.quantile(0.25)), hi=("effect_sum", lambda s_: s_.quantile(0.75)))
        ax.errorbar(b.x, b.y, yerr=[b.y - b.lo, b.hi - b.y], fmt="o", color=BLACK, ms=3, elinewidth=0.6, capsize=1.5)
        lim = (-4, 16)
        ax.plot(lim, lim, color=MID, lw=0.6, ls="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xticks([0, 5, 10, 15]); ax.set_yticks([0, 5, 10, 15])
        ax.set_xlabel("removed, all layers at once (nats)")
        ax.set_ylabel("sum of the three layer groups (nats)")
        for ax_, letter, x in zip(axes, "ab", (-0.45, -0.3)):
            ax_.text(x, 1.04, letter, transform=ax_.transAxes, fontsize=9, fontweight="bold", va="bottom")
        fig.tight_layout(w_pad=1.0)
        fig.savefig(OUT / "fig6_ablation_layers.png")
        plt.close(fig)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False)); print(stats.round(3).to_string(index=False))
    print(f"-> {OUT / 'fig6_ablation_layers.png'} (+ .csv, _stats.csv)")


# ------------------------------------------------------------------------------------------------------------------- fig 8

def fig8() -> None:
    t = pd.read_csv(SCREEN / "medxpertqa_ablation_conclusion.csv")
    pr = pd.read_csv(SCREEN / "medxpertqa_prompts.csv", usecols=["item_id", "nothink_gap"])
    t = t.merge(pr, on="item_id")
    t["y_none"] = np.where(t.correct, t.nothink_gap, -t.nothink_gap)                       # prompt only, toward the final answer
    t["y_additive"] = t.y_none + (t.y_onset - t.y_none) + (t.y_think_all - t.y_none)       # if the two parts added up
    t["rise"] = -t.effect_think_all                                                       # lean gained when the earlier thinking is hidden
    contexts = [("y_none", "prompt only"), ("y_onset", "prompt + earlier thinking"), ("y_think_all", "prompt + conclusion\n(earlier thinking hidden)"), ("y_base", "prompt + both"), ("y_additive", "if the two parts added up")]
    summ = pd.DataFrame([{"context": lab, "median": t[c].median(), "q25": t[c].quantile(0.25), "q75": t[c].quantile(0.75)} for c, lab in contexts])
    t["support_bin"] = pd.qcut(t.y_onset, 8, labels=False)
    by_support = t.groupby("support_bin").agg(y_onset=("y_onset", "median"), rise=("rise", "median"), q25=("rise", lambda s_: s_.quantile(0.25)), q75=("rise", lambda s_: s_.quantile(0.75)), n=("rise", "size")).reset_index()
    by_len = t.groupby("quintile").agg(rel=("rel", "median"), rise=("rise", "median"), q25=("rise", lambda s_: s_.quantile(0.25)), q75=("rise", lambda s_: s_.quantile(0.75)), n=("rise", "size")).reset_index()
    stats = pd.DataFrame([{"stat": "rise from hiding the earlier thinking, median (nats)", "value": t.rise.median()},
                          {"stat": "share of traces whose earlier thinking alone favours the final answer (y_onset > 0)", "value": float((t.y_onset > 0).mean())},
                          {"stat": "share of traces whose lean rises when the earlier thinking is hidden", "value": float((t.rise > 0).mean())},
                          {"stat": "within-prompt Spearman rho of the rise with the support of the earlier thinking (y_onset)", "value": wrho(t, "y_onset", "rise")},
                          {"stat": "within-prompt Spearman rho of the rise with thinking tokens", "value": wrho(t, "think_tokens", "rise")},
                          {"stat": "rise in the most-supportive quarter of traces (y_onset top quartile), median", "value": t[t.y_onset >= t.y_onset.quantile(0.75)].rise.median()}])
    summ.to_csv(OUT / "fig8_ablation_thinking.csv", index=False)
    pd.concat([by_support.assign(panel="b"), by_len.assign(panel="c")]).to_csv(OUT / "fig8_ablation_thinking_bins.csv", index=False)
    stats.to_csv(OUT / "fig8_ablation_thinking_stats.csv", index=False)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1.3, 1, 1]})
        ax = axes[0]
        y = np.arange(len(contexts))[::-1]
        for yi, (c, lab) in zip(y, contexts):
            ax.barh(yi, t[c].median(), color=MID if c == "y_base" else LIGHT, edgecolor=BLACK, linewidth=0.6, height=0.65, hatch="////" if c == "y_additive" else None)
            ax.errorbar(t[c].median(), yi, xerr=[[t[c].median() - t[c].quantile(0.25)], [t[c].quantile(0.75) - t[c].median()]], fmt="none", ecolor=BLACK, elinewidth=0.6, capsize=2)
        ax.axvline(0, color=BLACK, lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in contexts], fontsize=6.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("answer lean toward\nthe final answer (nats)")
        ax = axes[1]
        ax.errorbar(by_support.y_onset, by_support.rise, yerr=[by_support.rise - by_support.q25, by_support.q75 - by_support.rise], fmt="o-", color=BLACK, ms=3, lw=0.9, elinewidth=0.6, capsize=1.5)
        ax.axhline(0, color=MID, lw=0.6, ls="--")
        ax.set_xlabel("lean of the earlier thinking\non its own (nats)")
        ax.set_ylabel("lean gained when the earlier\nthinking is hidden (nats)")
        ax = axes[2]
        ax.errorbar(2 ** by_len.rel, by_len.rise, yerr=[by_len.rise - by_len.q25, by_len.q75 - by_len.rise], fmt="o-", color=BLACK, ms=3, lw=0.9, elinewidth=0.6, capsize=1.5)
        ax.axhline(0, color=MID, lw=0.6, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 1, 2]); ax.set_xticklabels(["0.5", "1", "2"])
        ax.set_xlabel("thinking length /\nprompt median")
        ax.set_ylabel("")
        axes[1].set_ylim(-3, 7); axes[2].set_ylim(-3, 7)
        for ax_, letter, x in zip(axes, "abc", (-0.75, -0.38, -0.2)):
            ax_.text(x, 1.04, letter, transform=ax_.transAxes, fontsize=9, fontweight="bold", va="bottom")
        fig.tight_layout(w_pad=1.0)
        fig.savefig(OUT / "fig8_ablation_thinking.png")
        plt.close(fig)
    pd.set_option("display.width", 200)
    print(summ.round(2).to_string(index=False)); print(stats.round(3).to_string(index=False))
    print(by_support.round(2).to_string(index=False)); print(by_len.round(2).to_string(index=False))
    print(f"-> {OUT / 'fig8_ablation_thinking.png'} (+ .csv, _bins.csv, _stats.csv)")


# ------------------------------------------------------------------------------------------------------------------- fig 9

def fig9() -> None:
    t = pd.read_csv(SCREEN / "medxpertqa_ablation_conclusion.csv")
    t["effect_dev"] = t.effect_concl_all - t.groupby("item_id").effect_concl_all.transform("median")
    t["share_dev"] = t.mass_conclusion - t.groupby("item_id").mass_conclusion.transform("median")
    t["share_late_dev"] = t.mass_conclusion_late - t.groupby("item_id").mass_conclusion_late.transform("median")
    by_len = t.groupby("quintile").agg(rel=("rel", "median"), effect=("effect_concl_all", "median"), q25=("effect_concl_all", lambda s_: s_.quantile(.25)), q75=("effect_concl_all", lambda s_: s_.quantile(.75)),
                                       share=("mass_conclusion", "median"), n=("effect_concl_all", "size")).reset_index()
    stats = pd.DataFrame([{"stat": "within-prompt Spearman rho of the conclusion's ablation effect with thinking tokens", "value": wrho(t, "think_tokens", "effect_concl_all")},
                          {"stat": "within-prompt rho of the effect with the conclusion's attention share (all layers)", "value": wrho(t, "mass_conclusion", "effect_concl_all")},
                          {"stat": "within-prompt rho of the effect with the conclusion's share in layers 24-35", "value": wrho(t, "mass_conclusion_late", "effect_concl_all")},
                          {"stat": "within-prompt rho of the effect with the truncation jump", "value": wrho(t, "jump", "effect_concl_all")}])
    bins = []
    for col, lab in (("share_dev", "all layers"), ("share_late_dev", "layers 24-35")):
        d = t.dropna(subset=[col]).assign(b=lambda x: pd.qcut(x[col], 10, labels=False))
        for b, h in d.groupby("b"):
            bins.append({"series": lab, "bin": b, "x_median": h[col].median(), "y_median": h.effect_dev.median(), "y_q25": h.effect_dev.quantile(.25), "y_q75": h.effect_dev.quantile(.75), "n": len(h)})
    by_len.to_csv(OUT / "fig9_effect_vs_length.csv", index=False)
    pd.DataFrame(bins).to_csv(OUT / "fig9_effect_vs_length_bins.csv", index=False)
    stats.to_csv(OUT / "fig9_effect_vs_length_stats.csv", index=False)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5))
        ax = axes[0]
        ax.errorbar(2 ** by_len.rel, by_len.effect, yerr=[by_len.effect - by_len.q25, by_len.q75 - by_len.effect], fmt="o-", color=BLACK, ms=3, lw=0.9, elinewidth=0.6, capsize=1.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 1, 2]); ax.set_xticklabels(["0.5", "1", "2"])
        ax.set_xlabel("thinking length / prompt median")
        ax.set_ylabel("removed by hiding the conclusion (nats)")
        ax.set_ylim(0, 7)
        ax = axes[1]
        b = pd.DataFrame(bins)
        for lab, color, mk, ls in (("all layers", BLACK, "o", "-"), ("layers 24-35", MID, "s", "--")):
            h = b[b.series == lab]
            ax.errorbar(h.x_median * 100, h.y_median, yerr=[h.y_median - h.y_q25, h.y_q75 - h.y_median], fmt=mk + ls, color=color, ms=3, lw=0.9, elinewidth=0.6, capsize=1.5, label=lab)
        ax.axhline(0, color=MID, lw=0.5); ax.axvline(0, color=MID, lw=0.5)
        ax.set_xlabel("conclusion's attention share, % points\n(deviation from the prompt median)")
        ax.set_ylabel("removed by hiding the conclusion\n(nats, deviation from the prompt median)")
        ax.legend(fontsize=6, loc="upper left")
        for ax_, letter, x in zip(axes, "ab", (-0.3, -0.35)):
            ax_.text(x, 1.04, letter, transform=ax_.transAxes, fontsize=9, fontweight="bold", va="bottom")
        fig.tight_layout(w_pad=1.5)
        fig.savefig(OUT / "fig9_effect_vs_length.png")
        plt.close(fig)
    pd.set_option("display.width", 200)
    print(by_len.round(3).to_string(index=False)); print(stats.round(3).to_string(index=False))
    print(f"-> {OUT / 'fig9_effect_vs_length.png'} (+ .csv, _bins.csv, _stats.csv)")


# ------------------------------------------------------------------------------------------------------------------- fig 0

EXAMPLES = ["medxpertqa:Text-699:I-B:ab#47", "medxpertqa:Text-699:I-B:ab#98", "medxpertqa:Text-699:I-B:ab#34"]   # short / medium / long (10th percentile, 941 tokens, 90th percentile)


def examples() -> None:
    """The closed readout along three traces of one question (short, medium, long), x = thinking tokens, with the conclusion sentence."""
    ls = pd.read_csv(SCREEN / "medxpertqa_last_sentence_main.csv").set_index("sample_id")
    order = EXAMPLES[0].rsplit("#", 1)[0].rsplit(":", 1)[1]
    run = RUNS[order]
    want = set(EXAMPLES)
    ro = {}
    for line in (run / "readouts/readouts-01.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        if d["sample_id"] in want:
            ro[d["sample_id"]] = d
    item = ls.loc[EXAMPLES[0], "item_id"]
    y0 = next((d["logp_A"] - d["logp_B"]) for d in map(json.loads, (run / "readouts/readouts_start.jsonl").open(encoding="utf-8")) if d["item_id"] == item)
    label = ls.loc[EXAMPLES[0], "label"]
    sgn = 1.0 if label == "A" else -1.0
    rows = []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(6.0, 5.4), sharex=True, sharey=True)
        for ax, sid, name in zip(axes, EXAMPLES, ("short", "medium", "long")):
            d = ro[sid]
            start, end = d["think_start"], d["think_end"]
            pos = [start] + [p for p, k in zip(d["pos"], d["kind"]) if k in ("s", "e")]
            y = [sgn * y0] + [sgn * (a - b) for (a, b), k in zip(zip(d["closed"]["logp_A"], d["closed"]["logp_B"]), d["kind"]) if k in ("s", "e")]
            x = [p - start for p in pos]
            onset_x = ls.loc[sid, "onset"] - start
            ax.axhline(0, color=MID, lw=0.6, ls="--")
            ax.plot(x, y, color=BLACK, lw=0.9, marker="o", ms=2.2, markerfacecolor=BLACK)
            ax.plot(x[-1], y[-1], marker="o", ms=5, color=BLACK)
            ax.plot(onset_x, ls.loc[sid, "y_closed_onset"] * (1 if label == "A" else 1), marker="o", ms=5, markerfacecolor="white", markeredgecolor=BLACK, markeredgewidth=0.8)
            last = ls.loc[sid, "last_sentence"]
            y_on = float(ls.loc[sid, "y_closed_onset"])
            mid = ((onset_x + x[-1]) / 2, (y_on + y[-1]) / 2)
            xmax = max(ls.loc[e, "think_tokens"] for e in EXAMPLES)
            ax.set_xlim(-0.03 * xmax, 1.16 * xmax)
            # the jump across the conclusion sentence: a double arrow beside the last segment with its size
            xa = x[-1] + 0.02 * xmax
            ax.annotate("", xy=(xa, y[-1]), xytext=(xa, y_on), arrowprops={"arrowstyle": "<->", "lw": 0.8, "color": BLACK, "shrinkA": 0, "shrinkB": 0})
            ax.text(xa + 0.012 * xmax, (y_on + y[-1]) / 2, f"+{y[-1] - y_on:.1f} nats", fontsize=7, va="center", ha="left")
            # the conclusion sentence, above the final point, pointing at the last segment
            ax.annotate(f'"{last}"', xy=mid, xytext=(x[-1] + 0.04 * xmax, min(y[-1] + 5, 18.5)), textcoords="data", fontsize=7, ha="right", va="center",
                        arrowprops={"arrowstyle": "-|>", "lw": 0.7, "color": BLACK, "shrinkB": 2})
            ax.text(0.99, 0.95, f"{name}: {int(ls.loc[sid, 'think_tokens'])} tokens, {int(ls.loc[sid, 'n_sentences'])} sentences", transform=ax.transAxes, fontsize=7.5, va="top", ha="right")
            ax.set_ylim(-6, 20)
            ax.set_yticks([-5, 0, 5, 10, 15, 20])
            ax.set_ylabel("log p(A) − log p(B)")
            if name == "short":
                from matplotlib.lines import Line2D
                ax.legend(handles=[Line2D([], [], marker="o", ms=5, markerfacecolor="white", markeredgecolor=BLACK, color="none", label="before the conclusion sentence"),
                                   Line2D([], [], marker="o", ms=5, color=BLACK, ls="none", label="after it (the stop)")], fontsize=6.5, loc="center right", handletextpad=0.4)
            for k, (xx, yy) in enumerate(zip(x, y)):
                rows.append({"sample_id": sid, "trace": name, "k": k, "tokens_kept": xx, "readout": yy})
        axes[-1].set_xlabel("thinking tokens kept before the cut")
        fig.tight_layout(h_pad=0.6)
        fig.savefig(OUT / "fig0_example_trajectories.png")
        plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "fig0_example_trajectories.csv", index=False)
    print(f"-> {OUT / 'fig0_example_trajectories.png'}, .csv")


# -------------------------------------------------------------------------------------------------------------- jump vs length

def jump_vs_length() -> None:
    """The jump across the conclusion sentence against thinking length: the model's own sentence and the written "Therefore, the
    answer is L."; pooled across questions (x = tokens) and within question (x = tokens / prompt median, y = jump − prompt median)."""
    from scipy.stats import spearmanr
    ls = pd.read_csv(SCREEN / "medxpertqa_last_sentence_main.csv", usecols=["sample_id", "item_id", "correct", "think_tokens", "jump_closed"])
    lt = pd.read_csv(SCREEN / "medxpertqa_letter_sentence.csv", usecols=["sample_id", "d_therefore_own"])
    t = ls.merge(lt, on="sample_id")
    t["jump_own"] = np.where(t.correct, t.jump_closed, -t.jump_closed)
    t["jump_written"] = t.d_therefore_own
    t["log_tokens"] = np.log(t.think_tokens)
    t["rel"] = np.log2(t.think_tokens / t.groupby("item_id").think_tokens.transform("median"))
    for c in ("jump_own", "jump_written"):
        t[f"{c}_dev"] = t[c] - t.groupby("item_id")[c].transform("median")
    stats, bins = [], []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.0))
        for row, (col, lab) in enumerate((("jump_own", "the model's own conclusion sentence"), ("jump_written", 'written "Therefore, the answer is L."'))):
            for coli, (xcol, ycol, xlabel, ylabel, kind) in enumerate(((("think_tokens", col, "thinking tokens", "jump across the conclusion (nats)", "pooled")),
                                                                      ("rel", f"{col}_dev", "thinking tokens / prompt median", "jump, deviation from\nthe prompt median (nats)", "within"))):
                ax = axes[row, coli]
                d = t.dropna(subset=[xcol, ycol])
                xs = d[xcol] if kind == "pooled" else 2 ** d[xcol]
                ax.scatter(xs, d[ycol], s=2, color=LIGHT, edgecolors="none", rasterized=True)
                b = d.assign(bin=pd.qcut(d[xcol], 10, labels=False)).groupby("bin").agg(x=(xcol, "median"), y=(ycol, "median"), lo=(ycol, lambda s_: s_.quantile(.25)), hi=(ycol, lambda s_: s_.quantile(.75)), n=(ycol, "size"))
                bx = b.x if kind == "pooled" else 2 ** b.x
                ax.errorbar(bx, b.y, yerr=[b.y - b.lo, b.hi - b.y], fmt="o-", color=BLACK, ms=3, lw=0.9, elinewidth=0.6, capsize=1.5)
                if kind == "pooled":
                    r = spearmanr(d[xcol], d[ycol]).correlation
                    ax.set_xscale("log")
                    ax.set_xticks([500, 1000, 2000, 4000]); ax.set_xticklabels(["500", "1000", "2000", "4000"])
                    ax.set_ylim(-4, 14)
                else:
                    r = wrho(d, "think_tokens", col)
                    ax.axhline(0, color=MID, lw=0.5)
                    ax.set_xscale("log", base=2)
                    ax.set_xticks([0.5, 1, 2]); ax.set_xticklabels(["0.5", "1", "2"])
                    ax.set_xlim(0.3, 3.5)
                    ax.set_ylim(-6, 8)
                ax.text(0.98, 0.95, f"Spearman rho = {r:+.2f}" + ("" if kind == "pooled" else " (median over questions)"), transform=ax.transAxes, fontsize=6.5, ha="right", va="top")
                if row == 1:
                    ax.set_xlabel(xlabel)
                if coli == 0:
                    ax.set_ylabel(ylabel)
                else:
                    ax.set_ylabel(ylabel)
                ax.set_title(lab if coli == 0 else "", fontsize=7.5, loc="left")
                stats.append({"sentence": lab, "comparison": kind, "spearman_rho": r, "n": len(d)})
                for bi, h in b.iterrows():
                    bins.append({"sentence": lab, "comparison": kind, "bin": bi, "x_median": h.x if kind == "pooled" else 2 ** h.x, "jump_median": h.y, "q25": h.lo, "q75": h.hi, "n": int(h.n)})
        for ax, letter in zip(axes.flatten(), "abcd"):
            ax.text(-0.28, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
        fig.tight_layout(w_pad=1.5, h_pad=1.5)
        fig.savefig(OUT / "fig_jump_vs_length.png")
        plt.close(fig)
    pd.DataFrame(stats).to_csv(OUT / "fig_jump_vs_length_stats.csv", index=False)
    pd.DataFrame(bins).to_csv(OUT / "fig_jump_vs_length.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(stats).round(3).to_string(index=False))
    print(f"-> {OUT / 'fig_jump_vs_length.png'} (+ .csv, _stats.csv)")


# ---------------------------------------------------------------------------------------------------- jump-length correlation hist

def jump_length_hist() -> None:
    """Per question (pair x option order), the Spearman correlation between thinking length and the jump across the conclusion
    sentence; histogram over the 24 questions, for the model's own sentence and for the written sentence, with a permutation null."""
    from scipy.stats import rankdata
    ls = pd.read_csv(SCREEN / "medxpertqa_last_sentence_main.csv", usecols=["sample_id", "item_id", "correct", "think_tokens", "jump_closed"])
    lt = pd.read_csv(SCREEN / "medxpertqa_letter_sentence.csv", usecols=["sample_id", "d_therefore_own"])
    t = ls.merge(lt, on="sample_id")
    t["jump_own"] = np.where(t.correct, t.jump_closed, -t.jump_closed)
    t["jump_written"] = t.d_therefore_own
    rng = np.random.default_rng(0)
    rows, null = [], {"jump_own": [], "jump_written": []}
    for iid, g in t.groupby("item_id"):
        rt = rankdata(g.think_tokens)
        rtc = (rt - rt.mean()) / rt.std()
        for col in ("jump_own", "jump_written"):
            ry = rankdata(g[col])
            rows.append({"item_id": iid, "sentence": col, "rho": float(np.corrcoef(rt, ry)[0, 1]), "n": len(g)})
            perm = np.array([rng.permutation(ry) for _ in range(200)])
            pc = (perm - perm.mean(1, keepdims=True)) / perm.std(1, keepdims=True)
            null[col].append(pc @ rtc / len(rt))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "fig_jump_length_hist.csv", index=False)
    bins = np.arange(-1, 1.0001, 0.1)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5), sharey=True)
        for ax, (col, lab) in zip(axes, (("jump_own", "the model's own conclusion sentence"), ("jump_written", 'written "Therefore, the answer is L."'))):
            v = r[r.sentence == col].rho
            ax.hist(v, bins=bins, color=LIGHT, edgecolor=BLACK, linewidth=0.6)
            hn, _ = np.histogram(np.concatenate(null[col]), bins=bins)
            hn = hn * len(v) / hn.sum()
            ax.step(bins, np.r_[hn, hn[-1]], where="post", color=BLACK, lw=0.8, ls="--", label="permutation null")
            ax.axvline(0, color=MID, lw=0.6)
            ax.set_xlim(-1, 1)
            ax.set_xlabel("Spearman correlation of thinking length\nwith the jump across the conclusion, per question")
            ax.set_title(lab, fontsize=7.5, loc="left")
            ax.text(0.03, 0.95, f"median {v.median():+.2f}\n{(v < 0).mean():.0%} of questions below 0", transform=ax.transAxes, fontsize=6.5, va="top")
        axes[0].set_ylabel("questions")
        axes[1].legend(fontsize=6, loc="upper right")
        for ax, letter in zip(axes, "ab"):
            ax.text(-0.18, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
        fig.tight_layout(w_pad=1.5)
        fig.savefig(OUT / "fig_jump_length_hist.png")
        plt.close(fig)
    pd.set_option("display.width", 200)
    print(r.groupby("sentence").rho.describe().round(3).to_string())
    print(f"null sd (100 traces per question): own {np.concatenate(null['jump_own']).std():.3f}, written {np.concatenate(null['jump_written']).std():.3f}")
    print(f"-> {OUT / 'fig_jump_length_hist.png'}, .csv")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fig1"
    OUT.mkdir(parents=True, exist_ok=True)
    {"examples": examples, "jump": jump_vs_length, "jumphist": jump_length_hist, "positions": positions_table, "fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "ablation_table": ablation_table, "fig5": fig5, "fig6": fig6, "fig8": fig8, "fig9": fig9}[cmd]()
