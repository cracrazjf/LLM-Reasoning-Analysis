"""Main-run analysis on the reasoning-helps pairs: accuracy and thinking length by condition.

    python src/helps_analysis.py --run runs/medxpertqa/main-qwen3-8b

Reads every traces-*.jsonl of the run (baseline, careful, speed; both orders), drops
pair_distance.NON_DIAGNOSIS_PAIRS, and joins the no-thinking accuracy of each pair from
data/pairs/medxpertqa_gain.csv. Pairs are nested in questions: confidence intervals come from a
bootstrap over questions.

Tables (printed and written to <run>/analysis/helps_summary.json, per_pair.csv, per_prompt.csv):
  1. accuracy and thinking length per condition; pairwise condition differences with CIs
  2. the same by baseline-accuracy stratum
  3. by order (ab: correct option is A)
  4. within-prompt relation between length and correctness: thinking tokens of correct vs wrong
     traces of the same prompt, and the accuracy of the shortest vs longest third of its traces
  5. answer format and truncation per condition
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
from medxpertqa_dataset import ROOT  # noqa: E402
from pair_distance import NON_DIAGNOSIS_PAIRS  # noqa: E402

COND = {"C0_baseline": "baseline", "C1_caution_think": "careful", "C2_speed_think": "speed"}


def load(run: Path) -> pd.DataFrame:
    rows = []
    for r in iter_traces(run, ("sample_id", "prompt_id", "item_id", "pair_id", "source_id", "order", "condition", "label",
                               "answer_text", "think_tokens", "finish_reason", "X", "sample")):
        if r["pair_id"] in NON_DIAGNOSIS_PAIRS or r["condition"] not in COND:
            continue
        a, fmt = parse_answer(r["answer_text"], ("A", "B"))
        rows.append({**r, "cond": COND[r["condition"]], "answer": a, "fmt": fmt,
                     "correct": float(a == r["label"]) if a else np.nan, "is_B": float(a == "B") if a else np.nan})
    df = pd.DataFrame(rows).drop_duplicates("sample_id")
    return df


def boot_q(P: pd.DataFrame, a: str, b: str, n: int = 4000, seed: int = 0) -> list[float]:
    """CI of mean(a - b) over pairs, resampling questions."""
    rng = np.random.default_rng(seed)
    by = {q: (g[a] - g[b]).values for q, g in P.groupby("source_id")}
    qs = list(by)
    m = [np.concatenate([by[q] for q in rng.choice(qs, len(qs))]).mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    df = load(args.run)
    out = args.run / "analysis"
    out.mkdir(exist_ok=True)
    gain = pd.read_csv(ROOT / "data" / "pairs" / "medxpertqa_gain.csv").set_index("pair_id")
    ans = df.dropna(subset=["correct"])

    # per prompt and per pair
    pp = ans.groupby(["item_id", "pair_id", "source_id", "order", "cond"]).agg(
        n=("correct", "size"), acc=("correct", "mean"), think=("think_tokens", "median"), share_B=("is_B", "mean")).reset_index()
    pp.to_csv(out / "per_prompt.csv", index=False)
    P = pp.pivot_table(index=["pair_id", "source_id"], columns="cond", values="acc").reset_index()
    T = pp.pivot_table(index="pair_id", columns="cond", values="think")
    P["nothink"] = P.pair_id.map(gain.acc_nothink)
    for c in COND.values():
        P[f"think_{c}"] = P.pair_id.map(T[c])
    P.to_csv(out / "per_pair.csv", index=False)

    S: dict = {"run": str(args.run), "n_traces": int(len(df)), "n_pairs": int(P.pair_id.nunique()),
               "n_questions": int(P.source_id.nunique()),
               "traces_per_prompt": {c: int(pp[pp.cond == c].n.median()) for c in COND.values()}}
    S["by_condition"] = {c: {"accuracy": float(P[c].mean()), "think_tokens_median": float(df[df.cond == c].think_tokens.median()),
                             "think_tokens_p10": float(df[df.cond == c].think_tokens.quantile(.1)),
                             "think_tokens_p90": float(df[df.cond == c].think_tokens.quantile(.9)),
                             "share_B": float(ans[ans.cond == c].is_B.mean()),
                             "answered": float(df[df.cond == c].answer.notna().mean()),
                             "bare_letter": float((df[df.cond == c].fmt == "letter").mean()),
                             "truncated": int((df[df.cond == c].finish_reason == "length").sum())} for c in COND.values()}
    S["no_thinking_accuracy"] = float(P.nothink.mean())
    S["differences"] = {}
    for a, b in (("careful", "baseline"), ("speed", "baseline"), ("careful", "speed")):
        d = P[a] - P[b]
        S["differences"][f"{a}-{b}"] = {"mean": float(d.mean()), "ci95": boot_q(P, a, b), "pairs_up": int((d > 0).sum()),
                                        "pairs_down": int((d < 0).sum()),
                                        "length_ratio_median": float((P[f"think_{a}"] / P[f"think_{b}"]).median())}
    S["monotone_pairs"] = {"strict": int(((P.speed < P.baseline) & (P.baseline < P.careful)).sum()),
                           "weak": int(((P.speed <= P.baseline) & (P.baseline <= P.careful)).sum())}
    P["stratum"] = pd.cut(P.baseline, [-0.01, 0.5, 0.8, 0.95, 1.0], labels=["<0.5", "0.5-0.8", "0.8-0.95", ">=0.95"])
    S["by_stratum"] = {str(k): {"n_pairs": int(len(g)), "nothink": float(g.nothink.mean()), "speed": float(g.speed.mean()),
                                "baseline": float(g.baseline.mean()), "careful": float(g.careful.mean())}
                       for k, g in P.groupby("stratum", observed=True)}
    S["by_order"] = {o: {c: float(g[g.cond == c].acc.mean()) for c in COND.values()} for o, g in pp.groupby("order")}

    # within-prompt: length vs correctness
    W = {}
    for c in COND.values():
        d = ans[ans.cond == c]
        diffs, lo_acc, hi_acc = [], [], []
        for _, g in d.groupby("item_id"):
            if g.correct.nunique() == 2:
                diffs.append(g[g.correct == 1].think_tokens.median() - g[g.correct == 0].think_tokens.median())
            if len(g) >= 15:
                q1, q2 = g.think_tokens.quantile([1 / 3, 2 / 3])
                lo_acc.append(g[g.think_tokens <= q1].correct.mean())
                hi_acc.append(g[g.think_tokens >= q2].correct.mean())
        W[c] = {"prompts_with_both_outcomes": len(diffs),
                "median_think_correct_minus_wrong": float(np.median(diffs)) if diffs else None,
                "share_prompts_wrong_traces_longer": float(np.mean([x < 0 for x in diffs])) if diffs else None,
                "acc_shortest_third": float(np.mean(lo_acc)), "acc_longest_third": float(np.mean(hi_acc))}
    S["within_prompt_length_vs_correct"] = W
    (out / "helps_summary.json").write_text(json.dumps(S, indent=1) + "\n")

    print(f"{S['n_traces']} traces, {S['n_pairs']} pairs, {S['n_questions']} questions; traces per prompt {S['traces_per_prompt']}")
    print(f"\n{'':<12}{'no-think':>9}{'speed':>8}{'baseline':>10}{'careful':>9}")
    print(f"{'accuracy':<12}{S['no_thinking_accuracy']:>9.3f}{S['by_condition']['speed']['accuracy']:>8.3f}"
          f"{S['by_condition']['baseline']['accuracy']:>10.3f}{S['by_condition']['careful']['accuracy']:>9.3f}")
    print(f"{'think med':<12}{0:>9}{S['by_condition']['speed']['think_tokens_median']:>8.0f}"
          f"{S['by_condition']['baseline']['think_tokens_median']:>10.0f}{S['by_condition']['careful']['think_tokens_median']:>9.0f}")
    for k, v in S["differences"].items():
        print(f"  {k:<18} {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  pairs up {v['pairs_up']} down {v['pairs_down']}  length x{v['length_ratio_median']:.2f}")
    print("monotone pairs:", S["monotone_pairs"])
    print("\nby baseline stratum (n, no-think, speed, baseline, careful):")
    for k, v in S["by_stratum"].items():
        print(f"  {k:<9}{v['n_pairs']:>3}  {v['nothink']:.2f}  {v['speed']:.2f}  {v['baseline']:.2f}  {v['careful']:.2f}")
    print("by order:", {o: {c: round(x, 3) for c, x in v.items()} for o, v in S["by_order"].items()})
    print("\nwithin prompt, length vs correctness:")
    for c, v in W.items():
        print(f"  {c:<9} prompts {v['prompts_with_both_outcomes']:>3}  think(correct)-think(wrong) median {v['median_think_correct_minus_wrong']:+.0f}  "
              f"wrong longer in {v['share_prompts_wrong_traces_longer']:.2f}  acc shortest third {v['acc_shortest_third']:.3f} longest third {v['acc_longest_third']:.3f}")
    print("\nformat:", {c: {k: round(v[k], 3) if isinstance(v[k], float) else v[k] for k in ("answered", "bare_letter", "truncated", "share_B")} for c, v in S["by_condition"].items()})
    print(f"-> {out}/helps_summary.json, per_pair.csv, per_prompt.csv")


if __name__ == "__main__":
    main()
