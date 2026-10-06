"""Readout paths of a run: X after every sentence of the thinking, with verdict sentences marked.

    python src/paths.py --run runs/medxpertqa/main-qwen3-8b

Joins <run>/readouts/ (src/readout.py) with the traces. The readout positions are the sentence
starts, so sentence k of a trace is the text between readout k-1 and readout k, and readout k is
the state after sentence k: X = log p(A) - log p(B) if the thinking were closed there. The last
readout of a trace is the one at the model's own stop.

A verdict sentence states an answer letter outright ("So the answer is B.", "I think A is more
likely.", "I'll go with A."); questions and rejections ("A is not correct") do not count. Verdicts
are found by pattern on the sentence text, so they are a marker with some misses, not ground truth.

Writes <run>/analysis/paths.npz and paths_traces.parquet; other scripts read them with load_paths().
  per sentence (flat, traces concatenated): t = thinking tokens up to the end of the sentence;
      verdict (0 none, 1 for A, 2 for B); from the closed context X = z_A - z_B, logp_A, logp_B,
      z_A, z_B, lse_other and max_other (the stored candidates that are not a letter), lse, mean,
      sd (whole vocabulary); from the open context z_think, logp_think
  per trace: ids, condition, label, answer, correct, think_tokens, n (sentences), off (offset of
      its first sentence in the flat arrays), and the prompt's readout with empty thinking: X0,
      z0_A, z0_B, lse0_other, lse0, mean0
Runs read out in 2026-10 with the earlier scripts (readouts/ with X and log p, readouts_logits/
with the raw logits) load the same way; values they did not record are NaN.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from generate import MODEL, iter_traces, parse_answer  # noqa: E402

COND = {"C0_baseline": "baseline", "C1_caution_think": "careful", "C2_speed_think": "speed"}
LETTER = r"(?<![\w'/-])((?-i:[AB]))(?![\w'/-])"   # capital letter only, also inside case-insensitive patterns
VERDICT = [
    # "the answer is B", "I'll go with A", "the most likely diagnosis is B"
    re.compile(r"\b(?:answer|go(?:ing)? with|choose|pick|select|stick with|settle on|lean(?:ing)? towards?|diagnosis (?:is|would be))\b[^.?!\n]{0,60}?" + LETTER),
    # "A is more likely", "B might be the answer", "B seems correct"
    re.compile(LETTER + r"\s*(?:is|seems|appears|looks|would be|might be|may be|must be|should be|could be)\b[^.?!\n]{0,40}?\b(?:answer|correct|right|more likely|most likely|likely|best|better)\b"),
    # "So B.", "Therefore, A.", "So maybe A", "So B: right-on-right sacral torsion"
    re.compile(r"^\W*(?:so|therefore|thus|hence)\b[,\s]*(?:maybe|probably|perhaps|I think|I'd say|it's|it is)?[,\s]*\(?" + LETTER + r"\)?\s*(?:[.:,]|$)", re.I),
]
NEGATION = re.compile(r"\b(?:not|isn't|wouldn't|can't|cannot|unlikely|less likely|incorrect|wrong|rule[sd]? out|rather than|instead of|if)\b|n't\b", re.I)


def verdict(sentence: str) -> int:
    """0 if the sentence states no answer letter, 1 for A, 2 for B."""
    s = sentence.strip()
    if not s or s.endswith("?"):
        return 0
    for pat in VERDICT:
        m = pat.search(s)
        if m and not NEGATION.search(s):
            return 1 if m.group(1) == "A" else 2
    return 0


SENTENCE_KEYS = ("X", "logp_A", "logp_B", "z_A", "z_B", "lse_other", "max_other", "lse", "mean", "sd", "z_think", "logp_think")
START_KEYS = ("X0", "z0_A", "z0_B", "lse0_other", "lse0", "mean0")


def read_dir(d: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Per-trace readout records by sample_id and start records by prompt_id."""
    by_trace, by_prompt = {}, {}
    for f in sorted(d.glob("readouts-*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                by_trace[r["sample_id"]] = r
    start = d / "readouts_start.jsonl"
    if start.exists():
        for line in start.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            by_prompt[r["prompt_id"]] = r
    return by_trace, by_prompt


def floats(values: list) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=np.float32)


def others_level(ids: list[int], logits: list[float], letters: set[int]) -> tuple[float, float]:
    """Log-sum-exp and maximum of the stored candidates that are not the two letters."""
    vals = [z for i, z in zip(ids, logits) if i not in letters]
    if not vals:
        return np.nan, np.nan
    m = max(vals)
    return m + float(np.log(np.sum(np.exp(np.array(vals) - m)))), m


def sentence_values(r: dict, z: dict | None, letters: set[int]) -> dict[str, np.ndarray]:
    """One array per SENTENCE_KEYS from a readout record: the current format (closed and open blocks) or the
    2026-10 format (readouts/ with X and log p, readouts_logits/ with z_A, z_B, lse_other, max_other)."""
    n = len(r["pos"])
    nan = np.full(n, np.nan, dtype=np.float32)
    if "closed" in r:
        c, o = r["closed"], r["open"]
        oth = [others_level(ids, zs, letters) for ids, zs in zip(c["top_ids"], c["top_logits"])]
        return {"X": floats(c["z_A"]) - floats(c["z_B"]), "logp_A": floats(c["logp_A"]), "logp_B": floats(c["logp_B"]),
                "z_A": floats(c["z_A"]), "z_B": floats(c["z_B"]), "lse_other": floats([x[0] for x in oth]), "max_other": floats([x[1] for x in oth]),
                "lse": floats(c["lse"]), "mean": floats(c["mean"]), "sd": floats(c["sd"]),
                "z_think": floats(o["z_think"]), "logp_think": floats(o["logp_think"])}
    v = {"X": floats(r["X"]), "logp_A": floats(r["logp_A"]), "logp_B": floats(r["logp_B"])}
    for key in ("z_A", "z_B", "lse_other", "max_other"):
        v[key] = floats(z[key]) if z else nan
    for key in ("lse", "mean", "sd", "z_think", "logp_think"):
        v[key] = nan
    return v


def start_values(s: dict | None, sz: dict | None, letters: set[int]) -> dict[str, float | None]:
    """The prompt's readout with empty thinking, either format."""
    if s is None:
        return {k: None for k in START_KEYS}
    if "lse" in s:   # current format
        oth = others_level(s["top_ids"], s["top_logits"], letters)[0]
        return {"X0": s["z_A"] - s["z_B"], "z0_A": s["z_A"], "z0_B": s["z_B"], "lse0_other": oth, "lse0": s["lse"], "mean0": s["mean"]}
    sz = sz or {}
    return {"X0": s.get("X"), "z0_A": sz.get("z_A"), "z0_B": sz.get("z_B"), "lse0_other": sz.get("lse_other"), "lse0": None, "mean0": None}


def build(run: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    texts = [tok.decode([i]) for i in range(len(tok))]
    letters = {tok.encode(x, add_special_tokens=False)[0] for x in ("A", "B")}
    readouts, start = read_dir(run / "readouts")
    logits, start_logits = read_dir(run / "readouts_logits")
    rows, ts, vs, off = [], [], [], 0
    cols: dict[str, list[np.ndarray]] = {key: [] for key in SENTENCE_KEYS}
    for t in iter_traces(run):
        r = readouts.get(t["sample_id"])
        if r is None:
            continue
        gen, pos = t["token_ids"], r["pos"]
        starts = [r["think_start"]] + pos[:-1]
        v = [verdict("".join(texts[g] for g in gen[a:b])) for a, b in zip(starts, pos)]
        answer, _ = parse_answer(t["answer_text"], ("A", "B"))
        z = logits.get(t["sample_id"])
        if z is not None and z["pos"] != pos:
            raise ValueError(f"logit readout positions differ for {t['sample_id']}")
        rows.append({
            "sample_id": t["sample_id"], "prompt_id": t["prompt_id"], "item_id": t["item_id"], "pair_id": t["pair_id"],
            "source_id": t["source_id"], "order": t["order"], "cond": COND.get(t["condition"], t["condition"]),
            "label": t["label"], "answer": answer, "correct": (answer == t["label"]) if answer else None,
            "think_tokens": t["think_tokens"], "n": len(pos), "off": off,
            **start_values(start.get(t["prompt_id"]), start_logits.get(t["prompt_id"]), letters),
        })
        for key, arr in sentence_values(r, z, letters).items():
            cols[key].append(arr)
        ts.append(np.asarray(pos, dtype=np.int32) - r["think_start"])
        vs.append(np.asarray(v, dtype=np.int8))
        off += len(pos)
    arrays = {"t": np.concatenate(ts), "verdict": np.concatenate(vs), **{key: np.concatenate(v) for key, v in cols.items()}}
    return pd.DataFrame(rows), arrays


def load_paths(run: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Trace table and flat per-sentence arrays; sentences of trace i are [off, off + n)."""
    out = run / "analysis"
    return pd.read_parquet(out / "paths_traces.parquet"), dict(np.load(out / "paths.npz"))


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    traces, arr = build(args.run)
    out = args.run / "analysis"
    out.mkdir(exist_ok=True)
    traces.to_parquet(out / "paths_traces.parquet", index=False)
    np.savez_compressed(out / "paths.npz", **arr)
    last = traces.off.values + traces.n.values - 1
    has = np.add.reduceat((arr["verdict"] > 0).astype(int), traces.off.values) > 0
    print(f"{len(traces)} traces, {len(arr['t'])} sentences ({traces.n.median():.0f} per trace, median)")
    print(f"verdict sentences: {(arr['verdict'] > 0).mean():.3f} of all sentences; traces with at least one {has.mean():.3f}; "
          f"last sentence is a verdict {(arr['verdict'][last] > 0).mean():.3f}")
    print(f"-> {out}/paths.npz, paths_traces.parquet")


if __name__ == "__main__":
    main()
