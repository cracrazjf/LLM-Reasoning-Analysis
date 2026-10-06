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
  per sentence (flat, traces concatenated): t = thinking tokens up to the end of the sentence,
      X, logp_A, logp_B (next-token log-probabilities of the two letters), verdict (0 none,
      1 for A, 2 for B)
  per trace: ids, condition, label, answer, correct, think_tokens, n (sentences), off (offset of
      its first sentence in the flat arrays), X0 (the prompt's readout with empty thinking)
If <run>/readouts_logits/ exists (readout.py --logits), the raw logits are added: per sentence z_A,
z_B, lse_other and max_other (NaN for traces without a logit readout), per trace z0_A, z0_B and
lse0_other from the readout with empty thinking.
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


LOGIT_KEYS = ("z_A", "z_B", "lse_other", "max_other")


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


def build(run: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    texts = [tok.decode([i]) for i in range(len(tok))]
    readouts, start = read_dir(run / "readouts")
    logits, start_logits = read_dir(run / "readouts_logits")
    rows, ts, xs, las, lbs, vs, off = [], [], [], [], [], [], 0
    zs: dict[str, list[np.ndarray]] = {key: [] for key in LOGIT_KEYS}

    def floats(values: list) -> np.ndarray:
        return np.array([np.nan if v is None else v for v in values], dtype=np.float32)

    for t in iter_traces(run):
        r = readouts.get(t["sample_id"])
        if r is None:
            continue
        gen, pos = t["token_ids"], r["pos"]
        starts = [r["think_start"]] + pos[:-1]
        v = [verdict("".join(texts[g] for g in gen[a:b])) for a, b in zip(starts, pos)]
        answer, _ = parse_answer(t["answer_text"], ("A", "B"))
        rows.append({
            "sample_id": t["sample_id"], "prompt_id": t["prompt_id"], "item_id": t["item_id"], "pair_id": t["pair_id"],
            "source_id": t["source_id"], "order": t["order"], "cond": COND.get(t["condition"], t["condition"]),
            "label": t["label"], "answer": answer, "correct": (answer == t["label"]) if answer else None,
            "think_tokens": t["think_tokens"], "n": len(pos), "off": off, "X0": start.get(t["prompt_id"], {}).get("X"),
            **{f"{key}0{sub}": start_logits.get(t["prompt_id"], {}).get(f"{key}{sub}") for key, sub in (("z", "_A"), ("z", "_B"), ("lse", "_other"))},
        })
        z = logits.get(t["sample_id"])
        if z is not None and z["pos"] != pos:
            raise ValueError(f"logit readout positions differ for {t['sample_id']}")
        for key in LOGIT_KEYS:
            zs[key].append(floats(z[key]) if z else np.full(len(pos), np.nan, dtype=np.float32))
        ts.append(np.asarray(pos, dtype=np.int32) - r["think_start"])
        xs.append(floats(r["X"]))
        las.append(floats(r["logp_A"]))
        lbs.append(floats(r["logp_B"]))
        vs.append(np.asarray(v, dtype=np.int8))
        off += len(pos)
    arrays = {"t": np.concatenate(ts), "X": np.concatenate(xs), "logp_A": np.concatenate(las),
              "logp_B": np.concatenate(lbs), "verdict": np.concatenate(vs)}
    if logits:
        arrays.update({key: np.concatenate(v) for key, v in zs.items()})
    return pd.DataFrame(rows), arrays


def load_paths(run: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Trace table and flat per-sentence arrays; sentences of trace i are [off, off + n)."""
    out = run / "analysis"
    return pd.read_parquet(out / "paths_traces.parquet"), dict(np.load(out / "paths.npz"))


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
