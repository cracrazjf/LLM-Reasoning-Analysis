"""Forced-answer readouts on the stored traces of a run.

    python src/readout.py --run runs/medxpertqa/screen-qwen3-8b --selection data/selections/medxpertqa_screen.json

At each readout position the thinking is cut, closed the way the model closes
it ("\\n</think>\\n\\n"), and the next-token distribution gives
X = log p(A) - log p(B). Nothing is sampled: readouts are forward passes over
the stored token ids, so they describe exactly the traces of the run.

Positions (p = number of generated tokens kept):
  start     before any reasoning: the prompt followed by Qwen3's empty-thinking
            block "<think>\\n\\n</think>\\n\\n"; identical for all samples of a
            prompt, so read once per prompt (readouts_start.jsonl)
  sentence  every sentence boundary in the thinking (after the punctuation or
            line break and any whitespace that follows it, i.e. where the next
            sentence starts)
  grid      with --grid N, also every N thinking tokens (most of these fall
            inside a sentence)
  stop      where the model itself wrote </think>; compare with the stored X
            read after the model's own "</think>\\n\\n"

Output in <run>/readouts/: readouts_start.jsonl, and readouts-NN.jsonl with one
line per trace: positions, their kinds ("s" sentence, "g" grid, "gs" both,
"e" stop), X, log p(A), log p(B) and bound flags (see generate.answer_x). An
interrupted run resumes from the traces already written. --sample-ids FILE
(one sample_id per line) restricts the readouts to those traces.

With --logits the same positions are read as raw logits (the scores before the softmax) and
written to <run>/readouts_logits/ instead: z_A, z_B, and for the other stored next-token
candidates their log-sum-exp (lse_other), the largest one (max_other) and its token id. The two
letters hold almost all the probability, so log p(A) = z_A - logsumexp(z_A, z_B, lse_other) up
to the candidates that are not stored. A logit has no fixed zero: only differences between
logits, or changes of one logit against a reference, carry meaning.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from generate import ANSWER_TOP_LOGPROBS, Tokens, answer_x, chat_ids, load_llm, load_selection  # noqa: E402

SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s*$")


def sentence_boundaries(gen: list[int], start: int, end: int, texts: list[str]) -> list[int]:
    """Positions inside the thinking where a new sentence starts."""
    out = []
    i = start
    while i < end:
        t = texts[gen[i]]
        nxt = texts[gen[i + 1]] if i + 1 < end else "\n"
        if "\n" in t or (SENTENCE_END.search(t) and nxt[:1].isspace()):
            j = i + 1
            while j < end and not texts[gen[j]].strip():  # keep trailing whitespace with the sentence
                j += 1
            if start < j < end:
                out.append(j)
            i = j
        else:
            i += 1
    return out


def positions(gen: list[int], start: int, end: int, texts: list[str], grid: int = 0) -> list[tuple[int, str]]:
    kinds: dict[int, str] = {}
    if grid:
        for p in range(start + grid, end, grid):
            kinds[p] = "g"
    for p in sentence_boundaries(gen, start, end, texts):
        kinds[p] = kinds.get(p, "") + "s"
    kinds[end] = "e"
    return sorted(kinds.items())


def answer_logits(top: dict[int, Any], tokens: Tokens) -> dict[str, Any]:
    """Raw logits of the two letters and the level of the other stored candidates."""
    z = {k: v.logprob for k, v in top.items()}  # under logprobs_mode="raw_logits" these values are logits
    a, b = tokens.options["A"], tokens.options["B"]
    other = {k: v for k, v in z.items() if k not in (a, b)}
    best = max(other, key=other.get)
    return {"z_A": z.get(a), "z_B": z.get(b), "max_other": other[best], "max_other_id": best,
            "lse_other": other[best] + math.log(sum(math.exp(v - other[best]) for v in other.values()))}


def done_ids(out: Path) -> set[str]:
    done = set()
    for f in sorted(out.glob("readouts-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="run directory with traces-*.jsonl")
    ap.add_argument("--selection", required=True, type=Path, help="the selection file the run was generated from")
    ap.add_argument("--grid", type=int, default=0, help="also read every N thinking tokens (0: sentence starts only)")
    ap.add_argument("--sample-ids", type=Path, help="read out only the traces listed in this file")
    ap.add_argument("--logits", action="store_true", help="read raw logits into <run>/readouts_logits/")
    ap.add_argument("--limit", type=int, default=0, help="only the first N traces (smoke test)")
    ap.add_argument("--max-traces", type=int, default=128,
                    help="traces being read out at once; their prefixes must fit the KV cache (380k tokens on an A100 80GB)")
    ap.add_argument("--max-num-seqs", type=int, default=1024)
    ap.add_argument("--max-num-batched-tokens", type=int, default=32768)
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()

    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    out = args.run / ("readouts_logits" if args.logits else "readouts")
    out.mkdir(exist_ok=True)
    read = answer_logits if args.logits else answer_x
    selection = load_selection(args.selection)
    done = done_ids(out)
    only = set(args.sample_ids.read_text(encoding="utf-8").splitlines()) if args.sample_ids else None

    llm, _ = load_llm(args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
                      logprobs_mode="raw_logits" if args.logits else "raw_logprobs")
    engine = llm.llm_engine
    tok = llm.get_tokenizer()
    tokens = Tokens(tok, tuple(selection["prompts"][0]["options"]))
    texts = [tok.decode([i]) for i in range(len(tok))]
    close = tok("\n</think>\n\n", add_special_tokens=False)["input_ids"]
    empty_think = tok("<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]
    prompt_ids = {p["prompt_id"]: chat_ids(tok, p["messages"]) for p in selection["prompts"]}
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=ANSWER_TOP_LOGPROBS,
                            output_kind=RequestOutputKind.FINAL_ONLY)

    def run(requests: dict[str, list[int]]) -> dict[str, Any]:
        for rid, ids in requests.items():
            engine.add_request(rid, {"prompt_token_ids": ids}, params)
        got = {}
        while engine.has_unfinished_requests():
            for o in engine.step():
                if o.finished:
                    got[o.request_id] = read(o.outputs[0].logprobs[0], tokens)
        return got

    start_path = out / "readouts_start.jsonl"
    if not start_path.exists():
        got = run({pid: ids + empty_think for pid, ids in prompt_ids.items()})
        start_path.write_text("".join(json.dumps({"prompt_id": pid, **got[pid]}) + "\n" for pid in prompt_ids))
        print(f"start readouts: {len(got)} prompts", flush=True)

    traces = []
    for f in sorted(args.run.glob("traces-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["sample_id"] in done or r["think_tokens"] is None or (only is not None and r["sample_id"] not in only):
                continue
            gen = r["token_ids"]
            end = gen.index(tokens.think_end)
            start = gen.index(tokens.think) + 1
            while start < end and not texts[gen[start]].strip():  # skip the newline after <think>
                start += 1
            traces.append({"sample_id": r["sample_id"], "prompt_id": r["prompt_id"], "gen": gen,
                           "think_start": start, "think_end": end,
                           "pos": positions(gen, start, end, texts, args.grid)})
    if args.limit:
        traces = traces[: args.limit]
    total, n_read = len(traces), sum(len(t["pos"]) for t in traces)
    print(f"{len(done)} traces already read out; {total} to do, {n_read} readouts", flush=True)

    writer = (out / f"readouts-{len(list(out.glob('readouts-*.jsonl'))) + 1:02d}.jsonl").open("a", encoding="utf-8")
    queue = list(reversed(traces))
    active: dict[str, dict[str, Any]] = {}
    written, n_done, t0 = 0, 0, time.time()

    def columns(rows: list[dict[str, Any]]) -> dict[str, list]:
        def col(key: str) -> list:
            return [None if r[key] is None else round(r[key], 4) for r in rows]
        if args.logits:
            return {"z_A": col("z_A"), "z_B": col("z_B"), "lse_other": col("lse_other"), "max_other": col("max_other"),
                    "max_other_id": [r["max_other_id"] for r in rows]}
        return {"X": col("X"), "logp_A": col("logp_A"), "logp_B": col("logp_B"), "bound": [r["X_bound"] for r in rows]}

    def request(t: dict[str, Any], p: int) -> None:
        engine.add_request(f"{t['sample_id']}@{p}",
                           {"prompt_token_ids": prompt_ids[t["prompt_id"]] + t["gen"][:p] + close}, params)

    while queue or active:
        while queue and len(active) < args.max_traces:
            t = queue.pop()
            t["got"] = {}
            active[t["sample_id"]] = t
            # The longest prefix goes first; once it has run, the shorter ones reuse its cached blocks.
            request(t, t["think_end"])
        for o in engine.step():
            if not o.finished:
                continue
            sid, p = o.request_id.rsplit("@", 1)
            t = active[sid]
            t["got"][int(p)] = read(o.outputs[0].logprobs[0], tokens)
            n_done += 1
            if len(t["got"]) == 1:
                for q, _ in t["pos"]:
                    if q != t["think_end"]:
                        request(t, q)
            if len(t["got"]) == len(t["pos"]):
                rows = [t["got"][q] for q, _ in t["pos"]]
                writer.write(json.dumps({
                    "sample_id": sid, "prompt_id": t["prompt_id"],
                    "think_start": t["think_start"], "think_end": t["think_end"],
                    "pos": [q for q, _ in t["pos"]], "kind": [k for _, k in t["pos"]], **columns(rows),
                }) + "\n")
                writer.flush()
                del active[sid]
                written += 1
                if written % args.log_every == 0 or written == total:
                    dt = time.time() - t0
                    print(f"  {written}/{total} traces, {n_done / dt:.0f} readouts/s, "
                          f"eta {(total - written) * dt / written / 60:.0f} min", flush=True)
    writer.close()
    print(f"done: {written} traces, {n_done} readouts in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
