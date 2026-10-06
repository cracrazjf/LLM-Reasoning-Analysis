"""Sentence readouts of the stored traces: after every sentence of its thinking, what would the model
answer, and would it stop?

    python src/readout.py --run runs/medxpertqa/hurts-qwen3-8b --selection data/selections/medxpertqa_hurts.json

Nothing is sampled: every readout is a forward pass over stored token ids, so it describes exactly
the traces of the run and can be redone or extended at any time.

Positions (p = number of generated tokens kept): every sentence start inside the thinking (a
sentence ends at a token that contains a line break, or that ends in . ! ? with the next token
starting with whitespace; the position is the next sentence's first non-blank token) and the stop
(where the model itself wrote </think>). Each position is read in two contexts:
  closed  prompt + thinking so far + "\\n</think>\\n\\n", the way the model closes its thinking:
          the forced answer. z_A, z_B (raw logits), logp_A, logp_B.
  open    prompt + thinking so far, nothing appended: the model's own next token. z_think and
          logp_think for </think> (would it stop here?), plus z_A, z_B.
Both contexts also record the top candidates (token ids and logits) and, over the whole
vocabulary, the mean, standard deviation and log-sum-exp of the logits. A logit has no fixed
zero; the log-sum-exp turns it into a probability (logp = z - lse) and the mean is a reference.
The full-vocabulary values come from a logits processor that reads every row before vLLM keeps
only the top candidates; with --no-vocab-stats they are skipped, logp is computed from the top
candidates (the two letters hold almost all the probability) and z_think is known only when
</think> is among them.

Also read once per prompt: the closed context on Qwen3's empty thinking block
"<think>\\n\\n</think>\\n\\n" (readouts_start.jsonl), the answer before any reasoning.

Output in <run>/readouts/: readouts-NN.jsonl, one line per trace with pos, kind ("s" sentence,
"e" stop) and the two blocks "closed" and "open", each a dict of lists aligned with pos. An
interrupted run resumes from the traces already written; --sample-ids FILE (one sample_id per
line) restricts the readouts to those traces.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from generate import ANSWER_TOP_LOGPROBS, MODEL, Tokens, chat_ids, load_llm, load_selection  # noqa: E402

SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s*$")
CLOSE = "\n</think>\n\n"
EMPTY_THINK = "<think>\n\n</think>\n\n"
CONTEXTS = ("closed", "open")


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


def positions(gen: list[int], start: int, end: int, texts: list[str]) -> list[tuple[int, str]]:
    return [(p, "s") for p in sentence_boundaries(gen, start, end, texts)] + [(end, "e")]


def done_ids(out: Path) -> set[str]:
    done = set()
    for f in sorted(out.glob("readouts-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def vocab_stats_processor(watch_ids: list[int]):
    """A vLLM logits processor that records, for every tagged request, the full-vocabulary mean, SD and
    log-sum-exp of the logits and the logits of the watched token ids. It runs inside the engine, so the
    engine must run in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0); results are read from
    VocabStats.instance.out, keyed by the tag in SamplingParams.extra_args."""
    import torch
    from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor, MoveDirectionality

    class VocabStats(LogitsProcessor):
        instance = None

        def __init__(self, vllm_config, device, is_pin_memory) -> None:
            self.rows: dict[int, str] = {}      # batch row -> tag
            self.out: dict[str, dict[str, Any]] = {}
            self.ids = torch.tensor(watch_ids, device=device)
            VocabStats.instance = self

        def is_argmax_invariant(self) -> bool:
            return True

        def update_state(self, batch_update: BatchUpdate | None) -> None:
            if batch_update is None:
                return
            for index in batch_update.removed:
                self.rows.pop(index, None)
            for a, b, direction in batch_update.moved:
                ra, rb = self.rows.pop(a, None), self.rows.pop(b, None)
                if ra is not None:
                    self.rows[b] = ra
                if direction == MoveDirectionality.SWAP and rb is not None:
                    self.rows[a] = rb
            for index, params, _prompt, _output in batch_update.added:
                tag = (getattr(params, "extra_args", None) or {}).get("tag")
                if tag is None:
                    self.rows.pop(index, None)
                else:
                    self.rows[index] = tag

        def apply(self, logits):
            if self.rows:
                idx = sorted(self.rows)
                sub = logits[torch.tensor(idx, device=logits.device)].float()
                mean, sd, lse = sub.mean(1).tolist(), sub.std(1).tolist(), torch.logsumexp(sub, 1).tolist()
                picked = sub[:, self.ids].tolist()
                for j, i in enumerate(idx):
                    self.out[self.rows[i]] = {"mean": mean[j], "sd": sd[j], "lse": lse[j], "watch": picked[j]}
            return logits

    return VocabStats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="run directory with traces-*.jsonl")
    ap.add_argument("--selection", required=True, type=Path, help="the selection file the run was generated from")
    ap.add_argument("--sample-ids", type=Path, help="read out only the traces listed in this file")
    ap.add_argument("--no-vocab-stats", action="store_true", help="skip the full-vocabulary values (no logits processor)")
    ap.add_argument("--top", type=int, default=ANSWER_TOP_LOGPROBS, help="candidates stored per readout")
    ap.add_argument("--limit", type=int, default=0, help="only the first N traces (smoke test)")
    ap.add_argument("--max-traces", type=int, default=128,
                    help="traces being read out at once; their prefixes must fit the KV cache (380k tokens on an A100 80GB)")
    ap.add_argument("--max-num-seqs", type=int, default=1024)
    ap.add_argument("--max-num-batched-tokens", type=int, default=32768)
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()
    stats_on = not args.no_vocab_stats
    if stats_on:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # the logits processor must live in this process

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    out = args.run / "readouts"
    out.mkdir(exist_ok=True)
    selection = load_selection(args.selection)
    done = done_ids(out)
    only = set(args.sample_ids.read_text(encoding="utf-8").splitlines()) if args.sample_ids else None

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokens = Tokens(tok, tuple(selection["prompts"][0]["options"]))
    watch = [tokens.options["A"], tokens.options["B"], tokens.think_end]
    processor = vocab_stats_processor(watch) if stats_on else None
    llm, _ = load_llm(args.max_num_seqs, max_logprobs=args.top, logprobs_mode="raw_logits",
                      max_num_batched_tokens=args.max_num_batched_tokens,
                      **({"logits_processors": [processor]} if processor else {}))
    engine = llm.llm_engine
    stats = processor.instance if processor else None
    if stats_on and stats is None:
        raise SystemExit("the logits processor was not instantiated in this process; rerun with --no-vocab-stats")
    texts = [tok.decode([i]) for i in range(len(tok))]
    close = tok(CLOSE, add_special_tokens=False)["input_ids"]
    empty_think = tok(EMPTY_THINK, add_special_tokens=False)["input_ids"]
    prompt_ids = {p["prompt_id"]: chat_ids(tok, p["messages"]) for p in selection["prompts"]}

    def params(rid: str) -> SamplingParams:
        kw = {"extra_args": {"tag": rid}} if stats_on else {}
        return SamplingParams(max_tokens=1, temperature=0.0, logprobs=args.top, output_kind=RequestOutputKind.FINAL_ONLY, **kw)

    def read(rid: str, o: Any) -> dict[str, Any]:
        """One readout: the watched logits, their log-probabilities, the top candidates and the vocabulary values."""
        top = {k: v.logprob for k, v in o.outputs[0].logprobs[0].items()}   # raw logits of the top candidates
        order = sorted(top, key=top.get, reverse=True)[: args.top]
        s = stats.out.pop(rid, None) if stats_on else None
        if s is not None:
            lse, z = s["lse"], dict(zip(watch, s["watch"]))
            rec = {"mean": s["mean"], "sd": s["sd"], "lse": lse}
        else:
            m = max(top.values())
            lse = m + math.log(sum(math.exp(v - m) for v in top.values()))
            z = {t: top.get(t) for t in watch}
            rec = {"mean": None, "sd": None, "lse": lse}
        for name, t in (("A", tokens.options["A"]), ("B", tokens.options["B"]), ("think", tokens.think_end)):
            rec[f"z_{name}"] = z[t]
            rec[f"logp_{name}"] = None if z[t] is None else z[t] - lse
        rec["top_ids"], rec["top_logits"] = order, [top[t] for t in order]
        return rec

    def rounded(rec: dict[str, Any]) -> dict[str, Any]:
        def r(v: Any) -> Any:
            return None if v is None else round(v, 4)
        return {k: ([r(x) for x in v] if k == "top_logits" else v) if isinstance(v, list) else r(v) for k, v in rec.items()}

    def run(requests: dict[str, list[int]]) -> dict[str, Any]:
        for rid, ids in requests.items():
            engine.add_request(rid, {"prompt_token_ids": ids}, params(rid))
        got = {}
        while engine.has_unfinished_requests():
            for o in engine.step():
                if o.finished:
                    got[o.request_id] = read(o.request_id, o)
        return got

    start_path = out / "readouts_start.jsonl"
    if not start_path.exists():
        got = run({pid: ids + empty_think for pid, ids in prompt_ids.items()})
        start_path.write_text("".join(json.dumps({"prompt_id": pid, **rounded(got[pid])}) + "\n" for pid in prompt_ids))
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
                           "think_start": start, "think_end": end, "pos": positions(gen, start, end, texts)})
    if args.limit:
        traces = traces[: args.limit]
    total, n_read = len(traces), sum(2 * len(t["pos"]) for t in traces)
    print(f"{len(done)} traces already read out; {total} to do, {n_read} readouts", flush=True)

    writer = (out / f"readouts-{len(list(out.glob('readouts-*.jsonl'))) + 1:02d}.jsonl").open("a", encoding="utf-8")
    queue = list(reversed(traces))
    active: dict[str, dict[str, Any]] = {}
    written, n_done, t0 = 0, 0, time.time()

    def request(t: dict[str, Any], p: int, ctx: str) -> None:
        ids = prompt_ids[t["prompt_id"]] + t["gen"][:p] + (close if ctx == "closed" else [])
        rid = f"{t['sample_id']}@{p}@{ctx}"
        engine.add_request(rid, {"prompt_token_ids": ids}, params(rid))

    while queue or active:
        while queue and len(active) < args.max_traces:
            t = queue.pop()
            t["got"] = {}
            active[t["sample_id"]] = t
            # The longest prefix goes first; once it has run, the shorter ones reuse its cached blocks.
            request(t, t["think_end"], "closed")
        for o in engine.step():
            if not o.finished:
                continue
            sid, p, ctx = o.request_id.rsplit("@", 2)
            t = active[sid]
            t["got"][(int(p), ctx)] = read(o.request_id, o)
            n_done += 1
            if len(t["got"]) == 1:
                for q, _ in t["pos"]:
                    for c in CONTEXTS:
                        if (q, c) != (t["think_end"], "closed"):
                            request(t, q, c)
            if len(t["got"]) == 2 * len(t["pos"]):
                blocks = {}
                for c in CONTEXTS:
                    rows = [rounded(t["got"][(q, c)]) for q, _ in t["pos"]]
                    blocks[c] = {k: [r[k] for r in rows] for k in rows[0]}
                writer.write(json.dumps({
                    "sample_id": sid, "prompt_id": t["prompt_id"], "think_start": t["think_start"], "think_end": t["think_end"],
                    "pos": [q for q, _ in t["pos"]], "kind": [k for _, k in t["pos"]], **blocks,
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
