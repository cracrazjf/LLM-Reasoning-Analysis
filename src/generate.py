"""Sample Qwen3-8B reasoning on every prompt of a selection file and store the traces.

    python src/generate.py --selection data/selections/medxpertqa_hurts.json \
        --out runs/medxpertqa/hurts-qwen3-8b --samples 100

The selection (src/medxpertqa_dataset.py select) lists prompts with their chat
messages, options and label plus the task fields (pair_id, order, source_id, ...);
every field except the messages is copied into each trace record, so the traces
describe themselves.

Sampling: Qwen3-8B in thinking mode with the model card's thinking-mode settings
and no truncation; --samples samples per prompt, one fixed seed per (selection
seed, prompt, sample index). An interrupted run resumes from the traces already
stored; a line cut off by a crash is re-sampled.

Stored per trace (one JSON line): the selection fields, generated token ids and
text, the raw log-probability of every sampled token, thinking length (tokens
between <think> and </think>), the final answer and whether it is correct, and at
the answer position (after the model's own "</think>\n\n") logp_A, logp_B and
X = logp_A - logp_B from the raw next-token distribution. Everything else (raw
logits, vocabulary statistics, answers forced at truncation points) is a readout
over the stored token ids: src/readout.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
# Qwen3 model card, thinking mode: temperature 0.6, top-p 0.95, top-k 20, min-p 0; outputs up to 32,768 tokens.
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_tokens": 32768}
MAX_MODEL_LEN = 40960
ANSWER_TOP_LOGPROBS = 20  # next-token candidates read at the answer position


def sample_seed(select_seed: str, prompt_id: str, k: int) -> int:
    return int(hashlib.sha256(f"{select_seed}:{prompt_id}:{k}".encode()).hexdigest()[:8], 16)


def done_ids(out: Path) -> set[str]:
    """sample_ids already written; a line cut off by a crash is ignored and re-sampled."""
    done = set()
    for f in sorted(out.glob("traces-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def load_selection(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_traces(run: Path, fields: tuple[str, ...] | None = None):
    """Trace records of a run; `fields` keeps memory small on large runs."""
    for f in sorted(run.glob("traces-*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # a line cut off by a crash
                    continue
                yield {k: r.get(k) for k in fields} if fields else r


def parse_answer(text: str | None, options) -> tuple[str | None, str | None]:
    """The answer letter in the text after </think>, and how it was written.

    "letter": the letter alone, optionally with a period ("A", "B.");
    "letter+text": the letter first, then more words ("A. Bedbug bite", "A\n\n**Reasoning:** ...").
    The answer-position readout X is taken at the first token either way.
    """
    if not text or not text.strip():
        return None, None
    t = text.strip()
    head = t.split()[0].strip("*()[].:")
    if head not in options:
        return None, "other"
    return head, "letter" if t.rstrip(".") == head else "letter+text"


class Tokens:
    """Special and answer-letter token ids of the model's tokenizer."""

    def __init__(self, tok: Any, letters: tuple[str, ...] = ("A", "B")) -> None:
        self.tok = tok
        self.think = tok.convert_tokens_to_ids("<think>")
        self.think_end = tok.convert_tokens_to_ids("</think>")
        self.options = {}
        for letter in letters:
            ids = tok.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{letter!r} is not a single token: {ids}")
            self.options[letter] = ids[0]

    def parse(self, gen: list[int]) -> dict[str, Any]:
        """Thinking length, answer text and the position of the answer's first token."""
        if self.think_end not in gen:
            return {"think_tokens": None, "answer_text": None, "answer_pos": None, "answer": None, "answer_format": None}
        end = gen.index(self.think_end)
        start = gen.index(self.think) + 1 if self.think in gen[:end] else 0
        pos = next((i for i in range(end + 1, len(gen)) if self.tok.decode([gen[i]]).strip()), None)
        text = self.tok.decode(gen[end + 1:], skip_special_tokens=True).strip()
        answer, fmt = parse_answer(text, self.options)
        return {"think_tokens": end - start, "answer_text": text, "answer_pos": pos, "answer": answer, "answer_format": fmt}


def answer_x(top: dict[int, Any], tokens: Tokens) -> dict[str, Any]:
    """X = log p(A) - log p(B) at the answer position.

    If one letter is outside the returned candidates, X is a bound: its log
    probability is at most that of the least likely returned candidate.
    """
    lp = {k: v.logprob for k, v in top.items()}  # greedy readout: every entry is in the top candidates
    la, lb = lp.get(tokens.options["A"]), lp.get(tokens.options["B"])
    floor = min(lp.values())
    if la is not None and lb is not None:
        return {"logp_A": la, "logp_B": lb, "X": la - lb, "X_bound": None}
    if la is None and lb is None:
        return {"logp_A": None, "logp_B": None, "X": None, "X_bound": None}
    x = (la - floor) if la is not None else (floor - lb)
    return {"logp_A": la, "logp_B": lb, "X": x, "X_bound": "lower" if la is not None else "upper"}


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
            return False   # vLLM runs argmax-invariant processors only for random sampling; the readouts are greedy

        def update_state(self, batch_update: BatchUpdate | None) -> None:
            if batch_update is None:
                return
            # vLLM's own order: added, then removed, then moved
            for index, params, _prompt, _output in batch_update.added:
                tag = (getattr(params, "extra_args", None) or {}).get("tag")
                if tag is None:
                    self.rows.pop(index, None)
                else:
                    self.rows[index] = tag
            for index in batch_update.removed:
                self.rows.pop(index, None)
            for a, b, direction in batch_update.moved:
                ra, rb = self.rows.pop(a, None), self.rows.pop(b, None)
                if ra is not None:
                    self.rows[b] = ra
                if direction == MoveDirectionality.SWAP and rb is not None:
                    self.rows[a] = rb

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


def load_llm(max_num_seqs: int, max_logprobs: int = ANSWER_TOP_LOGPROBS, logprobs_mode: str = "raw_logprobs", **kwargs: Any):
    from huggingface_hub import snapshot_download
    from vllm import LLM

    model_path = snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True)
    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=0, max_num_seqs=max_num_seqs,
              enable_prefix_caching=True, logprobs_mode=logprobs_mode, max_logprobs=max_logprobs, **kwargs)
    return llm, model_path


def chat_ids(tok: Any, messages: list[dict[str, str]]) -> list[int]:
    text = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True, tokenize=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="run directory; an existing one is resumed")
    ap.add_argument("--samples", type=int, required=True, help="samples per prompt")
    ap.add_argument("--prompts", type=int, default=0, help="use only the first N prompts (smoke test)")
    ap.add_argument("--max-num-seqs", type=int, default=256, help="vLLM batch size")
    ap.add_argument("--max-inflight", type=int, default=384, help="traces queued in the engine at once")
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    import torch
    import vllm
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    selection = load_selection(args.selection)
    select_seed = selection["select_seed"]
    prompts = selection["prompts"][: args.prompts] if args.prompts else selection["prompts"]
    letters = tuple(prompts[0]["options"])
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    done = done_ids(out)

    llm, model_path = load_llm(args.max_num_seqs)
    engine = llm.llm_engine
    FINAL = RequestOutputKind.FINAL_ONLY  # step() returns a request only when it has finished
    tok = llm.get_tokenizer()
    tokens = Tokens(tok, letters)
    prompt_ids = {p["prompt_id"]: chat_ids(tok, p["messages"]) for p in prompts}
    by_id = {p["prompt_id"]: p for p in prompts}

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "model": MODEL, "sampling": SAMPLING, "samples_per_prompt": args.samples,
        "selection": str(args.selection), "select_seed": select_seed,
        "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "n_prompts": len(prompts), "sessions": []}
    session = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "vllm": vllm.__version__,
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
               "model_revision": Path(model_path).name, "already_done": len(done)}
    manifest["sessions"].append(session)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

    # Sample-major order: an interrupted run still covers every prompt evenly.
    queue = [(p["prompt_id"], k) for k in range(args.samples) for p in prompts
             if f"{p['prompt_id']}#{k}" not in done]
    queue.reverse()  # pop() from the end
    total = len(queue)
    print(f"{len(done)} traces already stored, {total} to sample", flush=True)

    shard = out / f"traces-{len(manifest['sessions']):02d}.jsonl"
    writer = shard.open("a", encoding="utf-8")
    pending: dict[str, dict[str, Any]] = {}   # traces waiting for their answer-position readout
    inflight = 0
    written, gen_tokens, t0 = 0, 0, time.time()

    def add_sample() -> None:
        nonlocal inflight
        pid, k = queue.pop()
        params = SamplingParams(**SAMPLING, seed=sample_seed(select_seed, pid, k), logprobs=0, output_kind=FINAL)
        engine.add_request(f"gen|{pid}#{k}", {"prompt_token_ids": prompt_ids[pid]}, params)
        inflight += 1

    def write(rec: dict[str, Any]) -> None:
        nonlocal written
        writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
        writer.flush()
        written += 1
        if written % args.log_every == 0 or written == total:
            dt = time.time() - t0
            print(f"  {written}/{total} traces, {gen_tokens / dt:.0f} generated tok/s, "
                  f"eta {(total - written) * dt / written / 60:.0f} min", flush=True)

    while queue or engine.has_unfinished_requests():
        while queue and inflight < args.max_inflight:
            add_sample()
        for o in engine.step():
            if not o.finished:
                continue
            kind, sample_id = o.request_id.split("|", 1)
            if kind == "x":  # answer-position readout for a finished trace
                rec = pending.pop(sample_id)
                rec.update(answer_x(o.outputs[0].logprobs[0], tokens))
                write(rec)
                continue
            inflight -= 1
            pid, k = sample_id.split("#")
            p, c = by_id[pid], o.outputs[0]
            gen = list(c.token_ids)
            gen_tokens += len(gen)
            parsed = tokens.parse(gen)
            rec = {"sample_id": sample_id}
            rec.update({key: v for key, v in p.items() if key != "messages"})
            rec.update({
                "sample": int(k), "seed": sample_seed(select_seed, pid, int(k)),
                "finish_reason": c.finish_reason, "n_tokens": len(gen), **parsed,
                "correct": parsed["answer"] == p["label"] if parsed["answer"] else None,
                "token_ids": gen,
                "token_logprobs": [round(lp[t].logprob, 4) for t, lp in zip(gen, c.logprobs)],
                "text": c.text,
            })
            if parsed["answer_pos"] is None:
                rec.update({"logp_A": None, "logp_B": None, "X": None, "X_bound": None})
                write(rec)
                continue
            # The trace's KV blocks are still cached, so this readout is almost free.
            pending[sample_id] = rec
            engine.add_request(f"x|{sample_id}",
                               {"prompt_token_ids": prompt_ids[pid] + gen[: parsed["answer_pos"]]},
                               SamplingParams(max_tokens=1, temperature=0.0, logprobs=ANSWER_TOP_LOGPROBS, output_kind=FINAL))
    writer.close()
    session["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    session["written"] = written
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"done: {written} traces in {(time.time() - t0) / 60:.1f} min -> {shard}", flush=True)


if __name__ == "__main__":
    main()
