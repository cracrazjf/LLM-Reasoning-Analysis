"""Screening runs of Qwen3-8B on the MedXpertQA two-option pairs.

    python src/generate.py --mode nothink --order ab --out runs/medxpertqa/nothink-qwen3-8b-ab
    python src/generate.py --mode think   --order ab --out runs/medxpertqa/think-qwen3-8b-ab --samples 30

Prompts: every item of data/tasks/medxpertqa.jsonl (981 pairs x 2 option orders = 1,962 prompts),
or with --order only the prompts of that option order (one pod per order; src/screen.py reads
several run directories). One user message = the question with its two options, a blank line, and
"Answer with only A or B. No other words."

--mode nothink: one greedy answer per prompt with thinking switched off (the chat template's
enable_thinking=False puts an empty <think></think> block before the answer).
--mode think: --samples answers per prompt in thinking mode with the model card's settings
(temperature 0.6, top-p 0.95, top-k 20, min-p 0) and up to 32,768 tokens; one fixed seed per
(prompt, sample index). An interrupted run resumes from the answers already stored.

Stored per answer (one JSON line in traces-NN.jsonl): the task fields, the generated token ids
and text, the log probability of every generated token, the thinking length, the answer letter
and whether it is correct, and at the answer position (the first non-blank token after
"</think>\n\n", read greedily from the raw next-token distribution): logp_A, logp_B, p_A, p_B,
X = logp_A - logp_B and gap = logp(correct letter) - logp(wrong letter). A difference of log
probabilities equals the difference of the logits (the softmax normaliser cancels), so gap is
the logit difference used by src/screen.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "data/tasks/medxpertqa.jsonl"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
INSTRUCTION = "Answer with only A or B. No other words."
SAMPLING = {
    # Qwen3 model card, thinking mode
    "think": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_tokens": 32768},
    # greedy; the answer is a letter, the cap only guards against run-on text
    "nothink": {"temperature": 0.0, "max_tokens": 256},
}
MAX_MODEL_LEN = 40960
TOP_LOGPROBS = 20  # next-token candidates read at the answer position
SEED_TAG = "20261007:medxpertqa"


def load_tasks(path: Path = TASKS) -> list[dict[str, Any]]:
    """One prompt per task item, with the task fields every answer record carries."""
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        m = t["meta"]
        items.append({
            "item_id": t["item_id"], "pair_id": m["pair_id"], "order": m["order"], "source_id": m["source_id"],
            "question_type": m["question_type"], "body_system": m["body_system"],
            "label": t["label"], "options": list(t["options"]),
            "correct_text": m["correct"]["text"], "distractor_text": m["distractor"]["text"],
            "source_correct_letter": m["correct"]["letter"], "source_distractor_letter": m["distractor"]["letter"],
            "messages": [{"role": "user", "content": t["question"].rstrip() + "\n\n" + INSTRUCTION}],
        })
    return items


def sample_seed(item_id: str, k: int) -> int:
    return int(hashlib.sha256(f"{SEED_TAG}:{item_id}:{k}".encode()).hexdigest()[:8], 16)


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


def iter_traces(run: Path, fields: tuple[str, ...] | None = None):
    """Answer records of a run; `fields` keeps memory small on large runs."""
    for f in sorted(run.glob("traces-*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # a line cut off by a crash
                    continue
                yield {k: r.get(k) for k in fields} if fields else r


def parse_answer(text: str | None, options) -> tuple[str | None, str | None]:
    """The answer letter in the text after the thinking block, and how it was written.

    "letter": the letter alone, optionally with a period ("A", "B.");
    "letter+text": the letter first, then more words ("A. Bedbug bite", "A\n\n**Reasoning:** ...").
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

    def chat_ids(self, messages: list[dict[str, str]], thinking: bool) -> list[int]:
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking,
                                            tokenize=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def parse(self, gen: list[int], thinking: bool) -> dict[str, Any]:
        """Thinking length, answer text and the position of the answer's first token in `gen`.

        With thinking the answer follows the model's own </think>; without it the empty thinking
        block is part of the prompt and the answer starts at the first non-blank generated token.
        """
        none = {"think_tokens": None, "answer_text": None, "answer_pos": None, "answer": None, "answer_format": None}
        if thinking:
            if self.think_end not in gen:
                return none
            end = gen.index(self.think_end)
            start = gen.index(self.think) + 1 if self.think in gen[:end] else 0
            think_tokens, first = end - start, end + 1
        else:
            think_tokens, first = 0, 0
        pos = next((i for i in range(first, len(gen)) if self.tok.decode([gen[i]]).strip()), None)
        text = self.tok.decode(gen[first:], skip_special_tokens=True).strip()
        answer, fmt = parse_answer(text, self.options)
        return {"think_tokens": think_tokens, "answer_text": text, "answer_pos": pos, "answer": answer,
                "answer_format": fmt}


def readout(top: dict[int, Any], tokens: Tokens, label: str) -> dict[str, Any]:
    """logp, p, X = logp_A - logp_B and gap = logp(correct) - logp(wrong) at the answer position.

    If one letter is outside the returned candidates its log probability is at most that of the
    least likely returned candidate, and X and gap are bounds (X_bound says which side of X).
    """
    lp = {k: v.logprob for k, v in top.items()}
    la, lb = lp.get(tokens.options["A"]), lp.get(tokens.options["B"])
    floor = min(lp.values())
    if la is None and lb is None:
        x, bound = None, None
    else:
        x = (la if la is not None else floor) - (lb if lb is not None else floor)
        bound = None if (la is not None and lb is not None) else ("lower" if la is not None else "upper")
    sign = 1.0 if label == "A" else -1.0
    return {"logp_A": la, "logp_B": lb,
            "p_A": math.exp(la) if la is not None else None, "p_B": math.exp(lb) if lb is not None else None,
            "X": x, "gap": sign * x if x is not None else None, "X_bound": bound}


def load_llm(max_num_seqs: int):
    from huggingface_hub import snapshot_download
    from vllm import LLM

    model_path = snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True)
    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, seed=0, max_num_seqs=max_num_seqs,
              enable_prefix_caching=True, logprobs_mode="raw_logprobs", max_logprobs=TOP_LOGPROBS)
    return llm, model_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["nothink", "think"])
    ap.add_argument("--out", required=True, type=Path, help="run directory; an existing one is resumed")
    ap.add_argument("--samples", type=int, default=None, help="answers per prompt (think: required; nothink: 1)")
    ap.add_argument("--order", choices=["ab", "ba"], default=None, help="only the prompts of this option order")
    ap.add_argument("--pairs", type=Path, default=None, help="text file with one pair_id per line: only these pairs")
    ap.add_argument("--prompts", type=int, default=0, help="use only the first N prompts (smoke test)")
    ap.add_argument("--max-num-seqs", type=int, default=256, help="vLLM batch size")
    ap.add_argument("--max-inflight", type=int, default=384, help="answers queued in the engine at once")
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()
    thinking = args.mode == "think"
    if thinking and not args.samples:
        ap.error("--samples is required with --mode think")
    samples = args.samples if thinking else 1
    sampling = SAMPLING[args.mode]

    import torch
    import vllm
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    items = load_tasks()
    if args.order:
        items = [it for it in items if it["order"] == args.order]
    if args.pairs:
        keep = {line.strip() for line in args.pairs.read_text().splitlines() if line.strip()}
        items = [it for it in items if it["pair_id"] in keep]
        missing = keep - {it["pair_id"] for it in items}
        if missing:
            raise SystemExit(f"pairs not in the tasks: {sorted(missing)}")
    if args.prompts:
        items = items[: args.prompts]
    letters = tuple(items[0]["options"])
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    done = done_ids(out)

    llm, model_path = load_llm(args.max_num_seqs)
    engine = llm.llm_engine
    FINAL = RequestOutputKind.FINAL_ONLY  # step() returns a request only when it has finished
    tokens = Tokens(llm.get_tokenizer(), letters)
    prompt_ids = {it["item_id"]: tokens.chat_ids(it["messages"], thinking) for it in items}
    by_id = {it["item_id"]: it for it in items}

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "model": MODEL, "model_revision": MODEL_REVISION, "mode": args.mode, "order": args.order,
        "pairs": str(args.pairs) if args.pairs else None,
        "sampling": sampling, "samples_per_prompt": samples, "instruction": INSTRUCTION, "seed_tag": SEED_TAG,
        "tasks": str(TASKS.relative_to(ROOT)), "tasks_sha256": hashlib.sha256(TASKS.read_bytes()).hexdigest(),
        "n_prompts": len(items), "sessions": []}
    session = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "vllm": vllm.__version__,
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
               "model_snapshot": Path(model_path).name, "already_done": len(done)}
    manifest["sessions"].append(session)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

    # Sample-major order: an interrupted run still covers every prompt evenly.
    queue = [(it["item_id"], k) for k in range(samples) for it in items if f"{it['item_id']}#{k}" not in done]
    queue.reverse()  # pop() from the end
    total = len(queue)
    print(f"{args.mode}: {len(done)} answers already stored, {total} to generate", flush=True)

    shard = out / f"traces-{len(manifest['sessions']):02d}.jsonl"
    writer = shard.open("a", encoding="utf-8")
    pending: dict[str, dict[str, Any]] = {}   # answers waiting for their answer-position readout
    inflight = 0
    written, gen_tokens, t0 = 0, 0, time.time()

    def add_sample() -> None:
        nonlocal inflight
        iid, k = queue.pop()
        params = SamplingParams(**sampling, seed=sample_seed(iid, k), logprobs=0, output_kind=FINAL)
        engine.add_request(f"gen|{iid}#{k}", {"prompt_token_ids": prompt_ids[iid]}, params)
        inflight += 1

    def write(rec: dict[str, Any]) -> None:
        nonlocal written
        writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
        writer.flush()
        written += 1
        if written % args.log_every == 0 or written == total:
            dt = time.time() - t0
            print(f"  {written}/{total} answers, {gen_tokens / dt:.0f} generated tok/s, "
                  f"eta {(total - written) * dt / written / 60:.0f} min", flush=True)

    while queue or engine.has_unfinished_requests():
        while queue and inflight < args.max_inflight:
            add_sample()
        for o in engine.step():
            if not o.finished:
                continue
            kind, sample_id = o.request_id.split("|", 1)
            if kind == "x":  # answer-position readout of a finished answer
                rec = pending.pop(sample_id)
                rec.update(readout(o.outputs[0].logprobs[0], tokens, rec["label"]))
                write(rec)
                continue
            inflight -= 1
            iid, k = sample_id.split("#")
            it, c = by_id[iid], o.outputs[0]
            gen = list(c.token_ids)
            gen_tokens += len(gen)
            parsed = tokens.parse(gen, thinking)
            rec = {"sample_id": sample_id, "mode": args.mode}
            rec.update({key: v for key, v in it.items() if key != "messages"})
            rec.update({
                "sample": int(k), "seed": sample_seed(iid, int(k)),
                "finish_reason": c.finish_reason, "n_tokens": len(gen), **parsed,
                "correct": parsed["answer"] == it["label"] if parsed["answer"] else None,
                "token_ids": gen,
                "token_logprobs": [round(lp[t].logprob, 4) for t, lp in zip(gen, c.logprobs)],
                "text": c.text,
            })
            if parsed["answer_pos"] is None:
                rec.update({"logp_A": None, "logp_B": None, "p_A": None, "p_B": None, "X": None, "gap": None,
                            "X_bound": None})
                write(rec)
                continue
            # The answer's KV blocks are still cached, so this one-token readout is almost free.
            pending[sample_id] = rec
            engine.add_request(f"x|{sample_id}", {"prompt_token_ids": prompt_ids[iid] + gen[: parsed["answer_pos"]]},
                               SamplingParams(max_tokens=1, temperature=0.0, logprobs=TOP_LOGPROBS, output_kind=FINAL))
    writer.close()
    session["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    session["written"] = written
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"done: {written} answers in {(time.time() - t0) / 60:.1f} min -> {shard}", flush=True)


if __name__ == "__main__":
    main()
