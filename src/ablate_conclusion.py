"""Ablation of the conclusion sentence: read the answer letter with the attention to the conclusion unit's tokens blocked.

The 104-question study, Qwen3-8B and Qwen3-32B (MODEL_NAME selects the model, as in src/generate.py):

    python src/ablate_conclusion.py prepare104                    # local: mask104_job.jsonl in each thinking run (earlier verdict sentences)
    MODEL_NAME=qwen3-32b python src/ablate_conclusion.py run104 --run runs/medxpertqa/study104-qwen3-32b-ab   # pod, one GPU
    python src/ablate_conclusion.py run104 --run runs/medxpertqa/nothink-qwen3-8b-ab                           # the no-thinking answers
    python src/ablate_conclusion.py run104 --run ... --dry --limit 3   # no model: the blocked tokens of every variant, as text
    python src/ablate_conclusion.py analyze104                         # local: tables of both models (no figures)

run104 blocks attention edges (attention knockout, Geva et al. 2023): the rows after the thinking (</think> and the token that
writes the letter; without thinking the template's "</think>\\n\\n") may not attend to a set of keys, in all layers or in a third of
them, and the letter logits are read at the last row. Everything before </think> is computed once and cached; every variant
reruns only the rows after it (their attention is the only thing a variant changes), and the cache is cut back after each.
Variants with thinking (MASK104; the conclusion unit from <run>/readouts/conclusion-*.jsonl, earlier verdicts from the job file):
    base                    nothing blocked (must reproduce the trace's stored logp_A / logp_B)
    conclusion              the conclusion unit [onset, think_end)
    conclusion_answer_row   the same, blocked for the letter-writing row only (the row of attention_at_answer.py run104)
    conclusion_end          the last token of the thinking only (the period that ends the conclusion)
    conclusion_letter       the tokens of the conclusion unit that are the answer's letter
    rand_for_conclusion     control: earlier non-verdict sentences in a seeded random order, added until they hold as many tokens
    earlier_verdicts        the sentences before the conclusion unit that name a conclusion (stopping_rule.sentence_type verdict_definitive /
                            verdict_hedged / option_text)
    rand_for_verdicts       control: earlier non-verdict sentences, as many tokens as the earlier verdicts
    thinking                all thinking before the conclusion unit
    question                the case text, both option lines and the instruction
    options                 the two option lines
    only_conclusion         question + thinking: the conclusion unit is the only text left
    conclusion_early / _mid / _late   the conclusion blocked only in the first / middle / last third of the layers
Without thinking (MASK104_NOTHINK): base, question, stem, options, instruction. The first token is never blocked.
Output: <run>/readouts/mask104-01.jsonl, one line per trace (resumes by sample_id): positions, and per variant the blocked token count,
z_A, z_B, logp_A, logp_B, p_letter = p(A) + p(B), top1, defined (False when there is nothing to block, e.g. no earlier verdict).
analyze104 writes data/screen/study104_mask_traces.csv (one row per trace x variant: y = log p(the trace's answer) - log p(the other
letter), dy = y - y at base, flip, p_letter) and study104_mask_summary.csv (per model x mode x variant: per-question medians, flip
rates, within-question Spearman of dy with the thinking length and with the step-1 attention share of the conclusion, wrong minus
right, dy of the conclusion / the conclusion's jump in the closed readout).

The twelve-question main run of Qwen3-8B:

    python src/ablate_conclusion.py run --run runs/medxpertqa/main-qwen3-8b-ab          # pod (HF transformers, one GPU)
    python src/ablate_conclusion.py run --run ... --dry --limit 3                       # no model: print the blocked token ranges
    python src/ablate_conclusion.py analyze                                             # local: tables and fig60 from both orders
    python src/ablate_conclusion.py classes --run ...   # pod: hide whole sentence classes of the earlier thinking (content-matched blocks)
    python src/ablate_conclusion.py stop --run ...      # pod: the stop decision: P(</think>) at the last thinking token with parts hidden from it
    python src/ablate_conclusion.py heads --run ...     # pod: hide the conclusion in selected heads only (data/screen/medxpertqa_top_heads.json)

classes  Variants (queries at or after think_end, all layers): base; hide_restate / hide_inference / hide_reconsider_doubt / hide_other
         = all earlier sentences of that kind (data/screen/medxpertqa_moves.csv move labels); hide_random_matched = random earlier
         sentences up to the restate block's token count (seeded). Output readouts/ablation_classes-01.jsonl with the block token counts.
stop     Sequence cut at think_end (the last thinking token is the last input); variants hidden from that last row only: base;
         hide_conclusion_rest (the conclusion unit's tokens before the last one); hide_prev; hide_thinking (all before the conclusion);
         hide_stem. Reads log p(</think>), the top token and the letter logits at the last row. Output readouts/ablation_stop-01.jsonl.
heads    Conclusion hidden (queries at or after think_end) only in a set of heads: late_top32 / late_top128 (the heads of layers 12-35
         with the largest mean conclusion share at the answer-writing position), late_rest (layers 12-35 minus the top 128),
         early_top32 (layers 0-11), all (every head; = concl_all). Output readouts/ablation_heads-01.jsonl.

For every trace the sequence prompt + generated tokens up to the answer letter goes through Qwen3-8B as a batch of 9 copies, each
with its own attention mask (causal, minus a block of keys for a set of query rows, in a set of layers). The letter logits at the
last position (the token before the letter) are read for each copy:
    base          nothing blocked (must reproduce the trace's stored logp_A / logp_B)
    concl_all     the conclusion unit's tokens [onset, think_end) blocked for every query at or after think_end (</think> and later), all layers
    concl_answer  the same block, but only for the last query row (the token that writes the letter)
    concl_early / concl_mid / concl_late   the concl_all block applied only in layers 0-11 / 12-23 / 24-35
    prev_all      control: the sentence before the conclusion unit blocked (same queries, all layers)
    rand_all      control: one earlier sentence (seeded by the sample id) blocked
    think_all     the complement: all thinking before the conclusion unit blocked (the conclusion is read without competition)
The conclusion unit's onset comes from data/screen/medxpertqa_last_sentence_main.csv (pushed by sync.sh); without it the last
sentence. The first token (the attention sink) is never blocked, so every row keeps at least one key.

Output: <run>/readouts/ablation-01.jsonl, one line per trace: sample_id, item_id, positions, the blocked ranges, and per variant
logp_A, logp_B (log-softmax over the vocabulary), z_A, z_B (raw logits), p_letter = p(A) + p(B), top1. Resumes by sample_id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_at_answer import load_onsets, prompt_segments, trace_rows  # noqa: E402
from generate import MODEL, MODEL_REVISION, ROOT, Tokens, iter_traces, load_tasks  # noqa: E402

VARIANTS = ["base", "concl_all", "concl_answer", "concl_early", "concl_mid", "concl_late", "prev_all", "rand_all", "think_all"]
LAYER_OF = {"concl_early": range(0, 12), "concl_mid": range(12, 24), "concl_late": range(24, 36)}
OUT = ROOT / "data/screen"
FIG = ROOT / "figures/formal_analysis"


def blocks_of(tr: dict, n_prompt: int, sample_id: str) -> dict[str, tuple[int, int] | None]:
    """Key ranges (absolute, half-open) blocked by each variant; None when the variant has no sentence to block."""
    starts, end, onset_row = tr["sentence_starts"], tr["think_end"], tr["onset_row"]
    onset = starts[onset_row]
    concl = (n_prompt + onset, n_prompt + end)
    prev = (n_prompt + starts[onset_row - 1], n_prompt + onset) if onset_row >= 1 else None
    rand = None
    if onset_row >= 2:
        rng = np.random.default_rng(int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16))
        k = int(rng.integers(0, onset_row - 1))                     # one of the sentences before the previous one
        rand = (n_prompt + starts[k], n_prompt + starts[k + 1])
    think = (n_prompt + tr["think_start"], n_prompt + onset) if onset > tr["think_start"] else None
    return {"concl": concl, "prev": prev, "rand": rand, "think": think}


class Masks:
    """Per-layer attention masks of a list of variants, True = may attend. A variant spec: {"ranges": [(lo, hi), ...] (keys hidden),
    "queries": "from" (rows >= q_from) | "last" (the last row only), "layers": set or None (all), "heads": {layer: set(head)} or None}."""

    def __init__(self) -> None:
        self.active = False

    def build(self, n: int, q_from: int, last: int, specs: list[dict], device) -> None:
        import torch
        self.n, self.q_from, self.last, self.specs, self.device = n, q_from, last, specs, device
        self.causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
        self.plain = {}
        for v, sp in enumerate(specs):
            if sp.get("heads") is None:
                m = self.causal.clone()
                for lo, hi in sp.get("ranges", []):
                    if sp.get("queries", "from") == "last":
                        m[last, lo:hi] = False
                    else:
                        m[q_from:, lo:hi] = False
                self.plain[v] = m
        self.any_heads = any(sp.get("heads") is not None for sp in specs)
        self.active = True

    def mask_for(self, layer: int, n_heads: int = 32):
        if not self.active:
            return None
        import torch
        rows = []
        for v, sp in enumerate(self.specs):
            layers = sp.get("layers")
            if sp.get("heads") is None:
                m = self.plain[v] if (layers is None or layer in layers) else self.causal
                rows.append(m[None].expand(n_heads, -1, -1) if self.any_heads else m[None])
            else:
                heads = sp["heads"].get(layer, ())
                m = self.causal[None].expand(n_heads, -1, -1).clone()
                for h in heads:
                    for lo, hi in sp.get("ranges", []):
                        if sp.get("queries", "from") == "last":
                            m[h, self.last, lo:hi] = False
                        else:
                            m[h, self.q_from:, lo:hi] = False
                rows.append(m)
        return torch.stack(rows, 0)          # [B, H or 1, n, n]


MASKS = Masks()


def install_attention() -> str:
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def masked_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        m = MASKS.mask_for(module.layer_idx, query.shape[1])
        kwargs.pop("is_causal", None)
        if m is None:
            return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        return sdpa_attention_forward(module, query, key, value, m, dropout=dropout, scaling=scaling, is_causal=False, **kwargs)

    AttentionInterface.register("ablate", masked_attention)
    return "ablate"


def done_ids(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def run(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    tokens = Tokens(tok, ("A", "B"))
    texts = [tok.decode([i]) for i in range(len(tok))]
    items = {it["item_id"]: it for it in load_tasks(args.tasks)}
    onset_of = load_onsets(args.positions)
    out = args.run / "readouts" / "ablation-01.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = done_ids(out)
    letter_ids = {L: tok(L, add_special_tokens=False)["input_ids"][0] for L in ("A", "B")}
    traces = [r for r in iter_traces(args.run, ("sample_id", "item_id", "token_ids", "answer_pos", "think_tokens", "logp_A", "logp_B"))
              if r["think_tokens"] is not None and r["answer_pos"] is not None and r["sample_id"] not in done]
    if args.limit:
        traces = traces[: args.limit]
    print(f"{len(done)} traces done, {len(traces)} to run; dry={args.dry}", flush=True)
    model = None
    if not args.dry:
        import torch
        from transformers import AutoModelForCausalLM
        name = install_attention()
        try:
            model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=name)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation=name)
        model.to("cuda").eval()
    prompts: dict[str, tuple[list[int], np.ndarray]] = {}
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(traces):
            if r["item_id"] not in prompts:
                prompts[r["item_id"]] = prompt_segments(tok, items[r["item_id"]])
            p_ids, _ = prompts[r["item_id"]]
            tr = trace_rows(r, tokens, texts, len(p_ids), onset_of)
            ids = p_ids + tr["gen"]
            n, P = len(ids), len(p_ids)
            blocks = blocks_of(tr, P, r["sample_id"])
            q_from, last = P + tr["think_end"], n - 1
            rec = {"sample_id": r["sample_id"], "item_id": r["item_id"], "n_prompt": P, "n_total": n, "think_start": tr["think_start"],
                   "onset": tr["sentence_starts"][tr["onset_row"]], "think_end": tr["think_end"], "answer_pos": tr["answer_pos"], "n_sent": tr["n_sent"],
                   "blocks": {k: list(v) if v else None for k, v in blocks.items()}, "q_from": q_from,
                   "trace_logp_A": r["logp_A"], "trace_logp_B": r["logp_B"]}
            if args.dry:
                print(f"{r['sample_id']}: n {n}, queries >= {q_from}, blocks " + ", ".join(f"{k} {v}" for k, v in blocks.items()), flush=True)
                continue
            import torch
            specs = []
            for name in VARIANTS:
                if name == "base":
                    specs.append({})
                elif name == "concl_answer":
                    specs.append({"ranges": [blocks["concl"]], "queries": "last"})
                elif name in ("concl_all",):
                    specs.append({"ranges": [blocks["concl"]]})
                elif name in LAYER_OF:
                    specs.append({"ranges": [blocks["concl"]], "layers": set(LAYER_OF[name])})
                elif name == "prev_all":
                    specs.append({"ranges": [blocks["prev"]] if blocks["prev"] else []})
                elif name == "rand_all":
                    specs.append({"ranges": [blocks["rand"]] if blocks["rand"] else []})
                elif name == "think_all":
                    specs.append({"ranges": [blocks["think"]] if blocks["think"] else []})
            MASKS.build(n, q_from, last, specs, "cuda")
            with torch.inference_mode():
                x = torch.as_tensor([ids] * len(VARIANTS), device="cuda")
                try:
                    o = model(input_ids=x, use_cache=False, logits_to_keep=1)
                except TypeError:
                    o = model(input_ids=x, use_cache=False)
            MASKS.active = False
            lg = o.logits[:, -1].float()                                   # [9, V]
            lsm = torch.log_softmax(lg, -1)
            top = lg.argmax(-1)
            for v, name in enumerate(VARIANTS):
                has = not ((name == "prev_all" and not blocks["prev"]) or (name == "rand_all" and not blocks["rand"]) or (name == "think_all" and not blocks["think"]))
                rec[name] = {"z_A": lg[v, letter_ids["A"]].item(), "z_B": lg[v, letter_ids["B"]].item(), "logp_A": lsm[v, letter_ids["A"]].item(),
                             "logp_B": lsm[v, letter_ids["B"]].item(), "p_letter": (lsm[v, letter_ids["A"]].exp() + lsm[v, letter_ids["B"]].exp()).item(),
                             "top1": tok.decode([top[v].item()]), "defined": has}
            del o, lg, lsm
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % args.log_every == 0:
                print(f"{i + 1}/{len(traces)} traces, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(traces)} traces in {time.time() - t0:.0f}s -> {out}", flush=True)


def load_model_and_tok(path):
    import torch
    from transformers import AutoModelForCausalLM
    name = install_attention()
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=name)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation=name)
    model.to("cuda").eval()
    return model


def read_last(model, ids: list[int], n_copies: int, watch: dict[str, int]):
    """Logits at the last position for every copy; returns per copy {token: (z, logp)}, top1 id."""
    import torch
    with torch.inference_mode():
        x = torch.as_tensor([ids] * n_copies, device="cuda")
        try:
            o = model(input_ids=x, use_cache=False, logits_to_keep=1)
        except TypeError:
            o = model(input_ids=x, use_cache=False)
    lg = o.logits[:, -1].float()
    lsm = torch.log_softmax(lg, -1)
    out = []
    top2 = lg.topk(2, dim=-1)
    for v in range(n_copies):
        out.append({"z": {k: lg[v, i].item() for k, i in watch.items()}, "logp": {k: lsm[v, i].item() for k, i in watch.items()}, "top1_id": int(lg[v].argmax().item()),
                    "top2_id": int(top2.indices[v, 1].item()), "z_top1": top2.values[v, 0].item(), "z_top2": top2.values[v, 1].item()})
    del o, lg, lsm
    return out


def setup_common(args):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    path = snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    tokens = Tokens(tok, ("A", "B"))
    texts = [tok.decode([i]) for i in range(len(tok))]
    items = {it["item_id"]: it for it in load_tasks(args.tasks)}
    onset_of = load_onsets(args.positions)
    traces = [r for r in iter_traces(args.run, ("sample_id", "item_id", "token_ids", "answer_pos", "think_tokens", "logp_A", "logp_B"))
              if r["think_tokens"] is not None and r["answer_pos"] is not None]
    return path, tok, tokens, texts, items, onset_of, traces


def sentence_ranges(tr: dict, n_prompt: int) -> list[tuple[int, int]]:
    """Absolute token ranges of the sentences before the conclusion unit, index k-1 for sentence k (1-based)."""
    starts = tr["sentence_starts"]
    out = []
    for k in range(tr["onset_row"]):
        out.append((n_prompt + starts[k], n_prompt + starts[k + 1]))
    return out


CLASS_OF = {"restate finding": "restate", "inference (finding -> option)": "inference", "claim about option": "inference",
            "reconsider: other": "reconsider_doubt", "reconsider: alternative": "reconsider_doubt", "doubt (no content)": "reconsider_doubt"}
CLASS_VARIANTS = ["base", "hide_restate", "hide_inference", "hide_reconsider_doubt", "hide_other", "hide_random_matched"]


def job_classes(args: argparse.Namespace) -> None:
    path, tok, tokens, texts, items, onset_of, traces = setup_common(args)
    mv = pd.read_csv(ROOT / "data/screen/medxpertqa_moves.csv", usecols=["sample_id", "k", "move"])
    mv["cls"] = mv.move.map(CLASS_OF).fillna("other")
    cls_of = {sid: dict(zip(g.k, g.cls)) for sid, g in mv.groupby("sample_id")}
    out = args.run / "readouts" / "ablation_classes-01.jsonl"
    done = done_ids(out)
    traces = [r for r in traces if r["sample_id"] not in done][: args.limit or None]
    print(f"classes: {len(done)} done, {len(traces)} to run", flush=True)
    model = load_model_and_tok(path)
    watch = {"A": tok("A", add_special_tokens=False)["input_ids"][0], "B": tok("B", add_special_tokens=False)["input_ids"][0]}
    prompts, t0 = {}, time.time()
    with out.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(traces):
            if r["item_id"] not in prompts:
                prompts[r["item_id"]] = prompt_segments(tok, items[r["item_id"]])
            p_ids, _ = prompts[r["item_id"]]
            tr = trace_rows(r, tokens, texts, len(p_ids), onset_of)
            P = len(p_ids)
            ids = p_ids + tr["gen"]
            n, q_from, last = len(ids), P + tr["think_end"], len(ids) - 1
            ranges = sentence_ranges(tr, P)
            classes = cls_of.get(r["sample_id"], {})
            by = {c: [rg for k, rg in enumerate(ranges, start=1) if classes.get(k, "other") == c] for c in ("restate", "inference", "reconsider_doubt", "other")}
            restate_tokens = sum(hi - lo for lo, hi in by["restate"])
            rng = np.random.default_rng(int(hashlib.sha256(r["sample_id"].encode()).hexdigest()[:8], 16))
            order = rng.permutation(len(ranges))
            rand, tot = [], 0
            for k in order:
                if tot >= restate_tokens:
                    break
                rand.append(ranges[k]); tot += ranges[k][1] - ranges[k][0]
            specs = [{}, {"ranges": by["restate"]}, {"ranges": by["inference"]}, {"ranges": by["reconsider_doubt"]}, {"ranges": by["other"]}, {"ranges": rand}]
            MASKS.build(n, q_from, last, specs, "cuda")
            res = read_last(model, ids, len(specs), watch)
            MASKS.active = False
            rec = {"sample_id": r["sample_id"], "item_id": r["item_id"], "n_total": n, "n_sentences_before": len(ranges),
                   "trace_logp_A": r["logp_A"], "trace_logp_B": r["logp_B"],
                   "blocks": {"restate": [restate_tokens, len(by["restate"])], "inference": [sum(h - l for l, h in by["inference"]), len(by["inference"])],
                              "reconsider_doubt": [sum(h - l for l, h in by["reconsider_doubt"]), len(by["reconsider_doubt"])], "other": [sum(h - l for l, h in by["other"]), len(by["other"])],
                              "random_matched": [tot, len(rand)]}}
            for name, sp, rr in zip(CLASS_VARIANTS, specs, res):
                rec[name] = {"logp_A": rr["logp"]["A"], "logp_B": rr["logp"]["B"], "z_A": rr["z"]["A"], "z_B": rr["z"]["B"], "top1": tok.decode([rr["top1_id"]]),
                             "defined": name == "base" or len(sp.get("ranges", [])) > 0}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if (i + 1) % args.log_every == 0:
                print(f"{i + 1}/{len(traces)} traces, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(traces)} traces in {time.time() - t0:.0f}s -> {out}", flush=True)


STOP_VARIANTS = ["base", "hide_conclusion_rest", "hide_prev", "hide_thinking", "hide_stem"]


def job_stop(args: argparse.Namespace) -> None:
    path, tok, tokens, texts, items, onset_of, traces = setup_common(args)
    out = args.run / "readouts" / "ablation_stop-01.jsonl"
    done = done_ids(out)
    traces = [r for r in traces if r["sample_id"] not in done][: args.limit or None]
    print(f"stop: {len(done)} done, {len(traces)} to run", flush=True)
    model = load_model_and_tok(path)
    watch = {"</think>": tok.convert_tokens_to_ids("</think>"), " A": tok(" A", add_special_tokens=False)["input_ids"][0], " B": tok(" B", add_special_tokens=False)["input_ids"][0]}
    prompts, t0 = {}, time.time()
    with out.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(traces):
            if r["item_id"] not in prompts:
                prompts[r["item_id"]] = prompt_segments(tok, items[r["item_id"]])
            p_ids, p_seg = prompts[r["item_id"]]
            tr = trace_rows(r, tokens, texts, len(p_ids), onset_of)
            P = len(p_ids)
            ids = p_ids + tr["gen"][: tr["think_end"]]                      # the last input token is the e row
            n, last = len(ids), len(ids) - 1
            onset = tr["sentence_starts"][tr["onset_row"]]
            stem = np.flatnonzero(p_seg == 2)
            prev = (P + tr["sentence_starts"][tr["onset_row"] - 1], P + onset) if tr["onset_row"] >= 1 else None
            specs = [{}, {"ranges": [(P + onset, last)], "queries": "last"}, {"ranges": [prev] if prev else [], "queries": "last"},
                     {"ranges": [(P + tr["think_start"], P + onset)] if onset > tr["think_start"] else [], "queries": "last"},
                     {"ranges": [(int(stem[0]), int(stem[-1]) + 1)], "queries": "last"}]
            MASKS.build(n, last, last, specs, "cuda")
            res = read_last(model, ids, len(specs), watch)
            MASKS.active = False
            rec = {"sample_id": r["sample_id"], "item_id": r["item_id"], "n_total": n, "onset": onset, "think_end": tr["think_end"]}
            for name, sp, rr in zip(STOP_VARIANTS, specs, res):
                rec[name] = {"logp_close": rr["logp"]["</think>"], "z_close": rr["z"]["</think>"], "logp_A": rr["logp"][" A"], "logp_B": rr["logp"][" B"],
                             "top1": tok.decode([rr["top1_id"]]), "top2": tok.decode([rr["top2_id"]]), "z_top1": rr["z_top1"], "z_top2": rr["z_top2"],
                             "margin_close": rr["z"]["</think>"] - (rr["z_top2"] if rr["top1_id"] == watch["</think>"] else rr["z_top1"]),
                             "defined": name == "base" or len(sp.get("ranges", [])) > 0}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if (i + 1) % args.log_every == 0:
                print(f"{i + 1}/{len(traces)} traces, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(traces)} traces in {time.time() - t0:.0f}s -> {out}", flush=True)


HEAD_VARIANTS = ["base", "late_top32", "late_top128", "late_rest", "early_top32", "all"]


def job_heads(args: argparse.Namespace) -> None:
    path, tok, tokens, texts, items, onset_of, traces = setup_common(args)
    ranking = json.loads((ROOT / "data/screen/medxpertqa_top_heads.json").read_text())
    late, early = ranking["ranked_late"], ranking["ranked_early"]

    def head_set(rows):
        d = {}
        for x in rows:
            d.setdefault(x["layer"], set()).add(x["head"])
        return d
    sets = {"late_top32": head_set(late[:32]), "late_top128": head_set(late[:128]), "late_rest": head_set(late[128:]), "early_top32": head_set(early[:32])}
    out = args.run / "readouts" / "ablation_heads-01.jsonl"
    done = done_ids(out)
    traces = [r for r in traces if r["sample_id"] not in done][: args.limit or None]
    print(f"heads: {len(done)} done, {len(traces)} to run", flush=True)
    model = load_model_and_tok(path)
    watch = {"A": tok("A", add_special_tokens=False)["input_ids"][0], "B": tok("B", add_special_tokens=False)["input_ids"][0]}
    prompts, t0 = {}, time.time()
    with out.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(traces):
            if r["item_id"] not in prompts:
                prompts[r["item_id"]] = prompt_segments(tok, items[r["item_id"]])
            p_ids, _ = prompts[r["item_id"]]
            tr = trace_rows(r, tokens, texts, len(p_ids), onset_of)
            P = len(p_ids)
            ids = p_ids + tr["gen"]
            n, q_from, last = len(ids), P + tr["think_end"], len(ids) - 1
            concl = (P + tr["sentence_starts"][tr["onset_row"]], P + tr["think_end"])
            specs = [{}] + [{"ranges": [concl], "heads": sets[v]} for v in HEAD_VARIANTS[1:-1]] + [{"ranges": [concl]}]
            MASKS.build(n, q_from, last, specs, "cuda")
            res = read_last(model, ids, len(specs), watch)
            MASKS.active = False
            rec = {"sample_id": r["sample_id"], "item_id": r["item_id"], "n_total": n, "trace_logp_A": r["logp_A"], "trace_logp_B": r["logp_B"]}
            for name, rr in zip(HEAD_VARIANTS, res):
                rec[name] = {"logp_A": rr["logp"]["A"], "logp_B": rr["logp"]["B"], "z_A": rr["z"]["A"], "z_B": rr["z"]["B"], "top1": tok.decode([rr["top1_id"]])}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if (i + 1) % args.log_every == 0:
                print(f"{i + 1}/{len(traces)} traces, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(traces)} traces in {time.time() - t0:.0f}s -> {out}", flush=True)


# --------------------------------------------------------------------------------------------------------------- analyze

def analyze(args: argparse.Namespace) -> None:
    from scipy.stats import spearmanr
    runs = args.runs or [ROOT / f"runs/medxpertqa/main-qwen3-8b-{o}" for o in ("ab", "ba")]
    la = pd.read_csv(OUT / "medxpertqa_length_accuracy_main.csv")
    ls = pd.read_csv(OUT / "medxpertqa_last_sentence_main.csv", usecols=["sample_id", "y_closed_onset", "y_closed_end", "jump_closed"])
    att = pd.read_csv(OUT / "medxpertqa_attention_answer.csv")
    att = att[att.row == "a"][["sample_id", "mass_conclusion", "mass_conclusion_late", "mass_evidence", "mass_own_text", "ent_norm_mean"]]
    rows = []
    for run in runs:
        for line in (run / "readouts/ablation-01.jsonl").open(encoding="utf-8"):
            m = json.loads(line)
            d = {"sample_id": m["sample_id"], "item_id": m["item_id"], "n_sent": m["n_sent"], "concl_tokens": m["think_end"] - m["onset"], "trace_X": m["trace_logp_A"] - m["trace_logp_B"]}
            for v in VARIANTS:
                d[f"X_{v}"] = (m[v]["logp_A"] - m[v]["logp_B"]) if m[v]["defined"] else np.nan
                d[f"pletter_{v}"] = m[v]["p_letter"] if m[v]["defined"] else np.nan
                d[f"top1_{v}"] = m[v]["top1"] if m[v]["defined"] else None
            rows.append(d)
    t = pd.DataFrame(rows).merge(la[["sample_id", "pair_id", "order", "label", "correct", "think_tokens", "rel", "quintile", "conf"]], on="sample_id")
    t = t.merge(ls, on="sample_id", how="left").merge(att, on="sample_id", how="left")
    sign_final = np.where(t.correct, 1.0, -1.0) * np.where(t.label == "A", 1.0, -1.0)   # X -> toward the trace's final answer
    for v in VARIANTS:
        t[f"y_{v}"] = t[f"X_{v}"] * sign_final
    t["y_onset"] = np.where(t.correct, t.y_closed_onset, -t.y_closed_onset)
    t["y_end"] = np.where(t.correct, t.y_closed_end, -t.y_closed_end)
    for v in VARIANTS[1:]:
        t[f"effect_{v}"] = t.y_base - t[f"y_{v}"]                      # nats removed from the answer's lean by the block
    t["jump"] = t.y_base - t.y_onset                                    # the conclusion's contribution by truncation
    t["removed_share_concl_all"] = t.effect_concl_all / t.jump
    t["direct_share"] = t.effect_concl_answer / t.effect_concl_all
    t.to_csv(OUT / "medxpertqa_ablation_conclusion.csv", index=False)
    print(f"{len(t)} traces -> {OUT / 'medxpertqa_ablation_conclusion.csv'}")
    c = t.dropna(subset=["X_base"])
    print(f"base check: corr(trace X, base X) = {c.trace_X.corr(c.X_base):.4f}, median |diff| = {(c.trace_X - c.X_base).abs().median():.3f} nats")
    pd.set_option("display.width", 250)
    print("\nanswer lean toward the final answer (nats), medians:")
    print(t[["y_onset", "y_base", "y_concl_all", "y_concl_answer", "y_concl_early", "y_concl_mid", "y_concl_late", "y_prev_all", "y_rand_all", "y_think_all"]].median().round(2).to_string())
    print("\neffect of each block = y_base - y_blocked (nats removed), median [IQR]; share of traces whose answer flips sign:")
    for v in VARIANTS[1:]:
        e = t[f"effect_{v}"].dropna()
        flip = (np.sign(t[f"y_{v}"]) != np.sign(t.y_base)).mean()
        print(f"  {v:13s} {e.median():+6.2f} [{e.quantile(.25):+6.2f}, {e.quantile(.75):+6.2f}]  n {len(e)}  flips {flip:.1%}")
    big = t[t.jump.abs() > 2]
    print(f"\nconcl_all vs truncation (traces with |jump| > 2 nats, n {len(big)}): removed share median {big.removed_share_concl_all.median():.2f} "
          f"[{big.removed_share_concl_all.quantile(.25):.2f}, {big.removed_share_concl_all.quantile(.75):.2f}]; y_concl_all - y_onset median {(big.y_concl_all - big.y_onset).median():+.2f}")
    b2 = t[t.effect_concl_all.abs() > 1]
    print(f"direct share (answer-row block / all-rows block, traces with |effect| > 1 nat, n {len(b2)}): median {b2.direct_share.median():.2f} [{b2.direct_share.quantile(.25):.2f}, {b2.direct_share.quantile(.75):.2f}]")
    print(f"layer groups, median effect: early {t.effect_concl_early.median():+.2f}, mid {t.effect_concl_mid.median():+.2f}, late {t.effect_concl_late.median():+.2f} (sum {(t.effect_concl_early + t.effect_concl_mid + t.effect_concl_late).median():+.2f} vs all {t.effect_concl_all.median():+.2f})")

    def wrho(col_x, col_y):
        r = [spearmanr(g[col_x], g[col_y], nan_policy="omit").correlation for _, g in t.groupby("item_id") if g[[col_x, col_y]].dropna().shape[0] > 10]
        return float(np.nanmedian(r))

    print("\nwithin-prompt Spearman medians:")
    for name, col in (("conclusion effect (concl_all)", "effect_concl_all"), ("conclusion effect, answer row only", "effect_concl_answer"), ("removing the earlier thinking (think_all)", "effect_think_all"),
                      ("control prev sentence", "effect_prev_all"), ("control random sentence", "effect_rand_all")):
        print(f"  {name:42s} vs length {wrho('think_tokens', col):+.2f}   vs conclusion attention share {wrho('mass_conclusion', col):+.2f}   vs late-layer share {wrho('mass_conclusion_late', col):+.2f}   vs truncation jump {wrho('jump', col):+.2f}")
    q = t.groupby("quintile").agg(rel=("rel", "median"), effect_concl=("effect_concl_all", "median"), effect_answer_row=("effect_concl_answer", "median"), effect_prev=("effect_prev_all", "median"),
                                   effect_rand=("effect_rand_all", "median"), effect_think=("effect_think_all", "median"), jump=("jump", "median"), share=("mass_conclusion", "median"))
    print("\nby relative-length quintile (medians):")
    print(q.round(3).to_string())
    fig(t)


def fig(t: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from length_accuracy import BLUE, PAIRS, RED, rho

    fig_, axes = plt.subplots(4, 6, figsize=(22, 13), sharex=True, sharey=True)
    axs = axes.flatten()
    i = 0
    for pair in PAIRS:
        for order in ("ab", "ba"):
            ax = axs[i]; i += 1
            g = t[(t.pair_id == pair) & (t.order == order)]
            ax.scatter(g.rel, g.effect_prev_all, s=9, color="0.6", alpha=0.5, edgecolors="none", label="control: previous sentence blocked")
            ax.scatter(g.rel, g.effect_concl_all, s=11, c=[BLUE if c else RED for c in g.correct], alpha=0.6, edgecolors="none", label="conclusion blocked")
            med = g.groupby("quintile").agg(x=("rel", "median"), y=("effect_concl_all", "median"), y2=("effect_think_all", "median"))
            ax.plot(med.x, med.y, "s-", color="black", ms=4, lw=0.8)
            ax.plot(med.x, med.y2, "^--", color="#2ca02c", ms=4, lw=0.8, label="earlier thinking blocked (median)")
            ax.axhline(0, color="0.4", lw=0.6, ls="--")
            ax.set_title(f"{pair} {order}  acc {g.correct.mean():.2f}\nrho(len, effect) {rho(g.think_tokens, g.effect_concl_all):+.2f}   rho(share, effect) {rho(g.mass_conclusion, g.effect_concl_all):+.2f}", fontsize=8.5)
            ax.grid(alpha=0.2)
    axs[0].legend(fontsize=6.5, loc="upper right")
    axs[0].set_xticks([-2, -1, 0, 1, 2]); axs[0].set_xticklabels(["1/4x", "1/2x", "1x", "2x", "4x"])
    for ax in axes[-1]:
        ax.set_xlabel("thinking length / the prompt's median length")
    fig_.supylabel("change of the answer lean when the block is applied (nats, toward the final answer; positive = the lean falls)")
    fig_.suptitle("Ablation at the answer: the conclusion unit's tokens hidden from </think> and the answer token (all layers), against thinking length; "
                  "grey = the sentence before the conclusion hidden instead; green = the earlier thinking hidden", fontsize=10)
    fig_.tight_layout(rect=(0.01, 0, 1, 0.96))
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "fig60_ablation_conclusion_vs_length.png"
    fig_.savefig(path, dpi=130)
    plt.close(fig_)
    print(f"figure -> {path}")


# ------------------------------------------------------------------------------------------- study104: masking at the answer

MASK104 = ["base", "conclusion", "conclusion_answer_row", "conclusion_end", "conclusion_letter", "rand_for_conclusion",
           "earlier_verdicts", "rand_for_verdicts", "thinking", "question", "options", "only_conclusion",
           "conclusion_early", "conclusion_mid", "conclusion_late"]
MASK104_NOTHINK = ["base", "question", "stem", "options", "instruction"]
JOB104 = "mask104_job.jsonl"


def prepare104(args: argparse.Namespace) -> None:
    """Job file of every thinking run: per trace the sentences before the conclusion unit that name a conclusion."""
    from attention_at_answer import RUNS, VERDICT_TYPES
    s = pd.read_csv(OUT / "study104_attention_sentences.csv.gz", usecols=["model", "sample_id", "k", "n_sent", "type", "in_conclusion"])
    for model in args.models:
        g = s[s.model == model]
        n_sent = g.groupby("sample_id").n_sent.first().to_dict()
        v = g[~g.in_conclusion & g.type.isin(VERDICT_TYPES)]
        verdicts = v.groupby("sample_id").k.apply(lambda x: sorted(int(k) for k in x)).to_dict()
        for o in ("ab", "ba"):
            run = RUNS / f"study104-{model}-{o}"
            ids = [r["sample_id"] for r in iter_traces(run, ("sample_id",)) if r["sample_id"] in n_sent]
            (run / JOB104).write_text("".join(json.dumps({"sample_id": sid, "verdict_ks": verdicts.get(sid, []), "n_sent": int(n_sent[sid])}) + "\n"
                                              for sid in ids), encoding="utf-8")
            print(f"{run / JOB104}: {len(ids)} traces, {sum(1 for sid in ids if verdicts.get(sid))} with an earlier verdict")


def runs_of(flags: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ranges of the True runs of a boolean array."""
    w = np.flatnonzero(flags)
    if not len(w):
        return []
    cut = np.flatnonzero(np.diff(w) > 1)
    return [(int(w[a]), int(w[b]) + 1) for a, b in zip(np.r_[0, cut + 1], np.r_[cut, len(w) - 1])]


def matched_sentences(cands: list[int], starts: list[int], ends: list[int], target: int, rng) -> list[int]:
    """Candidate sentences in a seeded random order, added until they hold at least `target` tokens (all of them if fewer)."""
    out, n = [], 0
    for k in rng.permutation(cands):
        if n >= target:
            break
        out.append(int(k))
        n += ends[k] - starts[k]
    return sorted(out)


def mask_specs104(ctx: dict, n_prompt: int, gen: list[int], texts: list[str], answer: str, verdict_ks: list[int], sample_id: str,
                  n_layers: int, thinking: bool) -> tuple[int, dict[str, dict]]:
    """The first row after the thinking (the cache holds everything before it) and, per variant, the blocked key ranges (absolute,
    half-open), the rows ("tail" = every row after the thinking, "last" = the letter-writing row) and the layers (None = all)."""
    seg, P = ctx["seg"], n_prompt
    question = runs_of(np.isin(seg, [2, 3, 4, 5]))
    if not thinking:
        n_prefix = int(np.flatnonzero(seg == 9)[0])
        specs = {"base": [], "question": question, "stem": runs_of(seg == 2), "options": runs_of(np.isin(seg, [3, 4])),
                 "instruction": runs_of(seg == 5)}
        return n_prefix, {k: {"ranges": v, "rows": "tail", "layers": None} for k, v in specs.items()}
    starts, ts, te, onset = ctx["sentence_starts"], ctx["think_start"], ctx["think_end"], ctx["onset"]
    ends = starts[1:] + [te]
    k_on = starts.index(onset)
    whole = lambda ks: [(P + starts[k], P + ends[k]) for k in ks]                       # noqa: E731
    vks = [k for k in verdict_ks if k < k_on]
    nonv = [k for k in range(k_on) if k not in set(vks)]
    rng = np.random.default_rng(int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16))
    rand_c = matched_sentences(nonv, starts, ends, te - onset, rng)
    rand_v = matched_sentences(nonv, starts, ends, sum(ends[k] - starts[k] for k in vks), rng) if vks else []
    concl = [(P + onset, P + te)]
    thinking_r = [(P + ts, P + onset)] if onset > ts else []
    thirds = [set(int(x) for x in t) for t in np.array_split(np.arange(n_layers), 3)]
    specs = {"base": ([], "tail", None), "conclusion": (concl, "tail", None), "conclusion_answer_row": (concl, "last", None),
             "conclusion_end": ([(P + te - 1, P + te)], "tail", None),
             "conclusion_letter": ([(P + j, P + j + 1) for j in range(onset, te) if texts[gen[j]].strip() == answer], "tail", None),
             "rand_for_conclusion": (whole(rand_c), "tail", None), "earlier_verdicts": (whole(vks), "tail", None),
             "rand_for_verdicts": (whole(rand_v), "tail", None), "thinking": (thinking_r, "tail", None), "question": (question, "tail", None),
             "options": (runs_of(np.isin(seg, [3, 4])), "tail", None), "only_conclusion": (question + thinking_r, "tail", None),
             "conclusion_early": (concl, "tail", thirds[0]), "conclusion_mid": (concl, "tail", thirds[1]), "conclusion_late": (concl, "tail", thirds[2])}
    return P + te, {k: {"ranges": r, "rows": rows, "layers": layers} for k, (r, rows, layers) in specs.items()}


class TailMask:
    """The attention mask of the rows after the cached prefix: causal, minus the blocked keys for the chosen rows in the chosen layers."""

    def __init__(self) -> None:
        self.active = False

    def set(self, n_prefix: int, n_total: int, spec: dict, device) -> None:
        import torch
        qpos = torch.arange(n_prefix, n_total, device=device)
        causal = torch.arange(n_total, device=device)[None, :] <= qpos[:, None]          # [q, k], True = may attend
        blocked = causal.clone()
        rows = slice(None) if spec["rows"] == "tail" else slice(-1, None)
        for lo, hi in spec["ranges"]:
            blocked[rows, max(lo, 1): hi] = False                                        # the first token is never blocked
        self.causal, self.blocked, self.layers = causal[None, None], blocked[None, None], spec["layers"]
        self.active = True

    def mask_for(self, layer: int):
        return self.blocked if self.layers is None or layer in self.layers else self.causal


TAIL = TailMask()


def install_attention104() -> str:
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def tail_masked_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        if not TAIL.active:
            return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        kwargs.pop("is_causal", None)
        return sdpa_attention_forward(module, query, key, value, TAIL.mask_for(module.layer_idx), dropout=dropout, scaling=scaling,
                                      is_causal=False, **kwargs)

    AttentionInterface.register("mask104", tail_masked_attention)
    return "mask104"


def load_model104(path: str):
    import torch
    from transformers import AutoModelForCausalLM
    name = install_attention104()
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=name, device_map="cuda")
    except (ImportError, ValueError, TypeError):
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=name)
    return model.to("cuda").eval()


def read_variants104(model, ids: list[int], n_prefix: int, specs: dict, names: list[str], letters: dict[str, int], tok) -> dict:
    """Prefix once with a cache, then the rows after it once per variant; the cache is cut back to the prefix after each."""
    import torch
    out = {}
    with torch.inference_mode():
        o = model(input_ids=torch.as_tensor([ids[:n_prefix]], device="cuda"), use_cache=True, logits_to_keep=1)
        cache = o.past_key_values
        tail = torch.as_tensor([ids[n_prefix:]], device="cuda")
        for name in names:
            TAIL.set(n_prefix, len(ids), specs[name], "cuda")
            o = model(input_ids=tail, past_key_values=cache, use_cache=True, logits_to_keep=1)
            TAIL.active = False
            cache.crop(-(len(ids) - n_prefix))                 # a negative count removes that many tokens (transformers 4.x and 5.x)
            assert cache.get_seq_length() == n_prefix, (cache.get_seq_length(), n_prefix)
            lg = o.logits[0, -1].float()
            lsm = torch.log_softmax(lg, -1)
            out[name] = {"z_A": lg[letters["A"]].item(), "z_B": lg[letters["B"]].item(), "logp_A": lsm[letters["A"]].item(),
                         "logp_B": lsm[letters["B"]].item(), "p_letter": (lsm[letters["A"]].exp() + lsm[letters["B"]].exp()).item(),
                         "top1": tok.decode([int(lg.argmax())])}
        del o, cache
    return out


def run104(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoTokenizer
    from attention_at_answer import FIELDS104, answer_context, conclusion_onsets, model_dir
    from generate import MODEL_NAME, TASKS, check_run_model

    check_run_model(args.run)
    thinking = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))["mode"] == "think"
    names = MASK104 if thinking else MASK104_NOTHINK
    path = model_dir()
    tok = AutoTokenizer.from_pretrained(path)
    n_layers = AutoConfig.from_pretrained(path).num_hidden_layers
    tokens = Tokens(tok, ("A", "B"))
    texts = [tok.decode([i]) for i in range(len(tok))]
    keep = set(pd.read_csv(args.questions).item_id)
    items = {it["item_id"]: it for it in load_tasks(TASKS) if it["item_id"] in keep}
    onset_of = conclusion_onsets(args.run) if thinking else {}
    job = {}
    if thinking:
        for line in (args.run / JOB104).open(encoding="utf-8"):
            j = json.loads(line)
            job[j["sample_id"]] = j
    out = args.run / "readouts" / "mask104-01.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set() if args.dry else done_ids(out)
    traces = [r for r in iter_traces(args.run, FIELDS104) if r["item_id"] in keep and r["answer_pos"] is not None and r["sample_id"] not in done]
    if args.limit:
        traces = traces[: args.limit]
    print(f"{MODEL_NAME} {'thinking' if thinking else 'no thinking'}: {len(done)} traces done, {len(traces)} to run, {len(names)} variants; "
          f"dry={args.dry}", flush=True)
    model = None if args.dry else load_model104(path)
    prompts: dict[str, tuple[list[int], np.ndarray]] = {}
    defined, n_blocked = {v: 0 for v in names}, {v: [] for v in names}
    t0 = time.time()
    with (open(os.devnull, "w") if args.dry else out.open("a", encoding="utf-8")) as fh:
        for i, r in enumerate(traces):
            iid = r["item_id"]
            if iid not in prompts:
                prompts[iid] = prompt_segments(tok, items[iid], thinking)
            p_ids, p_seg = prompts[iid]
            ctx = answer_context(r, tokens, texts, p_ids, p_seg, onset_of, thinking)
            ids = ctx["ids"]
            j = job.get(r["sample_id"], {"verdict_ks": [], "n_sent": ctx["n_sent"]})
            assert j["n_sent"] == ctx["n_sent"], (r["sample_id"], j["n_sent"], ctx["n_sent"])
            n_prefix, specs = mask_specs104(ctx, len(p_ids), r["token_ids"], texts, r["answer"], j["verdict_ks"], r["sample_id"], n_layers, thinking)
            for v in names:
                nb = sum(hi - max(lo, 1) for lo, hi in specs[v]["ranges"])
                specs[v]["n_blocked"] = nb
                defined[v] += v == "base" or nb > 0
                n_blocked[v].append(nb)
            if args.dry:
                if i < args.show:
                    print(f"=== {r['sample_id']} answer {r['answer']}: {len(ids)} tokens, rows after the thinking {ids[n_prefix:]} = "
                          f"{tok.decode(ids[n_prefix:])!r}")
                    for v in names[1:]:
                        sp = specs[v]
                        t = " | ".join(tok.decode(ids[lo:hi]).replace("\n", "\\n")[:60] for lo, hi in sp["ranges"][:4])
                        layers = "all" if sp["layers"] is None else f"{min(sp['layers'])}-{max(sp['layers'])}"
                        print(f"  {v:22s} {sp['n_blocked']:5d} tok  rows {sp['rows']:4s} layers {layers:6s} {t[:150]!r}")
                continue
            res = read_variants104(model, ids, n_prefix, specs, [v for v in names if v == "base" or specs[v]["n_blocked"] > 0], tokens.options, tok)
            rec = {"sample_id": r["sample_id"], "item_id": iid, "model": MODEL_NAME, "mode": "think" if thinking else "nothink",
                   "answer": r["answer"], "correct": r["correct"], "think_tokens": r["think_tokens"], "n_prefix": n_prefix, "n_total": len(ids),
                   "onset": ctx["onset"], "think_end": ctx["think_end"], "trace_logp_A": r["logp_A"], "trace_logp_B": r["logp_B"]}
            for v in names:
                rec[v] = {"defined": v in res, "n_blocked": specs[v]["n_blocked"], **res.get(v, {})}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % args.log_every == 0:
                print(f"{i + 1}/{len(traces)} traces, {time.time() - t0:.0f}s", flush=True)
    print("defined per variant: " + ", ".join(f"{v} {defined[v]}" for v in names))
    print("blocked tokens, median: " + ", ".join(f"{v} {int(np.median(n_blocked[v])) if n_blocked[v] else 0}" for v in names[1:]))
    print(f"done: {len(traces)} traces in {time.time() - t0:.0f}s -> {out}", flush=True)


def analyze104(args: argparse.Namespace) -> None:
    from attention_at_answer import RUNS, nanmedian, spearman, study104_runs
    rows = []
    for model in args.models:
        jumps = {}
        for mode, run in study104_runs(RUNS, model):
            f = run / "readouts" / "mask104-01.jsonl"
            if not f.exists():
                print(f"no {f}")
                continue
            if mode == "think":
                for cf in sorted((run / "readouts").glob("conclusion-*.jsonl")):
                    for line in cf.open(encoding="utf-8"):
                        c = json.loads(line)
                        a, b = ("A", "B") if c["letter"] == "A" else ("B", "A")
                        jumps[c["sample_id"]] = (c["own"][f"logp_{a}"] - c["own"][f"logp_{b}"]) - (c["none"][f"logp_{a}"] - c["none"][f"logp_{b}"]), c["letter"]
            for line in f.open(encoding="utf-8"):
                r = json.loads(line)
                a, b = ("A", "B") if r["answer"] == "A" else ("B", "A")
                y0 = r["base"]["logp_" + a] - r["base"]["logp_" + b]
                jump, letter = jumps.get(r["sample_id"], (np.nan, None)) if mode == "think" else (np.nan, None)
                base = {"model": model, "mode": mode, "sample_id": r["sample_id"], "item_id": r["item_id"], "answer": r["answer"],
                        "correct": bool(r["correct"]), "think_tokens": r["think_tokens"], "y_base": y0,
                        "trace_y": r["trace_logp_" + a] - r["trace_logp_" + b],
                        "jump": jump if letter == r["answer"] else np.nan}
                for v in (MASK104 if mode == "think" else MASK104_NOTHINK)[1:]:
                    x = r[v]
                    if not x["defined"]:
                        continue
                    y = x["logp_" + a] - x["logp_" + b]
                    rows.append({**base, "variant": v, "n_blocked": x["n_blocked"], "y": y, "dy": y - y0, "flip": bool((y > 0) != (y0 > 0)),
                                 "p_letter": x["p_letter"], "top1": x["top1"]})
    if not rows:
        raise SystemExit("no mask104 outputs found")
    d = pd.DataFrame(rows)
    att = pd.read_csv(OUT / "study104_attention_answer.csv", usecols=["model", "mode", "sample_id", "conclusion"])
    d = d.merge(att.rename(columns={"conclusion": "att_conclusion"}), on=["model", "mode", "sample_id"], how="left")
    d.to_csv(OUT / "study104_mask_traces.csv", index=False)
    b = d.drop_duplicates(["model", "mode", "sample_id"])
    for (model, mode), g in b.groupby(["model", "mode"]):
        print(f"base check, {model} {mode}: corr(trace y, base y) {g.trace_y.corr(g.y_base):.4f}, median |diff| {(g.trace_y - g.y_base).abs().median():.3f} nats, "
              f"{len(g)} traces")
    out = []
    for (model, mode, v), g in d.groupby(["model", "mode", "variant"], sort=False):
        q = g.groupby("item_id")
        med = q.dy.median()
        r_len = [spearman(h.think_tokens, h.dy) for _, h in q] if mode == "think" else []
        r_att = [spearman(h.att_conclusion, h.dy) for _, h in q] if mode == "think" else []
        rw = [h.loc[~h.correct, "dy"].median() - h.loc[h.correct, "dy"].median() for _, h in q if h.correct.sum() >= 5 and (~h.correct).sum() >= 5]
        out.append({"model": model, "mode": mode, "variant": v, "n_traces": len(g), "n_questions": len(med),
                    "blocked_tokens": g.n_blocked.median(), "dy_median": med.median(), "dy_q25": med.quantile(0.25), "dy_q75": med.quantile(0.75),
                    "n_questions_dy_negative": int((med < 0).sum()), "flip_rate": q.flip.mean().median(), "flip_rate_all": g.flip.mean(),
                    "p_letter_median": g.p_letter.median(), "rho_len_median": nanmedian(r_len), "n_rho_len_negative": int(np.sum(np.array(r_len) < 0)),
                    "rho_att_median": nanmedian(r_att), "n_rho_att_negative": int(np.sum(np.array(r_att) < 0)),
                    "wrong_minus_right": nanmedian(rw), "n_questions_rw": len(rw),
                    "dy_over_jump": nanmedian((-g.dy / g.jump)[g.jump.abs() > 0.5]) if mode == "think" else np.nan})
    s = pd.DataFrame(out)
    s.to_csv(OUT / "study104_mask_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"{len(d)} rows -> {OUT / 'study104_mask_traces.csv'}, {OUT / 'study104_mask_summary.csv'}")
    print("dy = change of log p(answer) - log p(other) when blocked; per-question medians, then the median over questions:")
    print(s.round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p4 = sub.add_parser("prepare104")
    p4.add_argument("--models", nargs="+", default=["qwen3-8b", "qwen3-32b"])
    r4 = sub.add_parser("run104")
    r4.add_argument("--run", required=True, type=Path)
    r4.add_argument("--questions", type=Path, default=ROOT / "data/tasks/study104.csv")
    r4.add_argument("--limit", type=int, default=0)
    r4.add_argument("--log-every", type=int, default=200)
    r4.add_argument("--dry", action="store_true")
    r4.add_argument("--show", type=int, default=2, help="dry: print the blocked tokens of the first N traces")
    a4 = sub.add_parser("analyze104")
    a4.add_argument("--models", nargs="+", default=["qwen3-8b", "qwen3-32b"])
    for name in ("run", "classes", "stop", "heads"):
        j = sub.add_parser(name)
        j.add_argument("--run", required=True, type=Path)
        j.add_argument("--tasks", type=Path, default=None)
        j.add_argument("--positions", type=Path, default=ROOT / "data/screen/medxpertqa_last_sentence_main.csv")
        j.add_argument("--limit", type=int, default=0)
        j.add_argument("--log-every", type=int, default=50)
        if name == "run":
            j.add_argument("--dry", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--runs", nargs="+", type=Path, default=None)
    args = ap.parse_args()
    if args.cmd in ("prepare104", "run104", "analyze104"):
        {"prepare104": prepare104, "run104": run104, "analyze104": analyze104}[args.cmd](args)
        return
    if args.cmd != "analyze" and args.tasks is None:
        from generate import TASKS
        args.tasks = TASKS
    {"run": run, "classes": job_classes, "stop": job_stop, "heads": job_heads, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
