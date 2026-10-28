"""Attention at the answer: where the model looks when it writes the letter, how spread that attention is, and how both change
with the length of the thinking.

The 104-question study, Qwen3-8B and Qwen3-32B (MODEL_NAME selects the model, as in src/generate.py):

    MODEL_NAME=qwen3-32b python src/attention_at_answer.py run104 --run runs/medxpertqa/study104-qwen3-32b-ab   # pod, one GPU
    python src/attention_at_answer.py run104 --run runs/medxpertqa/nothink-qwen3-8b-ab                          # the no-thinking answers
    python src/attention_at_answer.py run104 --run ... --dry --limit 5      # no model: segments as text, positions, random numbers
    python src/attention_at_answer.py analyze104                            # local: tables of both models (no figures)

run104 reads one query row: the token before the answer letter (with thinking the "\\n\\n" after the model's own </think>, without
thinking the last token of the prompt, whose chat template ends with the empty "<think>\\n\\n</think>\\n\\n"), the state that writes
the letter. Every question of data/tasks/study104.csv in the run is read, every trace that has an answer. Per layer and head, from
the queries and keys of that layer (after q/k norm and RoPE): the entropy of the attention over the keys, and the attention on the
segments of the context (SEGMENTS below; thinking = the sentences before the conclusion unit, conclusion = the conclusion unit from
<run>/readouts/conclusion-*.jsonl to </think>, close = </think> and what follows it). Each weight is also multiplied by the norm of
what that key writes into the head's output, |W_O^h v_j| (Kobayashi et al. 2020, "Attention is not only a weight"), and normalised
per head again: the value-weighted attention. Per layer (mean over heads) the attention on every sentence of the thinking, raw and
value-weighted, both with the first token left out and the rest renormalised (the first token takes much of the attention and
writes almost nothing). The 10 most attended tokens per layer and the 30 most attended over all layers (first token left out; raw
and value-weighted). The letter logits at the row check the position against the trace's own logp_A/logp_B.
Output per run: <run>/readouts/answer-attention-SS.npz (shards of --shard traces; keys "<i>/ent" [L, H], "<i>/mass" and "<i>/vmass"
[L, H, 10], "<i>/vtot" [L, H] (sum of weight x norm), "<i>/sent" and "<i>/vsent" [L, n_sent], "<i>/top_idx" "<i>/top_val" [L, 10],
"<i>/topall_*" and "<i>/vtopall_*" [30]) and answer-attention-SS.jsonl (one line per trace: positions, segments and their token
counts, sentence starts, onset, letter logits, answer, correctness).

analyze104 (both models, thinking and no-thinking runs; the options as the chosen and the other one) writes
    data/screen/study104_attention_answer.csv      one row per trace: first-token share; the share of every segment with the first
                                                   token left out (all layers; early / mid / late thirds of the layers), value-weighted
                                                   share, enrichment (share / share of the tokens); earlier verdict sentences (before the
                                                   conclusion unit, sentence_type verdict_definitive / verdict_hedged / option_text, the
                                                   conclusion sentences of conclusion_count_test.py) vs other thinking; entropy
    data/screen/study104_attention_sentences.csv.gz  one row per sentence of every thinking trace: type (stopping_rule.sentence_type),
                                                   form (last_sentence_check.classify), place from the end, inside the conclusion
                                                   unit or not, share (all layers, last third), value-weighted
    data/screen/study104_attention_top_tokens.csv.gz the 30 most attended tokens of every trace, as words, with where they sit
    data/screen/study104_attention_depth.csv       per model x mode x question x right/wrong x layer: median shares (per-question curves)
    data/screen/study104_attention_heads.csv       per model x layer x head: mean shares over the thinking traces
    data/screen/study104_attention_summary.csv     per model x measure: per-question medians with and without thinking, within-question
                                                   Spearman with the thinking length, wrong minus right

The twelve-question main run of Qwen3-8B (100 traces x 12 pairs x 2 orders):

    python src/attention_at_answer.py run --run runs/medxpertqa/main-qwen3-8b-ab      # pod: one forward pass per trace (HF transformers)
    python src/attention_at_answer.py run --run ... --dry --limit 5                   # no model: positions and segments only, random numbers
    python src/attention_at_answer.py analyze                                         # local: tables and fig55/fig56 from both orders
    python src/attention_at_answer.py show <sample_id> [--rows a e onset]             # local: the most attended tokens of one trace, as words

analyze reads the shards of both orders and builds two tables. Within trace (the "0"/"s"/"e" rows placed at the proportion
t = (sentence start - thinking start) / (thinking end - thinking start), stepped onto a 0..1 grid like length_accuracy.py
truncation): data/screen/medxpertqa_attention_within.csv, one row per trace x t with the entropy over the keys / log(keys)
(mean over layers and heads), the mass on the segments, on the last three completed sentences and on the first sentence, and
the entropy over units (prompt segments + sentences). fig55: entropy against t, one panel per prompt (median and quartiles,
right and wrong traces). fig56: the segment masses against t (medians, stacked). Across traces at the answer row and at the
onset of the conclusion unit: data/screen/medxpertqa_attention_answer.csv and _summary.csv (within-prompt Spearman with length,
confidence, correctness and the closed jump across the conclusion unit); fig59: the mass on the conclusion unit at the answer
row against relative length, one panel per prompt (the dilution test: does the verdict get a smaller share in long traces, and
does the jump follow that share within prompt).

Every trace's full sequence (prompt + its generated tokens up to the answer letter) goes through Qwen3-8B once (bf16, SDPA for the
forward pass). In every layer the attention weights of a few query rows are recomputed from the queries and keys of that layer
(after q/k norm and RoPE) and summarised; the rows:
    "0"      the token before the first thinking sentence (the empty thinking)
    "s"      the token before every later sentence start (the state after sentence k - 1 is complete), k = 1..n-1
    "e"      the last thinking token: the state that wrote </think>
    "a"      the token before the answer letter (after "</think>\\n\\n"): the state that writes the letter
The onset of the conclusion unit (data/screen/medxpertqa_last_sentence_main.csv, the model's verdict sentence or the last two) is
one of the "0"/"s" rows and is marked in the meta file (onset_row). Without that file the last sentence start is used.

Per row, layer and head: the entropy of the attention over the keys (nats) and the mass on the segments of the context:
    0 sink (the first token), 1 template (chat-template tokens), 2 stem (the case text), 3 option_A, 4 option_B, 5 instruction,
    6 think_open ("<think>\\n"), 7 thinking (sentences before the conclusion unit), 8 conclusion (the conclusion unit),
    9 close ("</think>" and what follows it)
Per row and layer (mean over heads): the mass on every sentence of the thinking (the conclusion unit's sentences included).
At the three key rows ("0", "e", "a") the per-head segment masses are kept; at the "s" rows only their mean over heads, to keep
the files small. For the key rows the most attended token positions are kept too: per layer the top 10 of the head-mean
distribution ("<i>/top_idx" [3, L, 10], "<i>/top_val"), and the top 30 of the distribution averaged over all layers ("<i>/topall_idx"
[3, 30], "<i>/topall_val"); `show` prints them as words with their sentence and segment. The raw letter logits at the "a" row are stored to check the position against the trace's own logp_A/logp_B.

Output per run: <run>/readouts/attention-SS.npz (shards of --shard traces; keys "<i>/ent" [R, L, H] float16, "<i>/mass" [3, L, H, 10]
float16 for the key rows, "<i>/mass_mean" [R, L, 10] float16, "<i>/sent" [R, L, n_sent] float16) and attention-SS.jsonl (one line per
trace: sample_id, item_id, i, rows (absolute positions), kinds, ks, onset_row, n_prompt, n_sent, sentence_starts, segments (token
ranges), z_A, z_B, logp_A, logp_B at the "a" row). Shards already on disk are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import INSTRUCTION, MODEL, MODEL_NAME, MODEL_REVISION, ROOT, Tokens, iter_traces, load_tasks  # noqa: E402
from readout import sentence_boundaries  # noqa: E402
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)   # the proportion grid of length_accuracy.truncation (scipy-free here: the pod has no scipy)

SEGMENTS = ["sink", "template", "stem", "option_A", "option_B", "instruction", "think_open", "thinking", "conclusion", "close"]
TOPK, TOPK_ALL = 10, 30
KEY_KINDS = ("0", "e", "a")
POSITIONS = ROOT / "data/screen/medxpertqa_last_sentence_main.csv"
OUT = ROOT / "data/screen"
FIG = ROOT / "figures/formal_analysis"


# ------------------------------------------------------------------------------------------------------------- positions

def prompt_segments(tok, item: dict, thinking: bool = True) -> tuple[list[int], np.ndarray]:
    """Prompt token ids and the segment id of every prompt token (sink / template / stem / option_A / option_B / instruction;
    without thinking the empty thinking block the chat template appends is think_open ("<think>\\n\\n") and close ("</think>\\n\\n"))."""
    messages = item["messages"]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    content = messages[0]["content"]
    question = content[: -len("\n\n" + INSTRUCTION)]
    assert content.endswith(INSTRUCTION) and "\nA. " in question, item["item_id"]
    stem = question[: question.rindex("\nA. ") + 1]       # the case text, up to and including the newline before "A. "
    a_line = question[len(stem): question.rindex("\nB. ") + 1]
    b_line = question[len(stem) + len(a_line):]
    c0 = text.index(content)
    spans = [(c0, c0 + len(stem), 2), (c0 + len(stem), c0 + len(stem) + len(a_line), 3),
             (c0 + len(stem) + len(a_line), c0 + len(question), 4), (c0 + len(question), c0 + len(content), 5)]
    seg = np.ones(len(ids), dtype=np.int64)   # template
    for t, (s, _e) in enumerate(offsets):
        for lo, hi, sid in spans:
            if lo <= s < hi:
                seg[t] = sid
                break
    seg[0] = 0   # the sink
    if not thinking:
        t0, t1 = text.rindex("<think>"), text.rindex("</think>")
        for t, (s, _e) in enumerate(offsets):
            if s >= t0:
                seg[t] = 9 if s >= t1 else 6
    return ids, seg


def trace_rows(r: dict, tokens: Tokens, texts: list[str], n_prompt: int, onset_of: dict[str, int] | None) -> dict:
    gen = r["token_ids"]
    end = gen.index(tokens.think_end)
    start = gen.index(tokens.think) + 1 if tokens.think in gen[:end] else 0
    while start < end and not texts[gen[start]].strip():
        start += 1
    b = sentence_boundaries(gen, start, end, texts)
    starts = [start] + b                                   # sentence k (1-based) spans [starts[k-1], starts[k]); the last ends at `end`
    onset = onset_of.get(r["sample_id"]) if onset_of else None
    if onset is None or onset not in starts:
        onset = starts[-1]
    answer_pos = r["answer_pos"]
    rows = [n_prompt + start - 1] + [n_prompt + p - 1 for p in b] + [n_prompt + end - 1, n_prompt + answer_pos - 1]
    kinds = ["0"] + ["s"] * len(b) + ["e", "a"]
    ks = [0] + list(range(1, len(b) + 1)) + [len(starts), len(starts)]
    seg = np.full(answer_pos, 9, dtype=np.int64)            # close: </think> and after
    seg[:start] = 6                                        # <think>\n
    seg[start:onset] = 7
    seg[onset:end] = 8
    sent = np.full(answer_pos, -1, dtype=np.int64)
    for k in range(len(starts)):
        lo, hi = starts[k], (starts[k + 1] if k + 1 < len(starts) else end)
        sent[lo:hi] = k
    return {"rows": rows, "kinds": kinds, "ks": ks, "onset_row": starts.index(onset), "n_sent": len(starts), "sentence_starts": starts,
            "think_start": start, "think_end": end, "answer_pos": answer_pos, "gen": gen[:answer_pos], "seg_gen": seg, "sent_gen": sent}


def load_onsets(path: Path) -> dict[str, int] | None:
    if not path.exists():
        print(f"no {path}: the onset of the conclusion unit = the last sentence start", flush=True)
        return None
    d = pd.read_csv(path, usecols=["sample_id", "onset"])
    return dict(zip(d.sample_id, d.onset.astype(int)))


# --------------------------------------------------------------------------------------------------------------- capture

class Capture:
    """Recomputes the attention of the selected query rows in every layer and keeps the summaries."""

    def __init__(self) -> None:
        self.active = False
        self.layers: dict[int, tuple] = {}

    def set(self, rows, seg, sent, n_sent: int, key_rows: list[int]) -> None:
        import torch
        self.rows = torch.as_tensor(rows, dtype=torch.long)
        self.seg = torch.as_tensor(seg, dtype=torch.long)
        self.sent = torch.as_tensor(np.where(sent < 0, n_sent, sent), dtype=torch.long)
        self.n_sent = n_sent
        self.key_rows = key_rows
        self.layers = {}
        self.layer_mean: dict[int, Any] = {}
        self.active = True

    def record(self, layer: int, query, key, scaling: float | None) -> None:
        import torch
        dev = query.device
        rows = self.rows.to(dev)
        q = query[0, :, rows, :].float()                                   # [H, R, d]
        k = key[0].float()                                                 # [Hkv, n, d]
        H, Hkv, n = q.shape[0], k.shape[0], k.shape[1]
        if Hkv != H:
            k = k.repeat_interleave(H // Hkv, dim=0)
        if scaling is None:
            scaling = q.shape[-1] ** -0.5
        s = torch.einsum("hrd,hnd->hrn", q, k) * scaling                   # [H, R, n]
        causal = torch.arange(n, device=dev)[None, :] > rows[:, None]      # keys after the query row
        s = s.masked_fill(causal[None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        ent = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)                 # [H, R]
        seg = self.seg.to(dev)
        mass = torch.zeros(H, p.shape[1], len(SEGMENTS), device=dev).index_add_(2, seg, p)   # [H, R, S]
        sent = self.sent.to(dev)
        sm = torch.zeros(H, p.shape[1], self.n_sent + 1, device=dev).index_add_(2, sent, p)[..., : self.n_sent].mean(0)   # [R, n_sent]
        self.layers[layer] = (ent.T.cpu(), mass[:, self.key_rows, :].permute(1, 0, 2).cpu(), mass.mean(0).cpu(), sm.cpu())
        self.layer_mean[layer] = p.mean(0)[self.key_rows].cpu()                           # [3, n]: head-mean distribution of the key rows

    def record_module(self, module, query, key, value, scaling: float | None) -> None:
        self.record(module.layer_idx, query, key, scaling)

    def arrays(self, n_layers: int) -> dict[str, np.ndarray]:
        import torch
        got = [self.layers[i] for i in range(n_layers)]
        lm = torch.stack([self.layer_mean[i] for i in range(n_layers)], 1)                         # [3, L, n]
        v, ix = lm.topk(min(TOPK, lm.shape[-1]), dim=-1)
        v2, ix2 = lm.mean(1).topk(min(TOPK_ALL, lm.shape[-1]), dim=-1)
        return {"ent": torch.stack([g[0] for g in got], 1).numpy().astype(np.float16),            # [R, L, H]
                "mass": torch.stack([g[1] for g in got], 1).numpy().astype(np.float16),           # [3, L, H, S]
                "mass_mean": torch.stack([g[2] for g in got], 1).numpy().astype(np.float16),      # [R, L, S]
                "sent": torch.stack([g[3] for g in got], 1).numpy().astype(np.float16),           # [R, L, n_sent]
                "top_idx": ix.numpy().astype(np.int32), "top_val": v.numpy().astype(np.float16),   # [3, L, TOPK]
                "topall_idx": ix2.numpy().astype(np.int32), "topall_val": v2.numpy().astype(np.float16)}   # [3, TOPK_ALL]


CAP = Capture()


def install_attention(cap=None) -> str:
    """Registers the capturing attention function (recording into `cap`, default CAP) with transformers and returns its name."""
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    cap = cap or CAP

    def capture_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        out, _ = sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        if cap.active:
            cap.record_module(module, query, key, value, scaling)
        return out, None

    AttentionInterface.register("capture", capture_attention)
    return "capture"


def load_model(path: str, cap=None):
    """bf16 on the GPU; loaded straight onto it where transformers can (Qwen3-32B's 61 GiB), else through the CPU."""
    import torch
    from transformers import AutoModelForCausalLM
    name = install_attention(cap)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=name, device_map="cuda")
    except (ImportError, ValueError, TypeError):
        try:
            model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=name)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation=name)
    model.to("cuda").eval()
    return model


# ------------------------------------------------------------------------------------------------------------------- run

def run(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    tokens = Tokens(tok, ("A", "B"))
    texts = [tok.decode([i]) for i in range(len(tok))]
    items = {it["item_id"]: it for it in load_tasks(args.tasks)}
    onset_of = load_onsets(args.positions)
    out = args.run / "readouts"
    out.mkdir(exist_ok=True)
    letter_ids = {L: tok(L, add_special_tokens=False)["input_ids"][0] for L in ("A", "B")}

    traces = [r for r in iter_traces(args.run, ("sample_id", "item_id", "token_ids", "answer_pos", "think_tokens", "logp_A", "logp_B"))
              if r["think_tokens"] is not None and r["answer_pos"] is not None]
    if args.limit:
        traces = traces[: args.limit]
    shards = [traces[i: i + args.shard] for i in range(0, len(traces), args.shard)]
    print(f"{len(traces)} traces in {len(shards)} shards of {args.shard}; dry={args.dry}", flush=True)

    model = None if args.dry else load_model(path)
    n_layers = 36 if args.dry else model.config.num_hidden_layers
    n_heads = 32 if args.dry else model.config.num_attention_heads
    prompts: dict[str, tuple[list[int], np.ndarray]] = {}
    t0, done_traces = time.time(), 0
    for si, shard in enumerate(shards):
        npz_path, meta_path = out / f"attention-{si:02d}.npz", out / f"attention-{si:02d}.jsonl"
        if npz_path.exists() and meta_path.exists():
            continue
        arrays, metas = {}, []
        for i, r in enumerate(shard):
            if r["item_id"] not in prompts:
                prompts[r["item_id"]] = prompt_segments(tok, items[r["item_id"]])
            p_ids, p_seg = prompts[r["item_id"]]
            tr = trace_rows(r, tokens, texts, len(p_ids), onset_of)
            ids = p_ids + tr["gen"]
            seg = np.concatenate([p_seg, tr["seg_gen"]])
            sent = np.concatenate([np.full(len(p_ids), -1), tr["sent_gen"]])
            key_rows = [tr["kinds"].index(k) for k in KEY_KINDS]
            R = len(tr["rows"])
            if args.dry:
                rng = np.random.default_rng(i)
                arr = {"ent": rng.uniform(0.5, 6, (R, n_layers, n_heads)).astype(np.float16),
                       "mass": rng.dirichlet(np.ones(len(SEGMENTS)), (3, n_layers, n_heads)).astype(np.float16),
                       "mass_mean": rng.dirichlet(np.ones(len(SEGMENTS)), (R, n_layers)).astype(np.float16),
                       "sent": rng.dirichlet(np.ones(tr["n_sent"]), (R, n_layers)).astype(np.float16)}
                n_tot = len(ids)
                arr["top_idx"] = rng.integers(0, n_tot, (3, n_layers, TOPK)).astype(np.int32)
                arr["top_val"] = np.sort(rng.dirichlet(np.ones(TOPK), (3, n_layers)), -1)[..., ::-1].astype(np.float16)
                arr["topall_idx"] = rng.integers(0, n_tot, (3, TOPK_ALL)).astype(np.int32)
                arr["topall_val"] = np.sort(rng.dirichlet(np.ones(TOPK_ALL), 3), -1)[..., ::-1].astype(np.float16)
                z_A = z_B = float("nan")
            else:
                import torch
                CAP.set(tr["rows"], seg, sent, tr["n_sent"], key_rows)
                with torch.inference_mode():
                    x = torch.as_tensor([ids], device="cuda")
                    try:
                        o = model(input_ids=x, use_cache=False, logits_to_keep=1)
                    except TypeError:
                        o = model(input_ids=x, use_cache=False)
                CAP.active = False
                arr = CAP.arrays(n_layers)
                lg = o.logits[0, -1].float()
                z_A, z_B = lg[letter_ids["A"]].item(), lg[letter_ids["B"]].item()
                lsm = torch.log_softmax(lg, -1)
                lp_A, lp_B = lsm[letter_ids["A"]].item(), lsm[letter_ids["B"]].item()
                del o, lg, lsm
            for k, v in arr.items():
                arrays[f"{i}/{k}"] = v
            segs = {name: [int(a), int(b)] for name, (a, b) in zip(SEGMENTS, segment_ranges(seg))}
            metas.append({"sample_id": r["sample_id"], "item_id": r["item_id"], "i": i, "rows": tr["rows"], "kinds": tr["kinds"], "ks": tr["ks"],
                          "onset_row": tr["onset_row"], "n_prompt": len(p_ids), "n_total": len(ids), "n_sent": tr["n_sent"],
                          "sentence_starts": tr["sentence_starts"], "think_start": tr["think_start"], "think_end": tr["think_end"],
                          "answer_pos": tr["answer_pos"], "segments": segs, "z_A": z_A, "z_B": z_B,
                          "logp_A": (lp_A if not args.dry else None), "logp_B": (lp_B if not args.dry else None),
                          "trace_logp_A": r["logp_A"], "trace_logp_B": r["logp_B"]})
            done_traces += 1
            if done_traces % args.log_every == 0:
                print(f"{done_traces}/{len(traces)} traces, {time.time() - t0:.0f}s", flush=True)
        tmp = npz_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.rename(npz_path)
        meta_path.write_text("".join(json.dumps(m) + "\n" for m in metas), encoding="utf-8")
    print(f"done: {done_traces} traces in {time.time() - t0:.0f}s -> {out}", flush=True)


def segment_ranges(seg: np.ndarray) -> list[tuple[int, int]]:
    """First and one-past-last token of every segment ((0, 0) when absent; the template is not contiguous, it surrounds the user text)."""
    out = []
    for sid in range(len(SEGMENTS)):
        w = np.flatnonzero(seg == sid)
        out.append((int(w[0]), int(w[-1]) + 1) if len(w) else (0, 0))
    return out


# ------------------------------------------------------------------------------------------- study104: the answer row

STUDY104 = ROOT / "data/tasks/study104.csv"
RUNS = ROOT / "runs/medxpertqa"
FIELDS104 = ("sample_id", "item_id", "token_ids", "answer_pos", "think_tokens", "answer", "correct", "logp_A", "logp_B")


class AnswerCapture:
    """The attention of one query row (the answer row) in every layer: raw, and weighted by |W_O^h v_j|."""

    def __init__(self) -> None:
        self.active = False
        self.gram: dict[int, Any] = {}   # layer -> [H, d, d] = W_O^h^T W_O^h, so that |W_O^h v|^2 = v^T G v

    def set(self, row: int, seg, sent, n_sent: int) -> None:
        import torch
        self.row = row
        self.seg = torch.as_tensor(seg[: row + 1], dtype=torch.long)
        self.sent = torch.as_tensor(np.where(sent[: row + 1] < 0, n_sent, sent[: row + 1]), dtype=torch.long)
        self.n_sent = n_sent
        self.layers: dict[int, dict] = {}
        self.acc = self.vacc = None
        self.active = True

    def gram_of(self, module, H: int, d: int):
        import torch
        g = self.gram.get(module.layer_idx)
        if g is None:
            w = module.o_proj.weight.float().view(-1, H, d)                # [hidden, H, d]: o_proj reads the heads one after another
            g = self.gram[module.layer_idx] = torch.einsum("khd,khe->hde", w, w)
        return g

    def record_module(self, module, query, key, value, scaling: float | None) -> None:
        import torch
        dev, n = query.device, self.row + 1
        q = query[0, :, self.row, :].float()                               # [H, d]
        k, v = key[0, :, :n].float(), value[0, :, :n].float()              # [Hkv, n, d]
        H, Hkv, d = q.shape[0], k.shape[0], q.shape[-1]
        g = H // Hkv                                                       # query head h reads key/value head h // g
        if scaling is None:
            scaling = d ** -0.5
        p = torch.softmax(torch.einsum("kgd,knd->kgn", q.view(Hkv, g, d), k).reshape(H, n) * scaling, dim=-1)   # [H, n]
        G = self.gram_of(module, H, d).view(Hkv, g, d, d)
        vn = torch.einsum("knd,kgde,kne->kgn", v, G, v).reshape(H, n).clamp_min(0).sqrt()                       # |W_O^h v_j|
        w = p * vn
        vtot = w.sum(-1)
        w = w / vtot[:, None].clamp_min(1e-30)
        seg, sent, S = self.seg.to(dev), self.sent.to(dev), len(SEGMENTS)
        ent = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)
        mass = torch.zeros(H, S, device=dev).index_add_(1, seg, p)
        vmass = torch.zeros(H, S, device=dev).index_add_(1, seg, w)
        p_ex, w_ex = p.clone(), w.clone()                                  # the first token left out, the rest renormalised
        p_ex[:, 0] = 0
        w_ex[:, 0] = 0
        p_ex = p_ex / p_ex.sum(-1, keepdim=True).clamp_min(1e-30)
        w_ex = w_ex / w_ex.sum(-1, keepdim=True).clamp_min(1e-30)
        sm = torch.zeros(H, self.n_sent + 1, device=dev).index_add_(1, sent, p_ex)[:, : self.n_sent].mean(0)
        vsm = torch.zeros(H, self.n_sent + 1, device=dev).index_add_(1, sent, w_ex)[:, : self.n_sent].mean(0)
        lm, vlm = p_ex.mean(0), w_ex.mean(0)
        tv, ti = lm.topk(min(TOPK, n))
        self.acc = lm if self.acc is None else self.acc + lm
        self.vacc = vlm if self.vacc is None else self.vacc + vlm
        self.layers[module.layer_idx] = {"ent": ent, "mass": mass, "vmass": vmass, "vtot": vtot, "sent": sm, "vsent": vsm,
                                         "top_idx": ti, "top_val": tv}

    def arrays(self, n_layers: int) -> dict[str, np.ndarray]:
        import torch
        out = {}
        for key in self.layers[0]:
            x = torch.stack([self.layers[i][key] for i in range(n_layers)]).cpu().numpy()
            out[key] = x.astype(np.int32 if key == "top_idx" else np.float32 if key == "vtot" else np.float16)
        for name, acc in (("topall", self.acc), ("vtopall", self.vacc)):
            v, ix = (acc / n_layers).topk(min(TOPK_ALL, acc.shape[0]))
            out[f"{name}_idx"], out[f"{name}_val"] = ix.cpu().numpy().astype(np.int32), v.cpu().numpy().astype(np.float16)
        return out


ANSWER_CAP = AnswerCapture()


def conclusion_onsets(run: Path) -> dict[str, int]:
    """sample_id -> first generated token of the conclusion unit (src/conclusion_readout.py)."""
    out = {}
    for f in sorted((run / "readouts").glob("conclusion-*.jsonl")):
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            out[r["sample_id"]] = int(r["onset"])
    return out


def answer_context(r: dict, tokens: Tokens, texts: list[str], p_ids: list[int], p_seg: np.ndarray, onset_of: dict[str, int],
                   thinking: bool) -> dict:
    """Token ids up to the answer row, the segment and sentence of every token, and the positions the meta line keeps."""
    if not thinking:
        gen = r["token_ids"][: r["answer_pos"]]                          # blanks before the letter, if any: close
        ids = p_ids + gen
        return {"ids": ids, "seg": np.concatenate([p_seg, np.full(len(gen), 9, dtype=np.int64)]), "sent": np.full(len(ids), -1),
                "n_sent": 0, "row": len(ids) - 1, "think_start": None, "think_end": None, "onset": None, "sentence_starts": [],
                "onset_found": None}
    tr = trace_rows(r, tokens, texts, len(p_ids), onset_of)
    ids = p_ids + tr["gen"]
    assert tr["rows"][-1] == len(ids) - 1, r["sample_id"]
    onset = tr["sentence_starts"][tr["onset_row"]]
    return {"ids": ids, "seg": np.concatenate([p_seg, tr["seg_gen"]]), "sent": np.concatenate([np.full(len(p_ids), -1), tr["sent_gen"]]),
            "n_sent": tr["n_sent"], "row": len(ids) - 1, "think_start": tr["think_start"], "think_end": tr["think_end"], "onset": onset,
            "sentence_starts": tr["sentence_starts"], "onset_found": onset_of.get(r["sample_id"]) == onset}


def dry_show(ctx: dict, tok, sample_id: str, next_token: int | None) -> None:
    ids, seg = ctx["ids"], ctx["seg"]
    print(f"=== {sample_id}: {len(ids)} tokens, {ctx['n_sent']} sentences, onset {ctx['onset']}, think_end {ctx['think_end']}")
    for s, name in enumerate(SEGMENTS):
        w = np.flatnonzero(seg == s)
        if len(w):
            t = tok.decode([ids[j] for j in w])
            print(f"  {name:12s} {len(w):5d} tokens  {t[:70]!r} ... {t[-50:]!r}")
    print(f"  answer row {ctx['row']}: {tok.decode([ids[ctx['row']]])!r}, next token {tok.decode([next_token]) if next_token is not None else None!r}")


def model_dir() -> str:
    """The local snapshot of MODEL; one holding only the tokenizer and config (enough for --dry) is accepted."""
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    try:
        return snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True)
    except Exception:
        cfg = try_to_load_from_cache(MODEL, "config.json", revision=MODEL_REVISION)
        if not isinstance(cfg, str):
            raise
        return str(Path(cfg).parent)


def run104(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoTokenizer
    from generate import TASKS, check_run_model

    check_run_model(args.run)
    if args.write_dry and (args.run.resolve() == RUNS or RUNS in args.run.resolve().parents):
        raise SystemExit("--write-dry writes random numbers: only into a copy of a run outside runs/")
    thinking = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))["mode"] == "think"
    path = model_dir()
    tok = AutoTokenizer.from_pretrained(path)
    cfg = AutoConfig.from_pretrained(path)
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    tokens = Tokens(tok, ("A", "B"))
    texts = [tok.decode([i]) for i in range(len(tok))]
    keep = set(pd.read_csv(args.questions).item_id)
    items = {it["item_id"]: it for it in load_tasks(TASKS) if it["item_id"] in keep}
    onset_of = conclusion_onsets(args.run) if thinking else {}
    traces = [r for r in iter_traces(args.run, FIELDS104) if r["item_id"] in keep and r["answer_pos"] is not None]
    if args.limit:
        traces = traces[: args.limit]
    out = args.run / "readouts"
    out.mkdir(exist_ok=True)
    shards = [traces[i: i + args.shard] for i in range(0, len(traces), args.shard)]
    print(f"{MODEL_NAME} {'thinking' if thinking else 'no thinking'}: {len(traces)} traces of {len({r['item_id'] for r in traces})} "
          f"questions in {len(shards)} shards of {args.shard}; {n_layers} layers x {n_heads} heads; dry={args.dry}", flush=True)

    model = None if args.dry else load_model(path, ANSWER_CAP)
    prompts: dict[str, tuple[list[int], np.ndarray]] = {}
    t0, done_traces, lengths, not_found = time.time(), 0, [], 0
    for si, shard in enumerate(shards):
        npz_path, meta_path = out / f"answer-attention-{si:02d}.npz", out / f"answer-attention-{si:02d}.jsonl"
        if npz_path.exists() and meta_path.exists() and not args.dry:
            continue
        arrays, metas = {}, []
        for i, r in enumerate(shard):
            iid = r["item_id"]
            if iid not in prompts:
                p_ids, p_seg = prompt_segments(tok, items[iid], thinking)
                assert p_ids == tokens.chat_ids(items[iid]["messages"], thinking), iid
                prompts[iid] = (p_ids, p_seg)
            p_ids, p_seg = prompts[iid]
            ctx = answer_context(r, tokens, texts, p_ids, p_seg, onset_of, thinking)
            ids, row = ctx["ids"], ctx["row"]
            lengths.append(len(ids))
            not_found += ctx["onset_found"] is False
            if args.dry:
                if done_traces < args.show:
                    dry_show(ctx, tok, r["sample_id"], r["token_ids"][r["answer_pos"]])
                rng = np.random.default_rng(done_traces)
                L, H, S, ns = n_layers, n_heads, len(SEGMENTS), ctx["n_sent"]
                arr = {"ent": rng.uniform(0.5, 6, (L, H)), "mass": rng.dirichlet(np.ones(S), (L, H)), "vmass": rng.dirichlet(np.ones(S), (L, H)),
                       "vtot": rng.uniform(1, 10, (L, H)), "sent": rng.dirichlet(np.ones(max(ns, 1)), L)[:, :ns],
                       "vsent": rng.dirichlet(np.ones(max(ns, 1)), L)[:, :ns], "top_idx": rng.integers(1, row + 1, (L, TOPK)),
                       "top_val": rng.uniform(0, 0.1, (L, TOPK)), "topall_idx": rng.integers(1, row + 1, TOPK_ALL),
                       "topall_val": np.sort(rng.uniform(0, 0.1, TOPK_ALL))[::-1], "vtopall_idx": rng.integers(1, row + 1, TOPK_ALL),
                       "vtopall_val": np.sort(rng.uniform(0, 0.1, TOPK_ALL))[::-1]}
                arr = {k: v.astype(np.int32 if k.endswith("idx") else np.float16) for k, v in arr.items()}
                z_A = z_B = lp_A = lp_B = None
            else:
                import torch
                ANSWER_CAP.set(row, ctx["seg"], ctx["sent"], ctx["n_sent"])
                with torch.inference_mode():
                    x = torch.as_tensor([ids], device="cuda")
                    try:
                        o = model(input_ids=x, use_cache=False, logits_to_keep=1)
                    except TypeError:
                        o = model(input_ids=x, use_cache=False)
                ANSWER_CAP.active = False
                arr = ANSWER_CAP.arrays(n_layers)
                lg = o.logits[0, -1].float()
                lsm = torch.log_softmax(lg, -1)
                z_A, z_B = lg[tokens.options["A"]].item(), lg[tokens.options["B"]].item()
                lp_A, lp_B = lsm[tokens.options["A"]].item(), lsm[tokens.options["B"]].item()
                del o, lg, lsm
            for k, v in arr.items():
                arrays[f"{i}/{k}"] = v
            seg_counts = np.bincount(ctx["seg"][: row + 1], minlength=len(SEGMENTS)).tolist()
            metas.append({"sample_id": r["sample_id"], "item_id": iid, "model": MODEL_NAME, "mode": "think" if thinking else "nothink", "i": i,
                          "row": row, "n_prompt": len(p_ids), "answer_pos": r["answer_pos"], "think_tokens": r["think_tokens"],
                          "think_start": ctx["think_start"], "think_end": ctx["think_end"], "onset": ctx["onset"], "onset_found": ctx["onset_found"],
                          "n_sent": ctx["n_sent"], "sentence_starts": ctx["sentence_starts"],
                          "segments": {name: [int(a), int(b)] for name, (a, b) in zip(SEGMENTS, segment_ranges(ctx["seg"][: row + 1]))},
                          "seg_counts": seg_counts, "answer": r["answer"], "correct": r["correct"], "z_A": z_A, "z_B": z_B,
                          "logp_A": lp_A, "logp_B": lp_B, "trace_logp_A": r["logp_A"], "trace_logp_B": r["logp_B"], "dry": args.dry})
            done_traces += 1
            if done_traces % args.log_every == 0:
                print(f"{done_traces}/{len(traces)} traces, {time.time() - t0:.0f}s", flush=True)
        if args.dry and not args.write_dry:
            continue
        tmp = npz_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.rename(npz_path)
        meta_path.write_text("".join(json.dumps(m) + "\n" for m in metas), encoding="utf-8")
    if lengths:
        print(f"context length up to the answer row: median {int(np.median(lengths))}, max {max(lengths)}; "
              f"conclusion onset not a sentence start (last sentence used): {not_found}", flush=True)
    print(f"done: {done_traces} traces in {time.time() - t0:.0f}s -> {out}", flush=True)


SEG104 = ["sink", "template", "stem", "option_chosen", "option_other", "instruction", "think_open", "thinking", "conclusion", "close"]
BLOCKS = {"question": ("stem", "option_chosen", "option_other", "instruction"), "reasoning": ("thinking", "conclusion"),
          "format": ("template", "think_open", "close")}
THIRDS = ("early", "mid", "late")
VERDICT_TYPES = ("verdict_definitive", "verdict_hedged", "option_text")   # a sentence naming the conclusion (conclusion_count_test.py)
NOTHINK_RUN = {"qwen3-8b": "nothink-qwen3-8b", "qwen3-32b": "study104-nothink-qwen3-32b"}   # the 8B's first reactions: its screening run
ANSWER_KEYS = ("ent", "mass", "vmass", "sent", "vsent", "topall_idx", "topall_val", "vtopall_idx", "vtopall_val")


def study104_runs(root: Path, model: str) -> list[tuple[str, Path]]:
    return [(mode, root / f"{stem}-{o}") for mode, stem in (("think", f"study104-{model}"), ("nothink", NOTHINK_RUN[model])) for o in ("ab", "ba")]


def load_answer_shards(run: Path):
    for meta_path in sorted((run / "readouts").glob("answer-attention-*.jsonl")):
        npz = np.load(meta_path.with_suffix(".npz"))
        for line in meta_path.open(encoding="utf-8"):
            m = json.loads(line)
            yield m, {k: npz[f"{m['i']}/{k}"] for k in ANSWER_KEYS}


def option_texts(it: dict) -> tuple[str, str]:
    return (it["correct_text"], it["distractor_text"]) if it["label"] == "A" else (it["distractor_text"], it["correct_text"])


def place104(pos: int, m: dict, p_seg: np.ndarray, chosen: str) -> tuple[str, int]:
    """Where a key token sits (a SEG104 segment) and, inside the thinking, its sentence (0-based; -1 elsewhere)."""
    if pos < m["n_prompt"]:
        name = SEGMENTS[int(p_seg[pos])]
        return (("option_chosen" if name[-1] == chosen else "option_other") if name.startswith("option_") else name), -1
    g = pos - m["n_prompt"]
    if m["mode"] == "nothink" or g >= m["think_end"]:
        return "close", -1
    if g < m["think_start"]:
        return "think_open", -1
    k = int(np.searchsorted(m["sentence_starts"], g, side="right")) - 1
    return ("conclusion" if g >= m["onset"] else "thinking"), k


def without_sink(mass: np.ndarray) -> np.ndarray:
    """Per layer and head, the segment shares of the attention that does not go to the first token (column 0 set to 0). Divided by
    the sum of the other segments, not by 1 - the first token's share: that share is often 0.999 and float16 near 1 moves in 0.0005 steps."""
    ex = mass.copy()
    ex[..., 0] = 0
    return ex / ex.sum(-1, keepdims=True).clip(1e-12)


def nanmedian(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.median(x)) if len(x) else np.nan


def spearman(x, y) -> float:
    x, y = pd.Series(np.asarray(x, dtype=float)), pd.Series(np.asarray(y, dtype=float))
    return float(x.corr(y, method="spearman")) if x.nunique() > 1 and y.nunique() > 1 else np.nan


def analyze104(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from generate import TASKS
    from last_sentence_check import classify
    from stopping_rule import sentence_type   # analysis-only module (scipy)

    tok = AutoTokenizer.from_pretrained(snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True))   # the 8B's and 32B's are identical
    items = {it["item_id"]: it for it in load_tasks(TASKS)}
    keep = set(pd.read_csv(STUDY104).item_id)
    rows, sent_rows, top_rows, checks = [], [], [], []
    depth: dict[tuple, list[np.ndarray]] = {}
    heads: dict[str, list] = {}
    prompts: dict[tuple[str, str], tuple[list[int], np.ndarray]] = {}
    for model in args.models:
        for mode, run in study104_runs(args.runs_root, model):
            if not any((run / "readouts").glob("answer-attention-*.jsonl")):
                print(f"no answer-attention shards in {run}")
                continue
            gens = {t["sample_id"]: t["token_ids"] for t in iter_traces(run, ("sample_id", "item_id", "token_ids")) if t["item_id"] in keep}
            for m, a in load_answer_shards(run):
                if m.get("dry") and (args.runs_root == RUNS or args.out == OUT):
                    raise SystemExit(f"{run} holds dry-run shards (random numbers): read them only with --runs-root and --out outside the repo")
                sid, iid, chosen = m["sample_id"], m["item_id"], m["answer"]
                if iid not in keep or chosen not in ("A", "B"):
                    continue
                it = items[iid]
                if (iid, mode) not in prompts:
                    prompts[(iid, mode)] = prompt_segments(tok, it, mode == "think")
                p_ids, p_seg = prompts[(iid, mode)]
                ids = p_ids + gens[sid][: m["answer_pos"]]
                order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] if chosen == "A" else [0, 1, 2, 4, 3, 5, 6, 7, 8, 9]   # option_A/B -> chosen/other
                mass, vmass = a["mass"].astype(np.float32)[..., order], a["vmass"].astype(np.float32)[..., order]
                ex, vex = without_sink(mass), without_sink(vmass)
                L = mass.shape[0]
                thirds = np.array_split(np.arange(L), 3)
                counts = np.array(m["seg_counts"])[order]
                n_keys = m["row"] + 1
                d = {"model": model, "mode": mode, "sample_id": sid, "item_id": iid, "answer": chosen, "correct": bool(m["correct"]),
                     "think_tokens": m["think_tokens"], "n_keys": n_keys, "n_sent": m["n_sent"],
                     "sink": float(mass[..., 0].mean()), "v_sink": float(vmass[..., 0].mean())}
                for gname, ls in zip(THIRDS, thirds):
                    d[f"sink_{gname}"] = float(mass[ls, :, 0].mean())
                for s, name in enumerate(SEG104[1:], 1):
                    d[name] = float(ex[..., s].mean())
                    for gname, ls in zip(THIRDS, thirds):
                        d[f"{name}_{gname}"] = float(ex[ls, :, s].mean())
                    d[f"v_{name}"] = float(vex[..., s].mean())
                    d[f"n_{name}"] = int(counts[s])
                    d[f"enrich_{name}"] = d[name] / (counts[s] / (n_keys - 1)) if counts[s] else np.nan
                for b, names in BLOCKS.items():
                    d[b] = sum(d[x] for x in names)
                    d[f"v_{b}"] = sum(d[f"v_{x}"] for x in names)
                ent = a["ent"].astype(np.float32)
                d["ent_norm"] = float((ent / math.log(n_keys)).mean())
                d["eff_tokens"] = float(np.exp(ent).mean())
                types: list[str] = []
                if mode == "think":
                    gen, sm, vsm = gens[sid], a["sent"].astype(np.float32), a["vsent"].astype(np.float32)
                    starts = m["sentence_starts"]
                    ends = starts[1:] + [m["think_end"]]
                    earlier, other, prev = [], [], None
                    for k, (lo, hi) in enumerate(zip(starts, ends)):
                        text = tok.decode(gen[lo:hi]).strip()
                        typ = sentence_type(text, option_texts(it))
                        form = classify(text, prev, option_texts(it))["form"]
                        prev = text
                        types.append(typ)
                        inside = lo >= m["onset"]
                        sent_rows.append({"model": model, "sample_id": sid, "item_id": iid, "correct": d["correct"], "k": k, "n_sent": m["n_sent"],
                                          "k_from_end": m["n_sent"] - 1 - k, "n_tokens": hi - lo, "type": typ, "form": form, "in_conclusion": inside,
                                          "share": float(sm[:, k].mean()), "share_late": float(sm[thirds[2], k].mean()),
                                          "v_share": float(vsm[:, k].mean()), "v_share_late": float(vsm[thirds[2], k].mean())})
                        if not inside:
                            (earlier if typ in VERDICT_TYPES else other).append(k)
                    n_tok = sum(ends[k] - starts[k] for k in earlier)
                    k_on = starts.index(m["onset"])
                    d.update({"verdict_earlier": float(sm[:, earlier].sum(1).mean()), "v_verdict_earlier": float(vsm[:, earlier].sum(1).mean()),
                              "thinking_other": float(sm[:, other].sum(1).mean()), "v_thinking_other": float(vsm[:, other].sum(1).mean()),
                              "n_verdict_earlier": len(earlier), "n_tokens_verdict_earlier": n_tok,
                              "enrich_verdict_earlier": float(sm[:, earlier].sum(1).mean()) / (n_tok / (n_keys - 1)) if n_tok else np.nan,
                              "last_before_conclusion": float(sm[:, k_on - 1].mean()) if k_on else np.nan,
                              "first_sentence": float(sm[:, 0].mean())})
                rows.append(d)
                for weight, pre in (("raw", "topall"), ("value", "vtopall")):
                    for rank, (pos, val) in enumerate(zip(a[f"{pre}_idx"], a[f"{pre}_val"])):
                        place, k = place104(int(pos), m, p_seg, chosen)
                        top_rows.append({"model": model, "mode": mode, "sample_id": sid, "item_id": iid, "correct": d["correct"], "weight": weight,
                                         "rank": rank, "pos": int(pos), "token": tok.decode([ids[int(pos)]]), "place": place, "k": k,
                                         "k_from_end": m["n_sent"] - 1 - k if k >= 0 else -1, "type": types[k] if k >= 0 else "", "share": float(val)})
                lay, vlay = ex.mean(1), vex.mean(1)                  # [L, S]: per layer, mean over heads; column 0 = the first token's raw share
                lay[:, 0], vlay[:, 0] = mass[..., 0].mean(1), vmass[..., 0].mean(1)
                depth.setdefault((model, mode, iid, d["correct"]), []).append(np.concatenate([lay, vlay], 1))
                if mode == "think":
                    hx = ex.copy()
                    hx[..., 0] = mass[..., 0]
                    acc = heads.setdefault(model, [0, np.zeros_like(hx)])
                    acc[0] += 1
                    acc[1] += hx
                if m.get("logp_A") is not None:
                    checks.append({"model": model, "mode": mode, "trace_X": m["trace_logp_A"] - m["trace_logp_B"], "X": m["logp_A"] - m["logp_B"]})
    if not rows:
        raise SystemExit("no shards found")
    d = pd.DataFrame(rows)
    d.to_csv(args.out / "study104_attention_answer.csv", index=False)
    s = pd.DataFrame(sent_rows)
    s.to_csv(args.out / "study104_attention_sentences.csv.gz", index=False)
    pd.DataFrame(top_rows).to_csv(args.out / "study104_attention_top_tokens.csv.gz", index=False)
    cols = SEG104 + [f"v_{x}" for x in SEG104]
    dep = []
    for (model, mode, iid, correct), arrs in depth.items():
        med = np.median(np.stack(arrs), 0)
        for layer in range(med.shape[0]):
            dep.append({"model": model, "mode": mode, "item_id": iid, "correct": correct, "n": len(arrs), "layer": layer,
                        "depth": layer / (med.shape[0] - 1), **dict(zip(cols, med[layer].round(5)))})
    pd.DataFrame(dep).to_csv(args.out / "study104_attention_depth.csv", index=False)
    hd = []
    for model, (n, acc) in heads.items():
        mean = acc / n
        for layer in range(mean.shape[0]):
            for h in range(mean.shape[1]):
                hd.append({"model": model, "layer": layer, "head": h, "n": n, **dict(zip(SEG104, mean[layer, h].round(5)))})
    pd.DataFrame(hd).to_csv(args.out / "study104_attention_heads.csv", index=False)
    print(f"{len(d)} traces -> {args.out}/study104_attention_answer.csv, _sentences.csv.gz ({len(s)} sentences), _top_tokens.csv.gz, _depth.csv, _heads.csv")

    if "thinking_other" in d:
        th = d[d["mode"] == "think"]
        gap = (th.verdict_earlier + th.thinking_other - th.thinking).abs().max()
        conc = s[s.in_conclusion].groupby(["model", "sample_id"]).share.sum()       # sample ids repeat across models
        gap_c = (th.set_index(["model", "sample_id"]).conclusion - conc).abs().max()
        print(f"checks: |earlier verdicts + other thinking - thinking| max {gap:.4f}; |conclusion - its sentences| max {gap_c:.4f}")
    if checks:
        c = pd.DataFrame(checks)
        for (model, mode), g in c.groupby(["model", "mode"]):
            print(f"letter logits at the row, {model} {mode}: corr(trace X, recomputed X) {g.trace_X.corr(g.X):.4f}, "
                  f"median |diff| {(g.trace_X - g.X).abs().median():.3f} nats over {len(g)} traces")
    summary104(d, args.out)
    t = pd.DataFrame(top_rows)
    print("\nwhere the most attended token sits (first token left out), share of traces:")
    t0 = t[t["rank"] == 0]
    print(pd.crosstab([t0.model, t0["mode"], t0.weight], t0.place, normalize="index").round(2).to_string())
    print("the most attended token of the text (template, <think>, </think> and what follows it left out), share of traces:")
    t1 = t[~t.place.isin(["template", "think_open", "close"])].sort_values("rank").groupby(["model", "mode", "sample_id", "weight"]).head(1)
    print(pd.crosstab([t1.model, t1["mode"], t1.weight], t1.place, normalize="index").round(2).to_string())


def summary104(d: pd.DataFrame, out: Path) -> None:
    """Per model and measure: the median over questions of the per-question median with and without thinking, the within-question
    Spearman with the thinking length, and the within-question median of wrong minus right (questions with >= 5 of each)."""
    measures = (["sink", "v_sink", "ent_norm", "eff_tokens"] + SEG104[1:] + list(BLOCKS)
                + ["verdict_earlier", "thinking_other", "last_before_conclusion", "first_sentence"]
                + [f"{x}_{g}" for x in ("sink", "stem", "thinking", "conclusion", "close") for g in THIRDS]
                + [f"v_{x}" for x in SEG104[1:] + list(BLOCKS) + ["verdict_earlier", "thinking_other"]]
                + [f"enrich_{x}" for x in SEG104[1:] + ["verdict_earlier"]])
    measures = [c for c in measures if c in d]
    res = []
    for model, g in d.groupby("model"):
        th, nt = g[g["mode"] == "think"], g[g["mode"] == "nothink"]
        q_th, q_nt = th.groupby("item_id")[measures].median(), nt.groupby("item_id")[measures].median()
        for c in measures:
            r_len = [spearman(h.think_tokens, h[c]) for _, h in th.groupby("item_id")]
            rw = [h.loc[~h.correct, c].median() - h.loc[h.correct, c].median() for _, h in th.groupby("item_id")
                  if h.correct.sum() >= 5 and (~h.correct).sum() >= 5]
            res.append({"model": model, "measure": c, "think_median": q_th[c].median(), "think_q25": q_th[c].quantile(0.25),
                        "think_q75": q_th[c].quantile(0.75), "nothink_median": nanmedian(q_nt[c]),
                        "rho_len_median": nanmedian(r_len), "n_rho_negative": int(np.sum(np.array(r_len) < 0)),
                        "n_questions": len(r_len), "wrong_minus_right_median": nanmedian(rw), "n_questions_rw": len(rw)})
    s = pd.DataFrame(res)
    s.to_csv(out / "study104_attention_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nper model (medians over questions of per-question medians; rho = within-question Spearman with the thinking length):")
    print(s.round(3).to_string(index=False))


# --------------------------------------------------------------------------------------------------------------- analyze

LAYER_GROUPS = {"early": (0, 12), "mid": (12, 24), "late": (24, 36)}
MOVE_CLASS = {"restate finding": "restate", "inference (finding -> option)": "inference", "claim about option": "inference",
              "reconsider: other": "reconsider", "reconsider: alternative": "reconsider", "doubt (no content)": "doubt",
              "verdict": "verdict", "option text": "verdict"}


def load_runs(runs: list[Path]):
    for run in runs:
        for meta_path in sorted((run / "readouts").glob("attention-*.jsonl")):
            npz = np.load(meta_path.with_suffix(".npz"))
            for line in meta_path.open(encoding="utf-8"):
                m = json.loads(line)
                i = m["i"]
                arr = {k: npz[f"{i}/{k}"].astype(np.float32) for k in ("ent", "mass", "mass_mean", "sent")}
                for k in ("topall_idx", "topall_val", "top_idx", "top_val"):
                    if f"{i}/{k}" in npz.files:
                        arr[k] = npz[f"{i}/{k}"]
                yield m, arr


def token_place(pos: int, m: dict, p_seg: np.ndarray) -> str:
    """Where a key token sits: a prompt segment, think_open, close, or the sentence's place relative to the row (conclusion / last3 / older)."""
    n_prompt = m["n_prompt"]
    if pos < n_prompt:
        return SEGMENTS[int(p_seg[pos])]
    g = pos - n_prompt
    if g < m["think_start"]:
        return "think_open"
    if g >= m["think_end"]:
        return "close"
    starts = m["sentence_starts"]
    k = int(np.searchsorted(np.array(starts), g, side="right"))
    if g >= starts[m["onset_row"]]:
        return "conclusion"
    return "last3" if k >= m["n_sent"] - 3 else "older"


def analyze(args: argparse.Namespace) -> None:
    runs = args.runs or [ROOT / f"runs/medxpertqa/main-qwen3-8b-{o}" for o in ("ab", "ba")]
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from generate import TASKS
    tok = AutoTokenizer.from_pretrained(snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True))
    items = {it["item_id"]: it for it in load_tasks(TASKS)}
    p_segs: dict[str, np.ndarray] = {}
    la = pd.read_csv(OUT / "medxpertqa_length_accuracy_main.csv")
    info = la.set_index("sample_id")
    moves = pd.read_csv(OUT / "medxpertqa_moves.csv", usecols=["sample_id", "k", "move"])
    moves["cls"] = moves.move.map(MOVE_CLASS).fillna("other")
    move_of = {sid: dict(zip(g.k, g.cls)) for sid, g in moves.groupby("sample_id")}
    rows, within, checks = [], [], []
    for m, a in load_runs(runs):
        sid = m["sample_id"]
        if sid not in info.index:
            continue
        kinds, ks = m["kinds"], m["ks"]
        n_keys = np.array(m["rows"]) + 1
        key_idx = {k: kinds.index(k) for k in KEY_KINDS}
        key_idx["onset"] = m["onset_row"]
        cls_k = move_of.get(sid, {})
        if m["item_id"] not in p_segs:
            p_segs[m["item_id"]] = prompt_segments(tok, items[m["item_id"]])[1]
        for name, ri in key_idx.items():
            d = row_summary(a, ri, n_keys[ri], m, cls_k, {"sample_id": sid, "item_id": m["item_id"], "row": name, "pos": int(m["rows"][ri])})
            if name in KEY_KINDS and "topall_idx" in a:
                j = KEY_KINDS.index(name)
                ix, v = a["topall_idx"][j], a["topall_val"][j]
                d["top1_place"] = token_place(int(ix[0]), m, p_segs[m["item_id"]])
                d["top1_mass"] = float(v[0])
                d["top30_mass"] = float(v.sum())
                places = [token_place(int(x), m, p_segs[m["item_id"]]) for x in ix]
                for pl in ("sink", "stem", "option_A", "option_B", "conclusion", "last3", "older", "close"):
                    d[f"top30_mass_{pl}"] = float(sum(val for val, q in zip(v, places) if q == pl))
                content = [(q, float(val)) for q, val in zip(places, v) if q not in ("sink", "template", "close", "think_open", "instruction")]
                d["top1_content_place"], d["top1_content_mass"] = (content[0] if content else ("none", 0.0))
            rows.append(d)
        # within trace: the "0" and "s" rows at their proportion, "e" at 1
        span = max(m["think_end"] - m["think_start"], 1)
        prop = np.array([0.0 if k == "0" else 1.0 if k == "e" else (m["sentence_starts"][kk] - m["think_start"]) / span
                         for k, kk in zip(kinds, ks) if k in ("0", "s", "e")])
        idx = [i for i, k in enumerate(kinds) if k in ("0", "s", "e")]
        for t in GRID:
            j = int(np.searchsorted(prop, t + 1e-9, side="right") - 1)
            ri = idx[max(j, 0)]
            within.append(row_summary(a, ri, n_keys[ri], m, cls_k, {"sample_id": sid, "item_id": m["item_id"], "t": t, "k": int(ks[ri]), "pos": int(m["rows"][ri])}, recency=True))
        if m.get("logp_A") is not None:
            checks.append({"sample_id": sid, "trace_X": m["trace_logp_A"] - m["trace_logp_B"], "recomputed_X": m["logp_A"] - m["logp_B"]})
    keep = ["sample_id", "pair_id", "order", "correct", "think_tokens", "rel", "quintile", "gap", "conf", "n_sentences", "nothink_correct"]
    d = pd.DataFrame(rows).merge(la[keep], on="sample_id")
    jumps = pd.read_csv(OUT / "medxpertqa_last_sentence_main.csv", usecols=["sample_id", "jump_closed", "y_closed_onset", "y_closed_end"])
    d = d.merge(jumps, on="sample_id", how="left")
    lt = OUT / "medxpertqa_letter_sentence.csv"
    if lt.exists():
        d = d.merge(pd.read_csv(lt, usecols=["sample_id", "d_therefore_own", "none", "therefore_own"]), on="sample_id", how="left")
    d.to_csv(OUT / "medxpertqa_attention_answer.csv", index=False)
    w = pd.DataFrame(within).merge(la[keep], on="sample_id")
    w.to_csv(OUT / "medxpertqa_attention_within.csv", index=False)
    print(f"{d.sample_id.nunique()} traces: {OUT / 'medxpertqa_attention_answer.csv'} ({d.row.nunique()} rows per trace), {OUT / 'medxpertqa_attention_within.csv'} ({len(GRID)} proportions per trace)")
    if checks:
        c = pd.DataFrame(checks)
        print(f"position check at the 'a' row: corr(trace X, recomputed X) = {c.trace_X.corr(c.recomputed_X):.4f}, "
              f"median |diff| = {(c.trace_X - c.recomputed_X).abs().median():.3f} nats over {len(c)} traces")
    summary(d)
    within_summary(w)
    fig_entropy(w)
    fig_mass(w)
    fig_cross(d)


def row_summary(a: dict, ri: int, n_keys: int, m: dict, cls_k: dict, base: dict, recency: bool = False) -> dict:
    ent = a["ent"][ri]                                   # [L, H]
    mm = a["mass_mean"][ri]                              # [L, S]
    sm = a["sent"][ri]                                   # [L, n_sent]
    hn = ent / math.log(n_keys)
    d = dict(base)
    d.update({"n_keys": int(n_keys), "ent_mean": float(ent.mean()), "ent_norm_mean": float(hn.mean()), "eff_tokens_mean": float(np.exp(ent).mean()),
              "ent_norm_median_head": float(np.median(hn))})
    for g, (lo, hi) in LAYER_GROUPS.items():
        d[f"ent_norm_{g}"] = float(hn[lo:hi].mean())
    for s, seg in enumerate(SEGMENTS):
        d[f"mass_{seg}"] = float(mm[:, s].mean())
        for g, (lo, hi) in LAYER_GROUPS.items():
            d[f"mass_{seg}_{g}"] = float(mm[lo:hi, s].mean())
    d["mass_evidence"] = d["mass_stem"] + d["mass_option_A"] + d["mass_option_B"]
    d["mass_own_text"] = d["mass_thinking"] + d["mass_conclusion"]
    units = np.concatenate([mm[:, :7], sm], axis=1)       # prompt segments + sentences (only those before the row carry mass)
    units = units / units.sum(1, keepdims=True).clip(1e-9)
    live = int((sm.mean(0) > 1e-6).sum()) + 7
    d["ent_units_norm"] = float((-(units * np.log(units.clip(1e-30))).sum(1) / math.log(max(live, 2))).mean())
    for c in ("restate", "inference", "reconsider", "doubt", "verdict", "other"):
        ks = [k for k in range(1, m["n_sent"] + 1) if cls_k.get(k) == c]
        d[f"mass_sent_{c}"] = float(sm[:, [k - 1 for k in ks]].sum(1).mean()) if ks else 0.0
    d["mass_first_sentence"] = float(sm[:, 0].mean())
    if recency:
        k = base["k"]                                    # sentences completed at this row
        d["mass_last1"] = float(sm[:, k - 1].mean()) if k >= 1 else 0.0
        d["mass_last3"] = float(sm[:, max(k - 3, 0): k].sum(1).mean()) if k >= 1 else 0.0
        d["mass_older"] = float(sm[:, : max(k - 3, 0)].sum(1).mean()) if k > 3 else 0.0
    else:
        d["mass_last_sentence"] = float(sm[:, -1].mean())
    return d


def summary(d: pd.DataFrame) -> None:
    from length_accuracy import rho
    out = []
    for row, g in d.groupby("row"):
        for col in ["ent_norm_mean", "ent_units_norm", "eff_tokens_mean", "mass_sink", "mass_evidence", "mass_own_text", "mass_conclusion", "mass_close",
                    "mass_sent_verdict", "mass_sent_inference", "mass_sent_restate", "mass_sent_reconsider", "mass_sent_doubt", "mass_last_sentence"]:
            r_len = [rho(h.think_tokens, h[col]) for _, h in g.groupby("item_id")]
            r_conf = [rho(h.conf, h[col]) for _, h in g.groupby("item_id")]
            r_cor = [rho(h.correct.astype(int), h[col]) for _, h in g.groupby("item_id") if 0 < h.correct.mean() < 1]
            r_jump = [rho(h.jump_closed, h[col]) for _, h in g.groupby("item_id")]
            r_jt = [rho(h.d_therefore_own, h[col]) for _, h in g.groupby("item_id")] if "d_therefore_own" in g else [np.nan]
            out.append({"row": row, "measure": col, "median": g[col].median(), "rho_len_median": np.nanmedian(r_len), "rho_conf_median": np.nanmedian(r_conf),
                        "rho_correct_median": np.nanmedian(r_cor) if r_cor else np.nan, "rho_jump_own_median": np.nanmedian(r_jump),
                        "rho_jump_therefore_median": np.nanmedian(r_jt), "n_prompts": g.item_id.nunique()})
    s = pd.DataFrame(out)
    s.to_csv(OUT / "medxpertqa_attention_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("across traces at the key rows (within-prompt Spearman medians):")
    print(s.round(3).to_string(index=False))
    if "top1_place" in d:
        t = d.dropna(subset=["top1_place"])
        print("\nwhere the most attended token sits (layer- and head-mean distribution), share of traces by row:")
        print(pd.crosstab(t.row, t.top1_place, normalize="index").round(2).to_string())
        print("the most attended CONTENT token (sink, template, instruction, <think> and </think> tokens excluded), share of traces by row:")
        print(pd.crosstab(t.row, t.top1_content_place, normalize="index").round(2).to_string())
        print("mass of the top-30 tokens by place, medians by row:")
        print(t.groupby("row")[[c for c in t.columns if c.startswith("top30_mass")]].median().round(3).to_string())


def within_summary(w: pd.DataFrame) -> None:
    show = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cols = ["ent_norm_mean", "ent_units_norm", "mass_sink", "mass_evidence", "mass_thinking", "mass_last1", "mass_last3", "mass_older", "mass_first_sentence",
            "mass_sent_restate", "mass_sent_inference", "mass_sent_reconsider", "mass_sent_doubt"]
    med = w[w.t.isin(show)].groupby("t")[cols].median().T
    med.to_csv(OUT / "medxpertqa_attention_within_curves.csv")
    print("within trace, medians by proportion kept:")
    print(med.round(3).to_string())


def panels(w: pd.DataFrame, ycol: str, ylabel: str, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from length_accuracy import BLUE, PAIRS, RED

    fig, axes = plt.subplots(4, 6, figsize=(22, 13), sharex=True, sharey=True)
    axs = axes.flatten()
    i = 0
    for pair in PAIRS:
        for order in ("ab", "ba"):
            ax = axs[i]; i += 1
            g = w[(w.pair_id == pair) & (w.order == order)]
            for corr, color in ((True, BLUE), (False, RED)):
                h = g[g.correct == corr]
                if h.sample_id.nunique() < 5:
                    continue
                q = h.groupby("t")[ycol].quantile([0.25, 0.5, 0.75]).unstack()
                ax.plot(q.index, q[0.5], color=color, lw=1.4, label=f"{'right' if corr else 'wrong'} (n {h.sample_id.nunique()})")
                ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.15, lw=0)
            ax.set_title(f"{pair} {order}  acc {g.drop_duplicates('sample_id').correct.mean():.2f}", fontsize=8.5)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=6.5, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("proportion of the thinking kept before the cut")
    fig.supylabel(ylabel)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0.01, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure -> {path}")


def fig_entropy(w: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    panels(w, "ent_norm_mean", "attention entropy over the keys / log(keys), mean over 36 layers x 32 heads",
           "Spread of the attention of the last token of the thinking kept, against the proportion kept (median and quartiles per prompt; blue right, red wrong)",
           FIG / "fig55_attention_entropy_vs_proportion.png")


def fig_mass(w: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from length_accuracy import PAIRS

    parts = [("mass_sink", "sink (first token)", "0.8"), ("mass_template", "template, instruction, <think>", "0.6"), ("mass_evidence", "stem + options", "#2ca02c"),
             ("mass_older", "own sentences, older than the last 3", "#1f77b4"), ("mass_last3", "the last 3 completed sentences", "#ff7f0e")]
    g0 = w.copy()
    g0["mass_template"] = g0.mass_template + g0.mass_instruction + g0.mass_think_open
    fig, axes = plt.subplots(4, 6, figsize=(22, 13), sharex=True, sharey=True)
    axs = axes.flatten()
    i = 0
    for pair in PAIRS:
        for order in ("ab", "ba"):
            ax = axs[i]; i += 1
            g = g0[(g0.pair_id == pair) & (g0.order == order)]
            med = g.groupby("t")[[c for c, _, _ in parts]].median()
            base = np.zeros(len(med))
            for c, lab, color in parts:
                ax.fill_between(med.index, base, base + med[c].values, color=color, alpha=0.85, label=lab, lw=0)
                base = base + med[c].values
            ax.set_title(f"{pair} {order}  acc {g.drop_duplicates('sample_id').correct.mean():.2f}", fontsize=8.5)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
    axs[0].legend(fontsize=6.5, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("proportion of the thinking kept before the cut")
    fig.supylabel("attention mass by part of the context (mean over layers and heads; medians over traces, stacked)")
    fig.suptitle("Where the attention of the last token of the thinking kept goes, against the proportion kept", fontsize=10)
    fig.tight_layout(rect=(0.01, 0, 1, 0.96))
    path = FIG / "fig56_attention_mass_vs_proportion.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure -> {path}")


def show(args: argparse.Namespace) -> None:
    """The most attended tokens of one trace at the key rows, printed as words with their sentence and segment."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    sid = args.sample_id
    order = sid.rsplit("#", 1)[0].rsplit(":", 1)[1]
    run = args.run or ROOT / f"runs/medxpertqa/main-qwen3-8b-{order}"
    found = None
    for meta_path in sorted((run / "readouts").glob("attention-*.jsonl")):
        for line in meta_path.open(encoding="utf-8"):
            m = json.loads(line)
            if m["sample_id"] == sid:
                found = (meta_path, m)
                break
        if found:
            break
    if not found:
        raise SystemExit(f"{sid}: no attention shard in {run / 'readouts'}")
    meta_path, m = found
    npz = np.load(meta_path.with_suffix(".npz"))
    i = m["i"]
    tok = AutoTokenizer.from_pretrained(snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True))
    items = {it["item_id"]: it for it in load_tasks(args.tasks or __import__("generate").TASKS)}
    p_ids, p_seg = prompt_segments(tok, items[m["item_id"]])
    trace = next(r for r in iter_traces(run, ("sample_id", "token_ids")) if r["sample_id"] == sid)
    ids = p_ids + trace["token_ids"][: m["answer_pos"]]
    n_prompt = m["n_prompt"]
    starts = m["sentence_starts"]

    def where(pos: int) -> str:
        if pos < n_prompt:
            return SEGMENTS[int(p_seg[pos])]
        g = pos - n_prompt
        if g < m["think_start"]:
            return "think_open"
        if g >= m["think_end"]:
            return "close"
        k = int(np.searchsorted(np.array(starts), g, side="right"))   # sentence number, 1-based
        return f"sentence {k}" + (" (conclusion)" if g >= starts[m["onset_row"]] else "")

    kinds = m["kinds"]
    key_idx = {k: kinds.index(k) for k in KEY_KINDS}
    for name in args.rows:
        r = key_idx[name] if name in key_idx else m["onset_row"]
        j = KEY_KINDS.index(name) if name in KEY_KINDS else None
        pos = m["rows"][r]
        print(f"\n== {sid}  row {name!r}: query = token {pos} {tok.decode([ids[pos]])!r} ({where(pos)}), {pos + 1} keys")
        if j is None:
            print("   (the onset row is an 's' row: only the per-sentence and segment masses are stored for it; top tokens exist for 0 / e / a)")
            continue
        print("   top tokens of the distribution averaged over all layers and heads:")
        for ix, v in zip(npz[f"{i}/topall_idx"][j], npz[f"{i}/topall_val"][j]):
            print(f"     {float(v):6.3f}  token {int(ix):5d}  {tok.decode([ids[int(ix)]])!r:20s}  {where(int(ix))}")
        print("   per layer, the single most attended token (head mean):")
        for L in range(0, npz[f"{i}/top_idx"].shape[1], args.every):
            ix, v = int(npz[f"{i}/top_idx"][j, L, 0]), float(npz[f"{i}/top_val"][j, L, 0])
            print(f"     layer {L:2d}: {v:6.3f} on token {ix:5d} {tok.decode([ids[ix]])!r:20s} {where(ix)}")


def fig_cross(d: pd.DataFrame) -> None:
    """fig59: at the answer row, the attention mass on the conclusion unit against the trace's relative length, one panel per prompt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from length_accuracy import BLUE, PAIRS, RED, rho

    g0 = d[d.row == "a"]
    fig, axes = plt.subplots(4, 6, figsize=(22, 13), sharex=True, sharey=True)
    axs = axes.flatten()
    i = 0
    for pair in PAIRS:
        for order in ("ab", "ba"):
            ax = axs[i]; i += 1
            g = g0[(g0.pair_id == pair) & (g0.order == order)]
            ax.scatter(g.rel, g.mass_conclusion, s=11, c=[BLUE if c else RED for c in g.correct], alpha=0.55, edgecolors="none")
            med = g.groupby("quintile").agg(x=("rel", "median"), y=("mass_conclusion", "median"))
            ax.plot(med.x, med.y, "s-", color="black", ms=4, lw=0.8)
            rj = rho(g.jump_closed, g.mass_conclusion)
            rt = rho(g.d_therefore_own, g.mass_conclusion) if "d_therefore_own" in g else np.nan
            ax.set_title(f"{pair} {order}  acc {g.correct.mean():.2f}\nrho(len, mass) {rho(g.think_tokens, g.mass_conclusion):+.2f}   rho(jump, mass) {rj:+.2f} / {rt:+.2f}", fontsize=8.5)
            ax.grid(alpha=0.2)
    axs[0].set_xticks([-2, -1, 0, 1, 2]); axs[0].set_xticklabels(["1/4x", "1/2x", "1x", "2x", "4x"])
    for ax in axes[-1]:
        ax.set_xlabel("thinking length / the prompt's median length")
    fig.supylabel("attention mass on the conclusion unit at the answer row (mean over 36 layers x 32 heads)")
    fig.suptitle("How much of the answer-writing token's attention falls on the conclusion sentence, against thinking length; blue right, red wrong; "
                 "squares = median per length quintile; jump = closed change across the conclusion unit (own sentence / written 'Therefore, the answer is L.')", fontsize=10)
    fig.tight_layout(rect=(0.01, 0, 1, 0.96))
    path = FIG / "fig59_conclusion_attention_vs_length.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", required=True, type=Path)
    r.add_argument("--tasks", type=Path, default=None)
    r.add_argument("--positions", type=Path, default=POSITIONS, help="per-trace onset of the conclusion unit (last_sentence_check.py main)")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--shard", type=int, default=50)
    r.add_argument("--log-every", type=int, default=50)
    r.add_argument("--dry", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--runs", nargs="+", type=Path, default=None, help="run directories (default: the two main runs)")
    w = sub.add_parser("show")
    w.add_argument("sample_id")
    w.add_argument("--rows", nargs="+", default=["a", "e", "onset"], choices=["0", "e", "a", "onset"])
    w.add_argument("--run", type=Path, default=None, help="run directory (default: the main run of the sample's order)")
    w.add_argument("--tasks", type=Path, default=None)
    w.add_argument("--every", type=int, default=4, help="print every Nth layer")
    r4 = sub.add_parser("run104")
    r4.add_argument("--run", required=True, type=Path, help="a thinking or no-thinking run of MODEL_NAME (its manifest says which)")
    r4.add_argument("--questions", type=Path, default=STUDY104)
    r4.add_argument("--limit", type=int, default=0)
    r4.add_argument("--shard", type=int, default=200)
    r4.add_argument("--log-every", type=int, default=200)
    r4.add_argument("--dry", action="store_true")
    r4.add_argument("--show", type=int, default=2, help="dry: print the segments of the first N traces as text")
    r4.add_argument("--write-dry", action="store_true", help="dry: write the random shards too (only into a copy of a run outside runs/)")
    a4 = sub.add_parser("analyze104")
    a4.add_argument("--models", nargs="+", default=["qwen3-8b", "qwen3-32b"], choices=sorted(NOTHINK_RUN))
    a4.add_argument("--runs-root", type=Path, default=RUNS)
    a4.add_argument("--out", type=Path, default=OUT, help="where the tables go")
    args = ap.parse_args()
    if args.cmd == "run104":
        run104(args)
    elif args.cmd == "analyze104":
        analyze104(args)
    elif args.cmd == "run":
        if args.tasks is None:
            from generate import TASKS
            args.tasks = TASKS
        run(args)
    elif args.cmd == "show":
        show(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
