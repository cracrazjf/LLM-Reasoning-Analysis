# Runbook

Two question sets, each a two-option version of a MedXpertQA diagnosis question (correct option vs
one distractor, shown in both orders): 70 pairs where reasoning helps (`data/pairs/medxpertqa_helps70.csv`)
and 40 where it hurts (`medxpertqa_hurts40.csv`). Model: Qwen3-8B in thinking mode, revision pinned
in `src/generate.py`.

## Pipeline

| Step | Where | Command | Output |
|---|---|---|---|
| Questions and prompts | local | `python src/medxpertqa_dataset.py build` | `data/tasks/`, `data/prompts/medxpertqa.jsonl` |
| Selection files and pair lists | local | `python src/selection.py helps` / `hurts --conditions C0_baseline` / `export` | `data/selections/`, `data/pairs/`, `data/prompts/conditions.json`, `medxpertqa_main.jsonl` |
| Generation | pod | `python src/generate.py --selection SEL --out RUN --samples 100` | `RUN/traces-NN.jsonl`, `manifest.json` |
| Sentence readouts | pod | `python src/readout.py --run RUN --selection SEL` | `RUN/readouts/` |
| Per-sentence table | local | `python src/paths.py --run RUN` | `RUN/analysis/paths.npz`, `paths_traces.parquet` |
| Analyses | local | `python src/analysis/stop_line.py --run RUN`, `python src/analysis/plot_logp_paths.py --run RUN --value logit_vs_other` | `RUN/analysis/` |

`scripts/pod_run.sh SET` runs generation and readouts for one set on the pod. Both steps resume
from what is already written.

## Pod

1x A100 80 GB, container disk >= 60 GB, no network volume. SSH key `~/.ssh/id_ed25519`.

1. Local: `scripts/sync.sh push HOST PORT` (code and data), then upload the traces if the readouts
   are for an existing run: `rsync -az -e "ssh -p PORT" runs/medxpertqa/RUN root@HOST:/root/LLM-Reasoning-Analysis/runs/medxpertqa/`.
2. Pod, once: `bash scripts/pod_setup.sh` (vLLM 0.30.0, cu130 torch, Qwen3-8B into /root/hf_cache; writes /root/env.sh).
3. Pod, inside tmux: `source /root/env.sh && cd /root/LLM-Reasoning-Analysis && bash scripts/pod_run.sh hurts`.
4. Local: `scripts/sync.sh pull HOST PORT`, then check the sha256 of every pulled file against the pod before closing it.

Measured on an A100: generation about 2,300 tokens/s (76 traces/min at 1,500 thinking tokens);
readouts about 435 per second with `--max-traces 128` (the KV cache holds 380k tokens). The
readout's full-vocabulary values need the engine in-process (set by the script); if the logits
processor cannot be loaded, run with `--no-vocab-stats`.

## What a run records

Per trace (`generate.py`): the selection fields, token ids and text, the log-probability of every
sampled token, thinking length, finish reason, the parsed answer and its correctness, and at the
answer position the raw log-probabilities of A and B (X = their difference).

Per sentence of the thinking and at the stop (`readout.py`), in two contexts: closed (thinking
cut and closed with `\n</think>\n\n`: the forced answer) and open (nothing appended: the model's own
next token). Each gives raw logits and log-probabilities of A, B and `</think>`, the top 20
candidates, and the mean, SD and log-sum-exp of the logits over the whole vocabulary. Values are
in steps of about 0.25 (half precision). Once per prompt: the closed context on the empty thinking
block.

## Runs

`runs/` is not in git. `runs/medxpertqa/main-qwen3-8b`: the 70 helps pairs x 2 orders x 3
conditions x 100 traces, with sentence readouts in the 2026-10 format (`readouts/` with X and
log p, `readouts_logits/` with raw logits for the baseline traces); `src/paths.py` reads both
formats. Keep a copy of this directory outside the machine.
