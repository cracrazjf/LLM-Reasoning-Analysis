#!/usr/bin/env bash
# The screening stage on the pod: smoke test, full screening run, 10-option prior.
#   source /root/env.sh && cd /root/LLM-Reasoning-Analysis && bash scripts/pod_run_screen.sh [smoke|screen|prior|all]
# Each step writes its log next to its output under runs/medxpertqa/. Steps are resumable:
# generate.py continues from the traces already stored.
set -euo pipefail
cd "$(dirname "$0")/.."
STEP="${1:-all}"
SEL=data/selections/medxpertqa_screen.json
RUNS=runs/medxpertqa
mkdir -p "${RUNS}"

smoke() {  # 20 prompts x 2 samples: format check (answer-first-word rate, thinking length) before the real run
  python src/generate.py --selection "${SEL}" --out "${RUNS}/smoke-qwen3-8b" --samples 2 --prompts 20 \
    2>&1 | tee "${RUNS}/smoke-qwen3-8b.log"
  python - <<'PY'
import json, glob, statistics as st
rows = [json.loads(l) for f in glob.glob("runs/medxpertqa/smoke-qwen3-8b/traces-*.jsonl") for l in open(f)]
ans = [r for r in rows if r["answer"]]
print(f"smoke: {len(rows)} traces, answered {len(ans)}, correct {sum(bool(r['correct']) for r in ans)}, "
      f"finish {dict((k, sum(r['finish_reason']==k for r in rows)) for k in set(r['finish_reason'] for r in rows))}, "
      f"think tokens median {st.median([r['think_tokens'] for r in rows if r['think_tokens']] or [0])}, "
      f"max {max([r['think_tokens'] or 0 for r in rows])}")
for r in rows[:3]:
    print("  answer_text:", repr(r["answer_text"]), "X:", r["X"])
PY
}

screen() {  # 1,962 prompts x 8 samples = 15,696 traces
  python src/generate.py --selection "${SEL}" --out "${RUNS}/screen-qwen3-8b" --samples 8 \
    2>&1 | tee -a "${RUNS}/screen-qwen3-8b.log"
}

prior() {  # 109 ten-option prompts, one forward pass each
  python src/pair_distance.py prior --out "${RUNS}/prior-qwen3-8b" 2>&1 | tee "${RUNS}/prior-qwen3-8b.log"
}

case "${STEP}" in
  smoke) smoke ;;
  screen) screen ;;
  prior) prior ;;
  all) smoke; screen; prior ;;
  *) echo "usage: $0 [smoke|screen|prior|all]"; exit 1 ;;
esac
