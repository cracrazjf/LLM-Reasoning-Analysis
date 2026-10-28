#!/usr/bin/env bash
# Attention at the answer row for the 104-question study (src/attention_at_answer.py run104, HF transformers): the model's
# no-thinking answers and its 60 thinking traces per question, both option orders: one order per GPU when two GPUs are given,
# both in turn on one (GPUS, default "0,1" on a pod with two or more GPUs, else "0").
#   local:  bash scripts/sync.sh push HOST PORT
#           bash scripts/sync.sh push-runs HOST PORT runs/medxpertqa/{nothink-qwen3-8b,study104-qwen3-8b,study104-nothink-qwen3-32b,study104-qwen3-32b}-{ab,ba}
#   pod:    source /root/env.sh && cd /root/LLM-Reasoning-Analysis && bash scripts/pod_attention104.sh qwen3-8b [GPUS]
#           bash scripts/pod_attention104.sh qwen3-32b [GPUS]      (three GPUs: qwen3-8b 0 and qwen3-32b 1,2 side by side)
#   local:  bash scripts/sync.sh pull HOST PORT; python src/attention_at_answer.py analyze104
# The weights go to /dev/shm/hf_cache (the 32B's 61 GiB do not fit the pod disk). Writes <run>/readouts/answer-attention-SS.{npz,jsonl};
# resumes shard by shard. Logs: <run>-attention104.log. Ends with POD_ATTENTION104_DONE.
set -uo pipefail
cd "$(dirname "$0")/.."
export MODEL_NAME="${1:?model name: qwen3-8b or qwen3-32b}"
export HF_HOME=/dev/shm/hf_cache
read -r REPO REV < <(python -c "import sys; sys.path.insert(0, 'src'); from generate import MODEL, MODEL_REVISION; print(MODEL, MODEL_REVISION)")
echo "model $REPO @ $REV -> $HF_HOME"
hf download "$REPO" --revision "$REV" > /dev/null || { echo "download failed"; exit 1; }
# the 8B's first reactions are its screening run's greedy answers (all 981 pairs; run104 keeps the 104 questions)
NOTHINK=$([ "$MODEL_NAME" = qwen3-8b ] && echo nothink-qwen3-8b || echo "study104-nothink-$MODEL_NAME")
GPUS="${2:-$([ "$(nvidia-smi -L | wc -l)" -ge 2 ] && echo 0,1 || echo 0)}"

job() {   # order, GPU
  export CUDA_VISIBLE_DEVICES=$2
  for RUN in runs/medxpertqa/$NOTHINK-$1 runs/medxpertqa/study104-$MODEL_NAME-$1; do
    python src/attention_at_answer.py run104 --run "$RUN" >> "$RUN-attention104.log" 2>&1 || return 1
  done
}
if [[ "$GPUS" == *,* ]]; then
  job ab "${GPUS%%,*}" & job ba "${GPUS#*,}" & wait
else
  job ab "$GPUS"; job ba "$GPUS"
fi
for o in ab ba; do
  for RUN in runs/medxpertqa/$NOTHINK-$o runs/medxpertqa/study104-$MODEL_NAME-$o; do
    grep -q "^done:" "$RUN-attention104.log" || { echo "$RUN did not finish (see $RUN-attention104.log)"; exit 1; }
  done
done
echo "POD_ATTENTION104_DONE"
