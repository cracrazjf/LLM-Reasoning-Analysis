#!/usr/bin/env bash
# Masking at the answer for the 104-question study (src/ablate_conclusion.py run104, HF transformers): the model's no-thinking answers
# and its 60 thinking traces per question, both option orders: one order per GPU when two GPUs are given, both in turn on one (GPUS,
# default "0,1" on a pod with two or more GPUs, else "0").
#   local:  python src/ablate_conclusion.py prepare104                 (writes mask104_job.jsonl into the thinking runs)
#           bash scripts/sync.sh push HOST PORT                         (pushes the job files with the code)
#           bash scripts/sync.sh push-runs HOST PORT runs/medxpertqa/{nothink-qwen3-8b,study104-qwen3-8b,study104-nothink-qwen3-32b,study104-qwen3-32b}-{ab,ba}
#   pod:    source /root/env.sh && cd /root/LLM-Reasoning-Analysis && bash scripts/pod_mask104.sh qwen3-8b [GPUS]
#           bash scripts/pod_mask104.sh qwen3-32b [GPUS]      (three GPUs: qwen3-8b 0 and qwen3-32b 1,2 side by side)
#   local:  bash scripts/sync.sh pull HOST PORT; python src/ablate_conclusion.py analyze104
# The weights go to /dev/shm/hf_cache. Writes <run>/readouts/mask104-01.jsonl; resumes by sample_id. Logs: <run>-mask104.log.
# Ends with POD_MASK104_DONE.
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
    python src/ablate_conclusion.py run104 --run "$RUN" >> "$RUN-mask104.log" 2>&1 || return 1
  done
}
if [[ "$GPUS" == *,* ]]; then
  job ab "${GPUS%%,*}" & job ba "${GPUS#*,}" & wait
else
  job ab "$GPUS"; job ba "$GPUS"
fi
for o in ab ba; do
  for RUN in runs/medxpertqa/$NOTHINK-$o runs/medxpertqa/study104-$MODEL_NAME-$o; do
    grep -q "^done:" "$RUN-mask104.log" || { echo "$RUN did not finish (see $RUN-mask104.log)"; exit 1; }
  done
done
echo "POD_MASK104_DONE"
