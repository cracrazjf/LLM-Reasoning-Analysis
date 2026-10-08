#!/usr/bin/env bash
# The main run on a two-GPU pod: 100 thinking samples per prompt for the study pairs, one option
# order per GPU, then the sentence readouts of both runs, again one per GPU. Every step resumes.
#   source /root/env.sh && cd /root/LLM-Reasoning-Analysis && bash scripts/pod_main.sh [samples]
# Needs data/screen/medxpertqa_study12_pairs.txt and, to continue from the screening samples,
# runs/medxpertqa/main-qwen3-8b-<order>/traces-00.jsonl copied from the think runs.
# Logs: runs/medxpertqa/main-qwen3-8b-<order>.log and -readout.log. Ends with POD_MAIN_DONE.
set -uo pipefail
cd "$(dirname "$0")/.."
SAMPLES="${1:-100}"
PAIRS=data/screen/medxpertqa_study12_pairs.txt
mkdir -p runs/medxpertqa
gpu() { [ "$1" = ab ] && echo 0 || echo 1; }

for o in ab ba; do
  RUN=runs/medxpertqa/main-qwen3-8b-$o
  CUDA_VISIBLE_DEVICES=$(gpu $o) python src/generate.py --mode think --order $o --pairs "$PAIRS" --out "$RUN" --samples "$SAMPLES" \
    >> "${RUN}.log" 2>&1 &
done
wait
for o in ab ba; do
  grep -q "^done:" runs/medxpertqa/main-qwen3-8b-$o.log || { echo "generation $o did not finish"; exit 1; }
done
echo "GENERATION_DONE"

for o in ab ba; do
  RUN=runs/medxpertqa/main-qwen3-8b-$o
  CUDA_VISIBLE_DEVICES=$(gpu $o) python src/readout.py --run "$RUN" >> "${RUN}-readout.log" 2>&1 &
done
wait
for o in ab ba; do
  grep -q "^done:" runs/medxpertqa/main-qwen3-8b-$o-readout.log || { echo "readout $o did not finish"; exit 1; }
done
echo "POD_MAIN_DONE"
