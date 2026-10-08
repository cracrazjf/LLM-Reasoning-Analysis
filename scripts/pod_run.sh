#!/usr/bin/env bash
# One option order on this pod: the no-thinking pass, then the thinking samples; both resume.
#   source /root/env.sh && cd /root/LLM-Reasoning-Analysis && bash scripts/pod_run.sh ab [samples]
# Writes runs/medxpertqa/nothink-qwen3-8b-<order> (one greedy answer per prompt, thinking off) and
# runs/medxpertqa/think-qwen3-8b-<order> (30 thinking samples per prompt by default), with a log
# next to each. Pull with scripts/sync.sh pull HOST PORT, then classify locally with src/screen.py
# over the run directories of both orders.
set -euo pipefail
cd "$(dirname "$0")/.."
ORDER="${1:?ab|ba}"; SAMPLES="${2:-30}"
NT="runs/medxpertqa/nothink-qwen3-8b-${ORDER}"
TH="runs/medxpertqa/think-qwen3-8b-${ORDER}"
mkdir -p runs/medxpertqa
python src/generate.py --mode nothink --order "${ORDER}" --out "${NT}" 2>&1 | tee -a "${NT}.log"
python src/generate.py --mode think --order "${ORDER}" --out "${TH}" --samples "${SAMPLES}" 2>&1 | tee -a "${TH}.log"
echo "POD_RUN_DONE ${ORDER}"
