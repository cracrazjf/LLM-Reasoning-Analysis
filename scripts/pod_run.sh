#!/usr/bin/env bash
# One question set on the pod: generate the traces, then the sentence readouts. Both steps resume.
#   source /root/env.sh && cd /root/LLM-Reasoning-Analysis && bash scripts/pod_run.sh hurts [samples]
# Writes runs/medxpertqa/<set>-qwen3-8b/ and the logs next to it; pull with scripts/sync.sh pull HOST PORT.
set -euo pipefail
cd "$(dirname "$0")/.."
SET="${1:?helps|hurts}"; SAMPLES="${2:-100}"
SEL="data/selections/medxpertqa_${SET}.json"
RUN="runs/medxpertqa/${SET}-qwen3-8b"
mkdir -p "$(dirname "$RUN")"
python src/generate.py --selection "$SEL" --out "$RUN" --samples "$SAMPLES" 2>&1 | tee -a "${RUN}.log"
python src/readout.py --run "$RUN" --selection "$SEL" 2>&1 | tee -a "${RUN}-readout.log"
echo "POD_RUN_DONE ${SET}"
