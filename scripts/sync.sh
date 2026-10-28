#!/usr/bin/env bash
# Copy code and data to a pod, or pull its run outputs back (run locally).
#   scripts/sync.sh push HOST PORT [KEY]   # src/, scripts/, data/tasks, data/raw, the positions CSV, job files under runs/ -> /root/LLM-Reasoning-Analysis
#   scripts/sync.sh pull HOST PORT [KEY]   # runs/medxpertqa on the pod -> local runs/medxpertqa (runs/ is gitignored)
#   scripts/sync.sh push-runs HOST PORT RUN...   # finished runs a pod script reads again: traces, manifest, readouts/conclusion-*
# KEY defaults to ~/.ssh/id_ed25519 (push-runs always uses it).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:?push|pull|push-runs}"; HOST="${2:?host}"; PORT="${3:?port}"
if [ "${MODE}" = push-runs ]; then KEY="$HOME/.ssh/id_ed25519"; shift 3; else KEY="${4:-$HOME/.ssh/id_ed25519}"; fi
SSH="ssh -p ${PORT} -i ${KEY} -o StrictHostKeyChecking=accept-new"
REMOTE="root@${HOST}:/root/LLM-Reasoning-Analysis"
case "${MODE}" in
  push)
    ${SSH} "root@${HOST}" "mkdir -p /root/LLM-Reasoning-Analysis/data /root/LLM-Reasoning-Analysis/runs"
    rsync -az --no-o --no-g -e "${SSH}" --exclude __pycache__ src scripts "${REMOTE}/"
    rsync -az --no-o --no-g -e "${SSH}" data/tasks data/raw "${REMOTE}/data/"
    # per-trace positions the attention readout needs (src/attention_at_answer.py); absent until last_sentence_check.py main has run
    [ -f data/screen/medxpertqa_last_sentence_main.csv ] && ${SSH} "root@${HOST}" "mkdir -p /root/LLM-Reasoning-Analysis/data/screen" \
      && rsync -az --no-o --no-g -e "${SSH}" data/screen/medxpertqa_last_sentence_main.csv "${REMOTE}/data/screen/"
    # the 12 study pairs, read when src/last_sentence_check.py is imported (src/conclusion_readout.py uses its onset rule)
    ${SSH} "root@${HOST}" "mkdir -p /root/LLM-Reasoning-Analysis/data/screen"
    rsync -az --no-o --no-g -e "${SSH}" data/screen/medxpertqa_study12_pairs.txt "${REMOTE}/data/screen/"
    # job files prepared locally for pod scripts (conditions.json, *job.jsonl); the run outputs themselves are never pushed
    rsync -az --no-o --no-g -e "${SSH}" --include='*/' --include='conditions.json' --include='*job.jsonl' --exclude='*' --prune-empty-dirs \
      runs/medxpertqa/ "${REMOTE}/runs/medxpertqa/"
    ;;
  push-runs)
    for RUN in "$@"; do
      RUN="${RUN%/}"
      [ -f "${RUN}/manifest.json" ] || { echo "${RUN}: no manifest.json"; exit 1; }
      ${SSH} "root@${HOST}" "mkdir -p /root/LLM-Reasoning-Analysis/${RUN}/readouts"
      rsync -az --no-o --no-g -e "${SSH}" --include='traces-*.jsonl' --include='manifest.json' --exclude='*' "${RUN}/" "${REMOTE}/${RUN}/"
      if [ -d "${RUN}/readouts" ]; then
        rsync -az --no-o --no-g -e "${SSH}" --include='conclusion-*.jsonl' --exclude='*' "${RUN}/readouts/" "${REMOTE}/${RUN}/readouts/"
      fi
      echo "pushed ${RUN}"
    done
    ;;
  pull)
    mkdir -p runs/medxpertqa
    rsync -az -e "${SSH}" "${REMOTE}/runs/medxpertqa/" runs/medxpertqa/
    ;;
  *) echo "usage: $0 push|pull HOST PORT [KEY] | push-runs HOST PORT RUN..."; exit 1 ;;
esac
