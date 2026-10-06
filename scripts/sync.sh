#!/usr/bin/env bash
# Copy code and data to the pod, or pull run outputs back (run locally).
#   scripts/sync.sh push HOST PORT      # src/, scripts/, data/selections, data/tasks -> /root/LLM-Reasoning-Analysis
#   scripts/sync.sh pull HOST PORT      # runs/medxpertqa on the pod -> local runs/medxpertqa (runs/ is gitignored)
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:?push|pull}"; HOST="${2:?host}"; PORT="${3:?port}"
SSH="ssh -p ${PORT} -o StrictHostKeyChecking=accept-new"
REMOTE="root@${HOST}:/root/LLM-Reasoning-Analysis"
case "${MODE}" in
  push)
    ${SSH} "root@${HOST}" "mkdir -p /root/LLM-Reasoning-Analysis/data /root/LLM-Reasoning-Analysis/runs"
    rsync -az --no-o --no-g -e "${SSH}" --exclude __pycache__ src scripts docs "${REMOTE}/"
    rsync -az --no-o --no-g -e "${SSH}" data/selections data/tasks data/raw "${REMOTE}/data/"
    ;;
  pull)
    mkdir -p runs/medxpertqa
    rsync -az -e "${SSH}" "${REMOTE}/runs/medxpertqa/" runs/medxpertqa/
    ;;
  *) echo "usage: $0 push|pull HOST PORT"; exit 1 ;;
esac
