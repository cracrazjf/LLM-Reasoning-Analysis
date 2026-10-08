#!/usr/bin/env bash
# Copy code and data to a pod, or pull its run outputs back (run locally).
#   scripts/sync.sh push HOST PORT [KEY]   # src/, scripts/, data/tasks, data/raw -> /root/LLM-Reasoning-Analysis
#   scripts/sync.sh pull HOST PORT [KEY]   # runs/medxpertqa on the pod -> local runs/medxpertqa (runs/ is gitignored)
# KEY defaults to ~/.ssh/id_ed25519.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:?push|pull}"; HOST="${2:?host}"; PORT="${3:?port}"; KEY="${4:-$HOME/.ssh/id_ed25519}"
SSH="ssh -p ${PORT} -i ${KEY} -o StrictHostKeyChecking=accept-new"
REMOTE="root@${HOST}:/root/LLM-Reasoning-Analysis"
case "${MODE}" in
  push)
    ${SSH} "root@${HOST}" "mkdir -p /root/LLM-Reasoning-Analysis/data /root/LLM-Reasoning-Analysis/runs"
    rsync -az --no-o --no-g -e "${SSH}" --exclude __pycache__ src scripts "${REMOTE}/"
    rsync -az --no-o --no-g -e "${SSH}" data/tasks data/raw "${REMOTE}/data/"
    ;;
  pull)
    mkdir -p runs/medxpertqa
    rsync -az -e "${SSH}" "${REMOTE}/runs/medxpertqa/" runs/medxpertqa/
    ;;
  *) echo "usage: $0 push|pull HOST PORT [KEY]"; exit 1 ;;
esac
