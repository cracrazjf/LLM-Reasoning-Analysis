# Runbook: screening stage on a RunPod A100

Pod: 1x A100 80 GB, container disk >= 60 GB (model 16 GB plus traces), no network volume needed.
SSH key: the local `~/.ssh/id_ed25519` (public key registered in RunPod).

1. Local, push code and data: `scripts/sync.sh push HOST PORT`
2. Pod, once: `bash /root/LLM-Reasoning-Analysis/scripts/pod_setup.sh` (vLLM 0.30.0, cu130 torch, cuda-compat if the driver is older than CUDA 13, Qwen3-8B at the pinned revision into /root/hf_cache; writes /root/env.sh)
3. Pod, inside tmux so an SSH drop does not kill it:
   `source /root/env.sh && cd /root/LLM-Reasoning-Analysis && bash scripts/pod_run_screen.sh smoke`
   then `... screen` (15,696 traces; resumable) and `... prior`
4. Local, pull: `scripts/sync.sh pull HOST PORT`, then
   `python src/pair_distance.py summarise --screen runs/medxpertqa/screen-qwen3-8b --prior runs/medxpertqa/prior-qwen3-8b`

Outputs live under runs/medxpertqa/ (gitignored): smoke-qwen3-8b, screen-qwen3-8b (traces-NN.jsonl + manifest.json), prior-qwen3-8b (prior.jsonl).
