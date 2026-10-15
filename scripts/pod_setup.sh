#!/usr/bin/env bash
# One-time environment setup on a RunPod A100 pod (run as root on the pod).
#   bash scripts/pod_setup.sh
# Installs vLLM 0.30.0 with a CUDA-13 torch into /root/venv, downloads Qwen3-8B at the pinned
# revision into /root/hf_cache (pod-local disk: network volumes turn HF symlinks into 0-byte
# files), and writes /root/env.sh to source before every run.
set -euo pipefail

VLLM_VERSION="0.30.0"
MODEL="Qwen/Qwen3-8B"
MODEL_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
export HF_HOME=/root/hf_cache
export DEBIAN_FRONTEND=noninteractive

echo "== GPU =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== venv =="
python3 -m venv /root/venv || { apt-get update -qq && apt-get install -y -qq python3-venv && python3 -m venv /root/venv; }
source /root/venv/bin/activate
pip install -q -U pip
pip install -q "vllm==${VLLM_VERSION}" pandas
CUDA_BUILD=$(python -c "import torch; print(torch.version.cuda or '')")
echo "torch $(python -c 'import torch; print(torch.__version__)') cuda build ${CUDA_BUILD}"
if [[ "${CUDA_BUILD}" != 13* ]]; then
  echo "vLLM ${VLLM_VERSION} wheels need the cu130 torch; reinstalling torch from the cu130 index"
  TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  pip install -q --force-reinstall --no-deps "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu130
fi

echo "== CUDA compat (drivers older than CUDA 13 need the compat package) =="
DRIVER_CUDA=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | awk '{print $3}')
LD_LINE=""
if [[ "${DRIVER_CUDA%%.*}" -lt 13 ]]; then
  apt-get update -qq && apt-get install -y -qq cuda-compat-13-0
  LD_LINE='export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}'
  eval "${LD_LINE}"
fi

cat > /root/env.sh <<ENV
source /root/venv/bin/activate
export HF_HOME=${HF_HOME}
export VLLM_LOGGING_LEVEL=WARNING
export PYTHONUNBUFFERED=1
${LD_LINE}
ENV
echo "wrote /root/env.sh"

echo "== torch sees the GPU? =="
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0), 'cuda build', torch.version.cuda)"

echo "== model =="
if command -v hf >/dev/null; then hf download "${MODEL}" --revision "${MODEL_REVISION}" >/dev/null; else huggingface-cli download "${MODEL}" --revision "${MODEL_REVISION}" >/dev/null; fi
python -c "from huggingface_hub import snapshot_download; p = snapshot_download('${MODEL}', revision='${MODEL_REVISION}', local_files_only=True); print('model at', p)"
echo "setup done"
