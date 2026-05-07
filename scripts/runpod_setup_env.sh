#!/usr/bin/env bash
set -euo pipefail

# Run this on the RunPod instance after cloning the repo.
# Assumes: 2x A100 80GB pod, PyTorch+CUDA template, volume at /workspace.
#
# Usage:
#   cd /workspace
#   git clone <repo_url> exp-setup
#   bash exp-setup/scripts/runpod_setup_env.sh

REPO_DIR="/workspace/exp-setup"

if [[ ! -f "${REPO_DIR}/run.py" ]]; then
  echo "Error: ${REPO_DIR}/run.py not found."
  echo "Clone the repo first: git clone <repo_url> ${REPO_DIR}"
  exit 1
fi

# --- System packages ---
apt-get update -qq
apt-get install -y -qq unzip git-lfs
git lfs install

# --- Python venv on container disk (more stable for pip writes than volume) ---
VENV_DIR="/root/venvs/exp-setup"
mkdir -p /root/venvs
python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"
pip install --no-cache-dir --ignore-installed -r requirements.txt

# --- HF cache on volume disk (avoids container disk OOM) ---
mkdir -p /workspace/hf-cache
export HF_HOME=/workspace/hf-cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub
unset TRANSFORMERS_CACHE
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

cat >> ~/.bashrc <<'BASHRC'
export HF_HOME=/workspace/hf-cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub
unset TRANSFORMERS_CACHE
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
BASHRC

# --- HF auth ---
echo "Log in to Hugging Face (needed for gated models like Llama 70B):"
huggingface-cli login

# --- Needham et al. dataset ---
if [[ ! -d "/workspace/eval_awareness" ]]; then
  cd /workspace
  git clone https://huggingface.co/datasets/jjpn2/eval_awareness
  cd eval_awareness
  git lfs pull
  bash scripts/decrypt.sh
  cd "${REPO_DIR}"
fi

# --- Sanity checks ---
echo
echo "=== Sanity checks ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
echo
echo "Setup complete. Next:"
echo "  cd ${REPO_DIR}"
echo
echo "  # Generic runner — pre-downloads with stall-watchdog, then invokes run.py."
echo "  # Model can be any alias from src/aliases.py:MODEL_ALIASES or a bare HF repo."
echo "  bash scripts/runpod_run.sh <model> [extra run.py args...]"
echo
echo "  # Examples:"
echo "  bash scripts/runpod_run.sh qwen-32b --n-gpus 2"
echo "  bash scripts/runpod_run.sh llama-70b --n-gpus 2 --sensitivity-limit 50"
echo "  bash scripts/runpod_run.sh gemma-27b --n-gpus 2"
echo
echo "  # Pre-download only (no run):"
echo "  bash scripts/hf_download_with_watchdog.sh <hf_repo_id>"
