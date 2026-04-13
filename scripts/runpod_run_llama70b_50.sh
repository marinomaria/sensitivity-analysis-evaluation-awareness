#!/usr/bin/env bash
set -euo pipefail

# Run this on the RunPod instance from the repository root (/workspace/exp-setup).
# Executes the full pipeline (probe train + Needham eval + sensitivity) on Llama 70B
# with sensitivity limited to 50 records. Writes a timestamped log file.
#
# Prerequisites:
#   - bash scripts/runpod_setup_env.sh <repo_url>  (already completed)
#   - HF_HOME exported to /workspace/hf-cache
#
# Usage:
#   cd /workspace/exp-setup
#   bash scripts/runpod_run_llama70b_50.sh

if [[ ! -f "run.py" ]]; then
  echo "Error: run.py not found. Run this from the repo root (/workspace/exp-setup)."
  exit 1
fi

source /root/venvs/exp-setup/bin/activate

export HF_HOME=/workspace/hf-cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub
unset TRANSFORMERS_CACHE
export HF_HUB_DISABLE_XET=1

LOG_FILE="runpod_llama70b_50_$(date +%Y%m%d_%H%M%S).log"

echo "Starting run. Log: ${LOG_FILE}"
echo "Model downloads may take 10-20 min on first run."
echo

python run.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --device cuda \
  --n-gpus 2 \
  --max-new-tokens 96 \
  --sensitivity-limit 50 2>&1 | tee "${LOG_FILE}"

echo
echo "Run complete."
echo "Log:       ${LOG_FILE}"
echo "Artifacts: output/probes.pt, output/results_*.json"
echo
echo "Next: download artifacts to local machine, then terminate the pod."
