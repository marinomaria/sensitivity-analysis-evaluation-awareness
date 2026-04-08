#!/usr/bin/env bash
set -euo pipefail

# Run this on the RunPod instance from the repository root.
# It executes one fixed run and writes a timestamped log file.

if [[ ! -f "run.py" ]]; then
  echo "Error: run.py not found. Run from repo root."
  exit 1
fi

source .venv/bin/activate

LOG_FILE="runpod_llama70b_50_$(date +%Y%m%d_%H%M%S).log"

python run.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --device cuda \
  --max-new-tokens 96 \
  --sensitivity-limit 50 | tee "${LOG_FILE}"

echo
echo "Run complete."
echo "Log: ${LOG_FILE}"
echo "Artifacts:"
echo "  - output/probes.pt"
echo "  - output/results_*.json"
