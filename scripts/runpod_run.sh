#!/usr/bin/env bash
set -euo pipefail

# Run the experiment for any model in the registry.
# Pre-downloads the model with the stall-watchdog, then invokes run.py.
# Always passes --n-gpus 2 and --skip-generation to run.py (after any extra
# args, so they cannot be overridden). Additional arguments after the model
# are forwarded to run.py before those flags.
#
# Usage:
#   bash scripts/runpod_run.sh <model> [extra run.py args...]
#
# Examples:
#   bash scripts/runpod_run.sh qwen-32b
#   bash scripts/runpod_run.sh llama-70b --sensitivity-limit 50
#   bash scripts/runpod_run.sh meta-llama/Llama-3.3-70B-Instruct
#   bash scripts/runpod_run.sh gemma-27b --sensitivity-limit 100

if [[ ! -f "run.py" ]]; then
  echo "Error: run.py not found. Run from repo root (e.g. /workspace/exp-setup)."
  exit 1
fi

MODEL="${1:?Usage: $0 <model_alias_or_hf_repo> [extra run.py args...]}"
shift

# Activate venv if present (no-op outside RunPod)
if [[ -f "/root/venvs/exp-setup/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source /root/venvs/exp-setup/bin/activate
fi

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# Resolve alias → HF repo ID. src/aliases.py is torch-free, so this stays
# fast (~200ms) and works even if the ML stack isn't loadable yet.
REPO_ID=$(python -m src.aliases "${MODEL}")

if [[ -z "${REPO_ID}" ]]; then
  echo "Error: could not resolve model '${MODEL}'."
  exit 1
fi

echo "=== Model: ${MODEL}  (HF repo: ${REPO_ID}) ==="

# Pre-download with stall-watchdog. Resumable, so any I/O hang on /workspace
# is killed and retried instead of wasting GPU-billed time.
bash scripts/hf_download_with_watchdog.sh "${REPO_ID}"

SAFE_NAME="${MODEL//\//_}"
LOG_FILE="runpod_${SAFE_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo
echo "=== Running pipeline for ${MODEL}. Log: ${LOG_FILE} ==="
echo "    extra args: $*"

# Pass the alias (not the resolved repo ID) — run.py / src.model.load_model
# handle the alias mapping, including TransformerLens architecture overrides.
python run.py --model "${MODEL}" --device cuda --n-gpus 2 "$@" --skip-generation 2>&1 | tee "${LOG_FILE}"

echo
echo "Run complete."
echo "Log:       ${LOG_FILE}"
echo "Artifacts: output/probes*.pt, output/results_*.json"
