#!/usr/bin/env bash
set -euo pipefail

# Per-token probe heatmaps for several models in ONE pod session.
#
# The forward passes are trivial (a couple of dozen prompts); the cost is model
# loading, so everything is done back to back while the pod is up. Probes are
# trained fresh for each model (~2 min even at 70B) and the read-out layer is
# picked by Needham AUROC, exactly as run.py does it. The 4320-prompt
# sensitivity set is never touched.
#
# Usage:
#   bash scripts/runpod_token_heatmap.sh                       # default model list
#   bash scripts/runpod_token_heatmap.sh qwen-32b llama-70b    # explicit list
#
# Env overrides:
#   EXAMPLES="a.json b.json"                  prompt sets (all share one model load)
#   N_GPUS=2 / DEVICE=cuda                    hardware
#   EVAL_DATASET=<path>/dataset.json          Needham et al. set, for layer selection
#   REUSE_PROBES=1                            reuse probes_<model>.pt when one exists
#   SKIP_DOWNLOAD=1                           weights are already on disk
#   PROBE_DIRS="output runpod-artifacts"      where REUSE_PROBES looks

if [[ ! -f "run.py" ]]; then
  echo "Error: run.py not found. Run from repo root (e.g. /workspace/exp-setup)."
  exit 1
fi

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(gemma-27b qwen-32b qwq-32b llama-70b qwen-72b deepseek-r1-70b)
fi

# Both sets by default: the sensitivity prompts show where the probe fires in
# the experiment, the contrastive pairs show where it fires in the data that
# defined it. They share a model load, so the second set is nearly free.
EXAMPLES="${EXAMPLES:-datasets/heatmap_examples.json datasets/contrastive_examples.json}"
N_GPUS="${N_GPUS:-2}"
DEVICE="${DEVICE:-cuda}"
EVAL_DATASET="${EVAL_DATASET:-$(dirname "$(pwd)")/eval_awareness/dataset.json}"
REUSE_PROBES="${REUSE_PROBES:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
PROBE_DIRS="${PROBE_DIRS:-output runpod-artifacts}"

# Fail before the first model is loaded rather than after — a missing input
# discovered mid-run wastes GPU-billed minutes.
read -ra EXAMPLE_FILES <<<"${EXAMPLES}"
for f in "${EXAMPLE_FILES[@]}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Error: examples file not found: ${f}"
    echo "Build them with: python scripts/build_heatmap_examples.py"
    echo "                 python scripts/build_contrastive_examples.py"
    exit 1
  fi
done
if [[ ! -f "${EVAL_DATASET}" ]]; then
  echo "Error: Needham et al. dataset not found: ${EVAL_DATASET}"
  echo "Layer selection needs it. See scripts/runpod_setup_env.sh, or set EVAL_DATASET."
  exit 1
fi

if [[ -f "/root/venvs/exp-setup/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source /root/venvs/exp-setup/bin/activate
fi

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

find_probe() {
  local model="$1" dir
  for dir in ${PROBE_DIRS}; do
    if [[ -f "${dir}/probes_${model}.pt" ]]; then
      echo "${dir}/probes_${model}.pt"
      return 0
    fi
  done
  return 1
}

echo "=== Token heatmaps for ${#MODELS[@]} model(s): ${MODELS[*]} ==="
echo "    examples: ${EXAMPLE_FILES[*]}"
echo

FAILED=()
for MODEL in "${MODELS[@]}"; do
  echo "--- ${MODEL} ---"
  REPO_ID=$(python -m src.aliases "${MODEL}")
  if [[ -z "${REPO_ID}" ]]; then
    echo "  could not resolve '${MODEL}', skipping"
    FAILED+=("${MODEL}")
    continue
  fi

  if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
    echo "  skipping download (SKIP_DOWNLOAD=1)"
  else
    # Resumable, watchdog-guarded: a stalled download is killed and retried
    # instead of burning GPU-billed time.
    bash scripts/hf_download_with_watchdog.sh "${REPO_ID}"
  fi

  PROBE_ARGS=(--train-probe)
  if [[ "${REUSE_PROBES}" == "1" ]] && PROBE=$(find_probe "${MODEL}"); then
    echo "  reusing saved probes: ${PROBE}"
    PROBE_ARGS=(--load-probe "${PROBE}")
  else
    echo "  training probes for ${MODEL}, then selecting the layer by Needham AUROC"
  fi

  SAFE_NAME="${MODEL//\//_}"
  LOG_FILE="runpod_heatmap_${SAFE_NAME}_$(date +%Y%m%d_%H%M%S).log"

  if python scripts/token_heatmap.py \
      --model "${MODEL}" --device "${DEVICE}" --n-gpus "${N_GPUS}" \
      --examples "${EXAMPLE_FILES[@]}" --eval-dataset "${EVAL_DATASET}" --select-layer \
      "${PROBE_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"; then
    echo "  done → output/heatmap_${SAFE_NAME}_*.npz  (log: ${LOG_FILE})"
  else
    echo "  FAILED (see ${LOG_FILE})"
    FAILED+=("${MODEL}")
  fi
  echo
done

echo "=== Finished. Artifacts: output/heatmap_*.npz, output/probes_*_heatmap.pt ==="
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed models: ${FAILED[*]}"
  exit 1
fi
