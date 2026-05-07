#!/usr/bin/env bash
set -euo pipefail

# Download a HuggingFace repo with a stall-watchdog.
#
# RunPod's /workspace volume occasionally hangs on writes — the download
# process freezes instead of crashing, wasting GPU-billed time. This script
# polls the cache size and, if it stops growing for STALL_THRESHOLD seconds,
# kills the download and resumes. huggingface-cli download is fully
# resumable, so the kill costs nothing.
#
# Usage:
#   bash scripts/hf_download_with_watchdog.sh <repo_id> [extra hf-cli args...]
#
# Examples:
#   bash scripts/hf_download_with_watchdog.sh meta-llama/Llama-3.3-70B-Instruct
#   bash scripts/hf_download_with_watchdog.sh google/gemma-2-27b-it
#
# Env (with defaults):
#   HF_HOME                = /workspace/hf-cache
#   HUGGINGFACE_HUB_CACHE  = $HF_HOME/hub
#   STALL_THRESHOLD        = 180   (seconds without disk growth before kill)
#   POLL_INTERVAL          = 30    (seconds between size polls)
#   MAX_ATTEMPTS           = 20
#   HF_DOWNLOAD_EXCLUDE    = "original/* *.gguf"   (skip redundant Meta-original
#                                                   weights and gguf quantizations)

REPO_ID="${1:?Usage: $0 <repo_id> [extra hf-cli args...]}"
shift || true

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

STALL_THRESHOLD="${STALL_THRESHOLD:-180}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
HF_DOWNLOAD_EXCLUDE="${HF_DOWNLOAD_EXCLUDE:-original/* *.gguf}"

mkdir -p "${HUGGINGFACE_HUB_CACHE}"

# Build --exclude args from space-separated patterns
EXCLUDE_ARGS=()
if [[ -n "${HF_DOWNLOAD_EXCLUDE}" ]]; then
  read -ra EXCLUDE_PATTERNS <<<"${HF_DOWNLOAD_EXCLUDE}"
  for pat in "${EXCLUDE_PATTERNS[@]}"; do
    EXCLUDE_ARGS+=(--exclude "$pat")
  done
fi

LOG_DIR="/tmp/hf_watchdog"
mkdir -p "${LOG_DIR}"
SAFE_REPO="${REPO_ID//\//_}"

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
  attempt=$((attempt + 1))
  LOG_FILE="${LOG_DIR}/${SAFE_REPO}.attempt${attempt}.log"
  echo "=== [${REPO_ID}] download attempt ${attempt}/${MAX_ATTEMPTS} ==="
  echo "    log: ${LOG_FILE}"

  huggingface-cli download "${REPO_ID}" \
    --cache-dir "${HUGGINGFACE_HUB_CACHE}" \
    "${EXCLUDE_ARGS[@]}" \
    "$@" \
    >"${LOG_FILE}" 2>&1 &
  DL_PID=$!

  last_size=$(du -sb "${HUGGINGFACE_HUB_CACHE}" 2>/dev/null | awk '{print $1}')
  last_size=${last_size:-0}
  last_change_ts=$(date +%s)
  killed=0

  while kill -0 "${DL_PID}" 2>/dev/null; do
    sleep "${POLL_INTERVAL}"
    cur_size=$(du -sb "${HUGGINGFACE_HUB_CACHE}" 2>/dev/null | awk '{print $1}')
    cur_size=${cur_size:-0}
    now=$(date +%s)

    if [[ "${cur_size}" != "${last_size}" ]]; then
      delta_mb=$(( (cur_size - last_size) / 1024 / 1024 ))
      total_gb=$(awk "BEGIN { printf \"%.1f\", ${cur_size}/1073741824 }")
      echo "    +${delta_mb} MB  (cache total: ${total_gb} GB)"
      last_size="${cur_size}"
      last_change_ts="${now}"
    else
      stalled=$(( now - last_change_ts ))
      echo "    no growth for ${stalled}s (kill at ${STALL_THRESHOLD}s)"
      if (( stalled >= STALL_THRESHOLD )); then
        echo "    STALL DETECTED — killing PID ${DL_PID}"
        kill -9 "${DL_PID}" 2>/dev/null || true
        killed=1
        break
      fi
    fi
  done

  set +e
  wait "${DL_PID}"
  exit_code=$?
  set -e

  if (( killed == 0 )) && (( exit_code == 0 )); then
    echo "=== [${REPO_ID}] download succeeded on attempt ${attempt} ==="
    exit 0
  fi

  if (( killed == 1 )); then
    echo "    killed by watchdog; resuming in 5s..."
  else
    echo "    download exited with code ${exit_code}; retrying in 5s..."
    echo "    last log lines:"
    tail -n 5 "${LOG_FILE}" | sed 's/^/      /'
  fi
  sleep 5
done

echo "FAILED to download ${REPO_ID} after ${MAX_ATTEMPTS} attempts" >&2
exit 1
