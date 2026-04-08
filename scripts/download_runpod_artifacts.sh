#!/usr/bin/env bash
set -euo pipefail

# Run this on your local machine to download artifacts from RunPod.
# Usage:
#   bash scripts/download_runpod_artifacts.sh <runpod_ssh_host> <runpod_repo_path> <local_target_dir>
#
# Example:
#   bash scripts/download_runpod_artifacts.sh root@1.2.3.4 /workspace/exp-setup ./runpod-artifacts

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <runpod_ssh_host> <runpod_repo_path> <local_target_dir>"
  exit 1
fi

RUNPOD_HOST="$1"
REMOTE_REPO_PATH="$2"
LOCAL_TARGET="$3"

mkdir -p "${LOCAL_TARGET}"

scp "${RUNPOD_HOST}:${REMOTE_REPO_PATH}/output/probes.pt" "${LOCAL_TARGET}/" || true
scp "${RUNPOD_HOST}:${REMOTE_REPO_PATH}/output/results_"*.json "${LOCAL_TARGET}/" || true
scp "${RUNPOD_HOST}:${REMOTE_REPO_PATH}/runpod_llama70b_50_"*.log "${LOCAL_TARGET}/" || true

echo "Downloaded available artifacts into: ${LOCAL_TARGET}"
echo "After verifying downloads, stop/delete the RunPod instance."
