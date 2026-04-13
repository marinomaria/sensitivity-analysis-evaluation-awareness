#!/usr/bin/env bash
set -euo pipefail

# Run this on your LOCAL machine to download artifacts from a RunPod pod.
#
# RunPod exposes SSH over TCP, so you need -P (port) for scp.
# Find the port and IP in the RunPod pod's "Connect" tab under "SSH over exposed TCP".
#
# Usage:
#   bash scripts/download_runpod_artifacts.sh <IP> <PORT> [local_target_dir]
#
# Example:
#   bash scripts/download_runpod_artifacts.sh 38.80.152.72 31044
#   bash scripts/download_runpod_artifacts.sh 38.80.152.72 31044 ./my-artifacts

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <IP> <PORT> [local_target_dir]"
  echo "Example: $0 38.80.152.72 31044"
  exit 1
fi

IP="$1"
PORT="$2"
LOCAL_TARGET="${3:-./runpod-artifacts}"
REMOTE_BASE="/workspace/exp-setup"
SSH_KEY="${HOME}/.ssh/id_ed25519"

mkdir -p "${LOCAL_TARGET}"

echo "Downloading artifacts from ${IP}:${PORT} to ${LOCAL_TARGET}/ ..."

scp -P "${PORT}" -i "${SSH_KEY}" "root@${IP}:${REMOTE_BASE}/output/probes.pt" "${LOCAL_TARGET}/" 2>/dev/null || echo "  probes.pt not found (skipped)"
scp -P "${PORT}" -i "${SSH_KEY}" "root@${IP}:${REMOTE_BASE}/output/results_*.json" "${LOCAL_TARGET}/" 2>/dev/null || echo "  results_*.json not found (skipped)"
scp -P "${PORT}" -i "${SSH_KEY}" "root@${IP}:${REMOTE_BASE}/runpod_llama70b_50_*.log" "${LOCAL_TARGET}/" 2>/dev/null || echo "  log file not found (skipped)"

echo
echo "Done. Artifacts saved to: ${LOCAL_TARGET}/"
ls -lh "${LOCAL_TARGET}/"
echo
echo "After verifying, terminate the RunPod pod to stop charges."
