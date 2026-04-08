#!/usr/bin/env bash
set -euo pipefail

# Run this on the RunPod instance after SSH login.
# Usage:
#   bash scripts/runpod_setup_env.sh <repo_url>

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <repo_url>"
  exit 1
fi

REPO_URL="$1"
REPO_DIR="exp-setup"

if [[ ! -d "${REPO_DIR}" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

if [[ ! -d "../eval_awareness" ]]; then
  git clone https://huggingface.co/datasets/jjpn2/eval_awareness ../eval_awareness
fi

pushd ../eval_awareness >/dev/null
bash scripts/decrypt.sh
popd >/dev/null

echo "Setup complete."
echo "Next: bash scripts/runpod_run_llama70b_50.sh"
