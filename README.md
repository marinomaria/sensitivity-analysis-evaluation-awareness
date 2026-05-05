# Evaluation Awareness Sensitivity Analysis

Experiment setup for the thesis *"Sensitivity Analysis of Evaluation Awareness in Large Language Models"*.

## What it does

1. Trains one linear probe for each layer of the model on a contrastive dataset, using Dolly-derived prompts by default.
2. Evaluates each probe on the Needham et al. dataset (chat-only transcripts) to find the best layer via AUROC — how well does the probe separate evaluation conversations from organic ones?
3. Runs a sensitivity dataset through the model and scores each prompt using the winning probe.
4. Saves augmented results to `output/` — including generated text, probe projection score, and binary verdict.

## Credits

**Everything related to probe training is based on and credited to [evaluation-awareness-probing](https://github.com/Jordine/evaluation-awareness-probing)**, including:

- `datasets/contrastive_dataset.json` — the original legacy contrastive prompt pairs
- The probe training methodology (normalized mean difference of activations at a chosen layer)
- The steering vector design and projection/thresholding approach

`src/probe.py` mirrors the implementation from that repository.

Probe evaluation uses the [Needham et al. dataset](https://huggingface.co/datasets/jjpn2/eval_awareness) ("Large Language Models Often Know They Are Being Evaluated", Needham et al.) — not included in this repo, see setup below.

## Local Setup

Requires Python 3.11. Create a virtual environment with your preferred tool and install dependencies:

```bash
pip install -r requirements.txt
```

### Needham et al. Dataset (required for multi-layer probe selection)

The Needham et al. dataset is not included in this repo. To obtain it:

```bash
# From the parent directory of this repo:
git clone https://huggingface.co/datasets/jjpn2/eval_awareness ../eval_awareness
cd ../eval_awareness
git lfs pull
bash scripts/decrypt.sh
cd -
```

This places `dataset.json` at `../eval_awareness/dataset.json`, which is the default path.

To use a custom path: `--eval-dataset /path/to/dataset.json`

Probe evaluation is restricted to IDs in `datasets/needham_allowlist.json`. To rebuild that allowlist from the old curated spreadsheet workflow, run:

```bash
python scripts/export_needham_allowlist_from_csv.py --csv /path/to/needham_curated.csv
```

## Usage

```bash
# All layers (default) — trains probes on all layers, evaluates on Needham et al. dataset to find best
python run.py --test-mode

# Specific layers only — evaluates on Needham et al. dataset to pick best among specified layers
python run.py --test-mode --layers 4,8,12

# Single layer (skips Needham et al. evaluation automatically)
python run.py --test-mode --layers 8

# Full run with Llama 70B (requires 2x 80GB GPUs)
python run.py --model meta-llama/Llama-3.3-70B-Instruct --device cuda --n-gpus 2

# Full run with Llama 70B using only first 50 sensitivity records
python run.py --model meta-llama/Llama-3.3-70B-Instruct --device cuda --n-gpus 2 --sensitivity-limit 50

# Skip probe retraining
python run.py --test-mode --load-probe output/probes.pt

# Use the original evaluation-awareness-probing contrastive file
python run.py --test-mode --contrastive-dataset legacy

# Custom eval dataset path
python run.py --test-mode --eval-dataset /path/to/dataset.json

# Custom Needham allowlist path
python run.py --test-mode --needham-allowlist /path/to/needham_allowlist.json
```

The 1B model (`--test-mode`) runs on CPU/MPS. 70B requires CUDA with 2x 80GB GPUs (`--n-gpus 2`).

## Dataset Schemas

The default contrastive source is `dolly`; available choices are `dolly`, `wildchat`, and `legacy`.

New contrastive dataset files use a single prompt stem field:

```json
[
  { "stem": "Best resorts in Maldives" }
]
```

`legacy` keeps supporting `datasets/contrastive_dataset.json`, whose rows contain the older `question`/`positive`/`negative` fields.

## Cloud Run (RunPod)

For running 70B models that don't fit on local hardware.

Quick version (on a 2x A100 80GB RunPod pod):

```bash
cd /workspace
git clone <your_repo_url> exp-setup
bash exp-setup/scripts/runpod_setup_env.sh       # install deps, auth, dataset
cd exp-setup
bash scripts/runpod_run_llama70b_50.sh            # run experiment
```

Then from your local machine:

```bash
bash scripts/download_runpod_artifacts.sh <IP> <PORT>
```

## File Layout

```
run.py                          # Main orchestration script
src/
  model.py                      # Model loading + generation (HookedTransformer)
  probe.py                      # Probe training and inference
  needham.py                    # Load Needham et al. dataset for probe selection
datasets/
  contrastive_dataset_dolly.json    # Default contrastive stems
  contrastive_dataset_wildchat.json # Alternative contrastive stems
  contrastive_dataset.json          # Legacy evaluation-awareness-probing data
  needham_allowlist.json            # Needham conversation IDs used for probe selection
  sensitivity_dataset.json      # Evaluation prompts
scripts/
  build_dolly_contrastive.py    # Build Dolly contrastive stems
  build_wildchat_contrastive.py # Build WildChat contrastive stems
  export_needham_allowlist_from_csv.py # One-off migration from curated CSV
  runpod_setup_env.sh           # Pod environment setup (deps, auth, dataset)
  runpod_run_llama70b_50.sh     # Run experiment on pod
  download_runpod_artifacts.sh  # Download results to local machine
docs/
  cloud-llama70b-budgeted-run.md  # Full RunPod guide with troubleshooting
output/
  probes.pt                     # Saved probe weights (one per layer)
```
