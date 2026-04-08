# Evaluation Awareness Sensitivity Analysis

Experiment setup for the thesis *"Sensitivity Analysis of Evaluation Awareness in Large Language Models"*.

## What it does

1. Trains one linear probe for each layer of the model on a contrastive dataset of evaluation-aware vs. non-aware model responses.
2. Evaluates each probe on the Needham et al. dataset (chat-only transcripts) to find the best layer via AUROC — how well does the probe separate evaluation conversations from organic ones?
3. Runs a sensitivity dataset through the model and scores each prompt using the winning probe.
4. Saves augmented results to `output/` — including generated text, probe projection score, and binary verdict.

## Credits

**Everything related to probe training is based on and credited to [evaluation-awareness-probing](https://github.com/Jordine/evaluation-awareness-probing)**, including:

- `datasets/contrastive_dataset.json` — the contrastive prompt pairs used for probe training
- The probe training methodology (normalized mean difference of activations at a chosen layer)
- The steering vector design and projection/thresholding approach

`src/probe.py` mirrors the implementation from that repository.

Probe evaluation uses the [Needham et al. dataset](https://huggingface.co/datasets/jjpn2/eval_awareness) ("Large Language Models Often Know They Are Being Evaluated", Needham et al.) — not included in this repo, see setup below.

## Setup

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
bash scripts/decrypt.sh
cd -
```

This places `dataset.json` at `../eval_awareness/dataset.json`, which is the default path.

To use a custom path: `--eval-dataset /path/to/dataset.json`

> Requires CUDA for 70B inference. The 1B model (`--test-mode`) runs on CPU/MPS.

## Usage

```bash
# All layers (default) — trains probes on all layers, evaluates on Needham et al. dataset to find best
python run.py --test-mode

# Specific layers only — evaluates on Needham et al. dataset to pick best among specified layers
python run.py --test-mode --layers 4,8,12

# Single layer (skips Needham et al. evaluation automatically)
python run.py --test-mode --layers 8

# Full run with Llama 70B
python run.py --model meta-llama/Llama-3.3-70B-Instruct --device cuda

# Full run with Llama 70B using only first 50 sensitivity records
python run.py --model meta-llama/Llama-3.3-70B-Instruct --device cuda --sensitivity-limit 50

# Skip probe retraining
python run.py --test-mode --load-probe output/probes.pt

# Custom eval dataset path
python run.py --test-mode --eval-dataset /path/to/dataset.json
```

## File Layout

```
run.py                          # Main orchestration script
src/
  model.py                      # Model loading + generation (HookedTransformer)
  probe.py                      # Probe training and inference
  needham.py                    # Load Needham et al. dataset for probe selection
datasets/
  contrastive_dataset.json      # Training data (from evaluation-awareness-probing)
  sensitivity_dataset.json      # Evaluation prompts
output/
  probes.pt                     # Saved probe weights (one per layer)
```
