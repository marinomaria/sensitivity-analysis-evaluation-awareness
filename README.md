# Evaluation Awareness Sensitivity Analysis

Experiment setup for the thesis *"Sensitivity Analysis of Evaluation Awareness in Large Language Models"*.

## What it does

1. Trains a linear probe on a contrastive dataset of evaluation-aware vs. non-aware model responses.
2. Runs a sensitivity dataset through the model and scores each prompt using the probe.
3. Saves augmented results to `output/` — including generated text, probe projection score, and binary verdict.

## Credits

**Everything related to probe training is based on and credited to [evaluation-awareness-probing](https://github.com/Jordine/evaluation-awareness-probing)**, including:

- `datasets/contrastive_dataset.json` — the contrastive prompt pairs used for probe training
- The probe training methodology (normalized mean difference of activations at a chosen layer)
- The steering vector design and projection/thresholding approach

`src/probe.py` mirrors the implementation from that repository.

## Setup

```bash
conda create -n ea-probing python=3.11
conda activate ea-probing
pip install -r requirements.txt
```

> Requires CUDA for 70B inference. The 1B model (`--test-mode`) runs on CPU.

## Usage

```bash
# Quick test with Llama 1B
python run.py --test-mode

# Full run with Llama 70B
python run.py --model meta-llama/Llama-3.3-70B-Instruct --device cuda

# Skip probe retraining
python run.py --load-probe output/probe.pt
```

## File Layout

```
run.py                          # Main orchestration script
src/
  model.py                      # Model loading + generation (HookedTransformer)
  probe.py                      # Probe training and inference
datasets/
  contrastive_dataset.json      # Training data (from evaluation-awareness-probing)
  sensitivity_dataset.json      # Evaluation prompts
output/
  probe.pt                      # Saved probe weights
```
