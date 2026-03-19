# Evaluation Awareness Sensitivity Analysis

Experiment setup for the thesis *"Sensitivity Analysis of Evaluation Awareness in Large Language Models"*.

## What it does

1. Trains one linear probe for each layer of the model on a contrastive dataset of evaluation-aware vs. non-aware model responses.
2. Analyzes which probe is best based on how well it distinguishes between WMDP questions vs. WMDP questions + a deploy prompt.
3. Runs a sensitivity dataset through the model and scores each prompt using the winning probe.
4. Saves augmented results to `output/` — including generated text, probe projection score, and binary verdict.

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
# All layers (default) — trains probes on all layers, evaluates on WMDP to find best
python run.py --test-mode

# Specific layers only — evaluates on WMDP to pick best among specified layers
python run.py --test-mode --layers 4,8,12

# Single layer (skips WMDP automatically)
python run.py --test-mode --layers 8

# Full run with Llama 70B
python run.py --model meta-llama/Llama-3.3-70B-Instruct --device cuda

# Skip probe retraining (auto-detects single vs multi-layer .pt format)
python run.py --test-mode --load-probe output/probes.pt
```

## File Layout

```
run.py                          # Main orchestration script
src/
  model.py                      # Model loading + generation (HookedTransformer)
  probe.py                      # Probe training and inference
  wmdp.py                       # Load WMDP dataset for probe analysis
datasets/
  contrastive_dataset.json      # Training data (from evaluation-awareness-probing)
  sensitivity_dataset.json      # Evaluation prompts
output/
  probes.pt                     # Saved probe weights (one per layer)
```
