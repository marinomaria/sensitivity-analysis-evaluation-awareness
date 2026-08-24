#!/usr/bin/env python
"""
Run a trained probe over *every* token position of a handful of example prompts
and dump the resulting layer x token matrices for plotting.

Costs one forward pass per example — the same forward the last-token pipeline
already does — so the only real expense is loading the model. The 4320-prompt
sensitivity set is never touched.

Usage:
    # Reuse the probes from the main run (cheapest: forwards only)
    python scripts/token_heatmap.py --model llama-70b --device cuda --n-gpus 2 \
        --load-probe runpod-artifacts/probes_llama-70b.pt

    # No saved probes: train them (~2 min at 70B) and pick the layer the way
    # run.py does, by Needham AUROC
    python scripts/token_heatmap.py --model llama-1b --train-probe --select-layer

    # Only a band of layers around the best one (smaller output file)
    python scripts/token_heatmap.py --model llama-70b --layers 30,35,39,45,50 ...
"""
# Imports are deliberately split around the sys.path bootstrap below.
# ruff: noqa: I001, E741
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aliases import resolve_model_name
from src.model import apply_chat_template_with_fallback, get_device, load_model
from src.probe import evaluate_probes, get_token_projections, load_probes, save_probes
from run import (  # run.py owns the contrastive-set registry and dataset defaults
    DEFAULT_CONTRASTIVE_DATASET,
    DEFAULT_EVAL_DATASET,
    _CONTRASTIVE_DATASETS,
    load_contrastive_dataset,
)

DEFAULT_EXAMPLES = "datasets/heatmap_examples.json"
OUTPUT_DIR = "output"

_METADATA_FIELDS = (
    "scenario", "entity_familiarity", "context_coherence",
    "ethical_pressure", "prompt_structure", "lang",
    # Contrastive examples carry these instead of the sensitivity factors.
    "system", "label", "pair_id",
)


# The first residual-stream positions (BOS / attention sink) carry an enormous
# norm that has nothing to do with the probe direction. Including them when
# summarising or scaling a map lets them swamp every real signal, so callers
# exclude them by default — they are still present in the data.
SKIP_FIRST_POSITIONS = 2


def normalize_by_layer(matrix, mode, skip_first=None):
    """
    Make a layer x token matrix comparable across depth.

    Residual-stream norm grows with layer index, so raw values are dominated by
    depth rather than by which token is doing the work. Centering each layer on
    its own median — and optionally scaling by its median absolute deviation —
    turns the question into "which positions stand out *within* this layer",
    which is what the map is read for.

    mode: 'robust-z' (center + MAD scale), 'center' (center only), or 'none'.
    """
    if mode == "none":
        return matrix
    skip = SKIP_FIRST_POSITIONS if skip_first is None else skip_first
    body = matrix[:, skip:] if matrix.shape[1] > skip else matrix
    centered = matrix - np.nanmedian(body, axis=1, keepdims=True)
    if mode == "center":
        return centered
    if mode == "robust-z":
        body_centered = centered[:, skip:] if centered.shape[1] > skip else centered
        mad = np.nanmedian(np.abs(body_centered), axis=1, keepdims=True)
        return centered / np.maximum(mad, 1e-6)
    raise ValueError(f"Unknown normalize mode: {mode!r}")


def examples_tag(path):
    """Short name for an examples file, used in the output filename."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = stem.removeprefix("heatmap_").removesuffix("_examples")
    return "sensitivity" if stem in ("", "examples") else stem


def _train_probes_for(model, tokenizer, args):
    """Train probes on every layer, reusing run.py's contrastive-set loading."""
    from src.probe import train_probes

    name = args.contrastive_dataset or DEFAULT_CONTRASTIVE_DATASET
    if name not in _CONTRASTIVE_DATASETS:
        raise SystemExit(f"Unknown contrastive dataset {name!r}; choices: {sorted(_CONTRASTIVE_DATASETS)}")

    contrastive = load_contrastive_dataset(_CONTRASTIVE_DATASETS[name], tokenizer)
    probes = train_probes(model, contrastive, list(range(model.cfg.n_layers)))

    # Never write over a probe file from the main pipeline: cache under a
    # heatmap-specific name, and leave an existing one alone.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache = os.path.join(OUTPUT_DIR, f"probes_{args.model.replace('/', '_')}_{name}_heatmap.pt")
    if os.path.exists(cache):
        print(f"Probe cache already exists, leaving it untouched: {cache}")
    else:
        save_probes(probes, cache)
    return probes


def _select_best_layer(model, probes, tokenizer, args):
    """
    Pick the layer to read the heatmap at, the same way run.py picks it: highest
    AUROC on the Needham et al. set. Reading the map at an arbitrary layer would
    beg the question of which depth carries the signal.
    """
    from src.needham import load_needham_dataset  # Needham et al. (2025)

    needham = load_needham_dataset(args.eval_dataset, tokenizer, allowlist_path=args.needham_allowlist)
    results = evaluate_probes(model, probes, needham)
    best = results["best_layer"]
    auroc = {str(l): float(v["auroc"]) for l, v in results.items() if l != "best_layer"}
    return best, auroc


def load_heatmap(path):
    """
    Read a heatmap .npz back into a list of dicts.

    Each entry has: tokens (list[str]), projection and cosine (np.ndarray
    [n_layers, seq_len]), layers (np.ndarray), plus the sensitivity-dataset
    metadata fields and the prompt.
    """
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        layers = data["layers"]
        examples = []
        for i, entry in enumerate(meta["examples"]):
            examples.append({
                **entry,
                "layers": layers,
                "projection": data[f"proj_{i}"],
                "cosine": data[f"cos_{i}"],
            })
    return examples, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HuggingFace model ID or alias")
    parser.add_argument("--model-path", default=None, help="Local directory with model files")
    parser.add_argument("--load-probe", default=None, help="Path to a saved probes .pt file")
    parser.add_argument(
        "--train-probe", action="store_true",
        help=(
            "Train the probes instead of loading them (~2 min even at 70B). Probe training is "
            "deterministic — mean(positive) - mean(negative) — so this reproduces the exact "
            "vectors the main run used, and the result is cached to output/probes_<model>.pt."
        ),
    )
    parser.add_argument(
        "--contrastive-dataset", default=None,
        help="Contrastive set for --train-probe (default: run.py's default, 'dolly')",
    )
    parser.add_argument(
        "--examples", nargs="+", default=[DEFAULT_EXAMPLES],
        help=(
            f"One or more example-prompt JSON files (default: {DEFAULT_EXAMPLES}). "
            "Several files share a single model load — the expensive part — and each "
            "gets its own output .npz."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--layers", default=None, help="Comma-separated subset of layers (default: every layer in the probe file)")
    parser.add_argument(
        "--select-layer", action="store_true",
        help="Evaluate every probe on the Needham et al. set and record the best-AUROC layer in the output",
    )
    parser.add_argument("--eval-dataset", default=DEFAULT_EVAL_DATASET, help="Needham et al. dataset.json")
    parser.add_argument("--needham-allowlist", default=None, help="Needham conversation ID allowlist JSON")
    parser.add_argument(
        "--output", default=None,
        help="Output .npz path; only valid with a single --examples file "
             "(default: output/heatmap_<model>_<tag>.npz)",
    )
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    device = get_device(args.device)

    if bool(args.load_probe) == bool(args.train_probe):
        parser.error("Provide exactly one of --load-probe or --train-probe")

    if args.output and len(args.examples) > 1:
        parser.error("--output takes a single path; omit it when passing several --examples files")
    for path in args.examples:
        if not os.path.exists(path):
            parser.error(f"Examples file not found: {path}")

    model, tokenizer = load_model(
        args.model, device=device, dtype=dtype_map[args.dtype],
        n_devices=args.n_gpus, model_path=args.model_path,
    )

    if args.load_probe:
        probes = load_probes(args.load_probe)
        probe_source = args.load_probe
    else:
        probes = _train_probes_for(model, tokenizer, args)
        probe_source = f"trained:{args.contrastive_dataset or DEFAULT_CONTRASTIVE_DATASET}"

    layers = [int(l.strip()) for l in args.layers.split(",")] if args.layers else sorted(probes)
    missing = [l for l in layers if l not in probes]
    if missing:
        parser.error(f"Probe file has no vectors for layers {missing}")

    # A layer whose contrastive difference vector had zero norm produces a NaN
    # probe (layer 0 in every model trained so far). Capturing it would only
    # write NaN columns, so drop it here and say so.
    degenerate = [l for l in layers if not torch.isfinite(probes[l]["vector"]).all()]
    if degenerate:
        print(f"Skipping layers with non-finite probe vectors: {degenerate}")
        layers = [l for l in layers if l not in degenerate]
    if not layers:
        parser.error("No usable probe layers left after dropping non-finite vectors")

    best_layer, auroc_by_layer = (None, None)
    if args.select_layer:
        best_layer, auroc_by_layer = _select_best_layer(
            model, {l: probes[l] for l in layers}, tokenizer, args
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for examples_path in args.examples:
        with open(examples_path) as f:
            examples = json.load(f)
        print(f"\n--- {examples_path}: {len(examples)} example prompts ---")

        arrays = {"layers": np.asarray(layers, dtype=np.int32)}
        meta_examples = []
        for i, item in enumerate(examples):
            # A 'system' field reproduces the contrastive training format exactly;
            # sensitivity prompts have none and go through as a bare user turn.
            messages = [{"role": "user", "content": item["prompt"]}]
            if item.get("system"):
                messages.insert(0, {"role": "system", "content": item["system"]})
            formatted = apply_chat_template_with_fallback(tokenizer, messages)

            str_tokens, result = get_token_projections(model, probes, formatted, layers=layers)

            arrays[f"proj_{i}"] = result["projection"].numpy().astype(np.float32)
            arrays[f"cos_{i}"] = result["cosine"].numpy().astype(np.float32)
            meta_examples.append({
                "index": i,
                "prompt": item["prompt"],
                "formatted_prompt": formatted,
                "tokens": str_tokens,
                **{k: item[k] for k in _METADATA_FIELDS if k in item},
            })
            tag = item.get("scenario") or item.get("label", "?")
            detail = (f"structure={item['prompt_structure']}" if "prompt_structure" in item
                      else f"pair={item.get('pair_id', '?')}")
            print(f"[{i+1}/{len(examples)}] {tag} {detail} seq_len={len(str_tokens)}")

        meta = {
            "model": resolve_model_name(args.model),
            "model_alias": args.model,
            "probe_path": probe_source,
            "examples_path": examples_path,
            "thresholds": {str(l): float(probes[l]["threshold"]) for l in layers},
            "best_layer": best_layer,
            "auroc": auroc_by_layer,
            "examples": meta_examples,
        }
        arrays["meta"] = np.asarray(json.dumps(meta, ensure_ascii=False))

        out_path = args.output or os.path.join(
            OUTPUT_DIR,
            f"heatmap_{args.model.replace('/', '_')}_{examples_tag(examples_path)}.npz",
        )
        np.savez_compressed(out_path, **arrays)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Saved {len(examples)} examples x {len(layers)} layers to {out_path} ({size_mb:.1f} MB)")

    if best_layer is not None:
        print(f"\nBest layer by Needham AUROC: {best_layer} (AUROC={auroc_by_layer[str(best_layer)]:.4f})")


if __name__ == "__main__":
    main()
