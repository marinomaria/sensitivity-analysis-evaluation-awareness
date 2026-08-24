#!/usr/bin/env python
"""
Turn the contrastive training pairs into an examples file for token heatmaps.

The probe vector is defined as mean(eval-framed activations) - mean(deploy-framed
activations) at the last token. Running the probe token-by-token over those very
prompts shows where the direction lives in the data that defined it: on the
"you are being evaluated" sentence, or on the chat scaffolding that follows it.

If the signal sits on the system sentence during training but on the template tail
at inference time, the probe generalises by position rather than by concept.

Each selected stem yields two examples — one per framing — sharing a pair_id, so
their maps can be read side by side.

Usage:
    python scripts/build_contrastive_examples.py
    python scripts/build_contrastive_examples.py --n-pairs 20 --dataset legacy
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import _CONTRASTIVE_DATASETS, _DEPLOY_SYSTEM, _EVAL_SYSTEM, _contrastive_stem

DEFAULT_OUTPUT = "datasets/contrastive_examples.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="dolly", choices=list(_CONTRASTIVE_DATASETS))
    parser.add_argument("--n-pairs", type=int, default=12, help="Stems to include (default: 12 → 24 examples)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with open(_CONTRASTIVE_DATASETS[args.dataset]) as f:
        raw = json.load(f)

    # Evenly spaced rather than the first N, so the sample spans the whole file.
    step = max(1, len(raw) // args.n_pairs)
    selected = [raw[i] for i in range(0, len(raw), step)][: args.n_pairs]

    examples = []
    for pair_id, entry in enumerate(selected):
        stem = _contrastive_stem(entry)
        for label, system in (("eval", _EVAL_SYSTEM), ("deploy", _DEPLOY_SYSTEM)):
            examples.append({"prompt": stem, "system": system, "label": label, "pair_id": pair_id})

    with open(args.output, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(examples)} examples ({len(selected)} pairs) from {args.dataset} to {args.output}")


if __name__ == "__main__":
    main()
