#!/usr/bin/env python
"""
Select a small, deterministic set of sensitivity-dataset records for per-token
probe heatmaps.

The interesting contrast for the "which tokens carry the eval-awareness signal"
question is prompt_structure (0 = sloppy chat, 1 = imperative, 2 = markdown,
3 = multiple-choice benchmark), held against every scenario. All other factors
are pinned so the only thing varying inside a scenario is the structure.

Usage:
    python scripts/build_heatmap_examples.py
    python scripts/build_heatmap_examples.py --lang 0 --output datasets/heatmap_examples_es.json
"""
import argparse
import json

SENSITIVITY_DATASET = "datasets/sensitivity_dataset.json"
DEFAULT_OUTPUT = "datasets/heatmap_examples.json"

# Factors pinned so prompt_structure is the only axis that moves within a scenario.
PINNED = {"entity_familiarity": 1, "context_coherence": 1, "ethical_pressure": 0, "lang": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=SENSITIVITY_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--lang", type=int, default=PINNED["lang"], help="0=es, 1=en, 2=pt")
    parser.add_argument("--entity-familiarity", type=int, default=PINNED["entity_familiarity"])
    parser.add_argument("--context-coherence", type=int, default=PINNED["context_coherence"])
    parser.add_argument("--ethical-pressure", type=int, default=PINNED["ethical_pressure"])
    args = parser.parse_args()

    pinned = {
        "entity_familiarity": args.entity_familiarity,
        "context_coherence": args.context_coherence,
        "ethical_pressure": args.ethical_pressure,
        "lang": args.lang,
    }

    with open(args.dataset) as f:
        records = json.load(f)

    matching = [r for r in records if all(r[k] == v for k, v in pinned.items())]
    if not matching:
        raise SystemExit(f"No records match the pinned factors: {pinned}")

    # Deterministic: sort by (scenario, prompt_structure) and keep the first of each cell.
    selected = {}
    for r in sorted(matching, key=lambda r: (r["scenario"], r["prompt_structure"])):
        selected.setdefault((r["scenario"], r["prompt_structure"]), r)

    examples = [dict(r) for r in selected.values()]
    with open(args.output, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    scenarios = sorted({r["scenario"] for r in examples})
    structures = sorted({r["prompt_structure"] for r in examples})
    print(f"Wrote {len(examples)} examples to {args.output}")
    print(f"  pinned:     {pinned}")
    print(f"  scenarios:  {len(scenarios)} ({', '.join(scenarios)})")
    print(f"  structures: {structures}")


if __name__ == "__main__":
    main()
