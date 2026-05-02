#!/usr/bin/env python
"""
Build datasets/contrastive_dataset_dolly.json from Dolly-15k.
Run from project root: python scripts/build_dolly_contrastive.py

Canonical tokenizer: Qwen/Qwen2.5-72B-Instruct
Pin DOLLY_REVISION to the commit hash printed on first run.
"""
import json
import os
import random
from datasets import load_dataset
from transformers import AutoTokenizer

DOLLY_REPO     = "databricks/databricks-dolly-15k"
DOLLY_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
OUTPUT         = "datasets/contrastive_dataset_dolly.json"
TOKENIZER_ID   = "Qwen/Qwen2.5-72B-Instruct"
SEED           = 42
N_PAIRS        = 204
TOKEN_CAP      = 60
CATEGORIES     = {"open_qa", "classification", "generation", "brainstorming"}

BLOCKLIST = [
    "ai", "llm", "gpt", "chatgpt", "claude", "anthropic", "openai",
    "language model", "benchmark", "evaluat", "neural", "machine learning",
    "assistant", "chatbot", "dataset", "prompt engineer",
    "jailbreak", "nsfw", "aware", "training data", "model",
]


def passes_blocklist(text):
    t = text.lower()
    return not any(kw in t for kw in BLOCKLIST)


def main():
    assert os.path.isdir("datasets"), "Run from project root (datasets/ not found)"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    ds = load_dataset(DOLLY_REPO, revision=DOLLY_REVISION, split="train")
    print(f"Loaded Dolly-15k revision: {DOLLY_REVISION}")

    entries = []
    for row in ds:
        if row["category"] not in CATEGORIES:
            continue
        if row.get("context", "").strip():
            continue
        text = row["instruction"].strip()
        if not passes_blocklist(text):
            continue
        if len(tokenizer.encode(text)) > TOKEN_CAP:
            continue
        entries.append(text)

    print(f"{len(entries)} entries passed filter (need {N_PAIRS})")

    if len(entries) < N_PAIRS:
        print("Warning: fewer entries than target — writing all available")
        sampled = entries
    else:
        random.seed(SEED)
        sampled = random.sample(entries, N_PAIRS)

    # positive/negative are schema placeholders — load_contrastive_dataset() only reads 'question'
    output = [{"question": q, "positive": "(A)", "negative": "(B)"} for q in sampled]

    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(output)} pairs to {OUTPUT}")


if __name__ == "__main__":
    main()
