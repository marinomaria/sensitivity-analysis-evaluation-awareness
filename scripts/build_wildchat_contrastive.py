#!/usr/bin/env python
"""
Build datasets/contrastive_dataset_wildchat.json from WildChat-1M.
Run from project root: python scripts/build_wildchat_contrastive.py

Canonical tokenizer: Qwen/Qwen2.5-72B-Instruct
Pin WILDCHAT_REVISION to the commit hash printed on first run.
Note: WildChat-1M is large (~10 GB); first load will download and cache.
"""
import json
import os
import random
from datasets import load_dataset
from transformers import AutoTokenizer

WILDCHAT_REPO     = "allenai/WildChat-1M"
WILDCHAT_REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
OUTPUT            = "datasets/contrastive_dataset_wildchat.json"
TOKENIZER_ID      = "Qwen/Qwen2.5-72B-Instruct"
SEED              = 42
N_PAIRS           = 204
TOKEN_CAP         = 60

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
    ds = load_dataset(WILDCHAT_REPO, revision=WILDCHAT_REVISION, split="train")
    print(f"Loaded WildChat-1M revision: {WILDCHAT_REVISION}, {len(ds)} rows")

    entries = []
    for row in ds:
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        if first.get("language", "English") != "English":
            continue
        if first.get("toxic", False):
            continue
        if first.get("redacted", False):
            continue
        text = first.get("content", "").strip()
        if not text:
            continue
        if not text.isascii():
            continue
        if len(tokenizer.encode(text)) < 5:
            continue
        if not passes_blocklist(text):
            continue
        if len(tokenizer.encode(text)) > TOKEN_CAP:
            continue
        entries.append(text)

    # Deduplicate
    entries = list(dict.fromkeys(entries))
    print(f"{len(entries)} unique entries passed filter (need {N_PAIRS})")

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
