"""
Needham et al. dataset loading and formatting for probe evaluation.

Dataset: "Large Language Models Often Know They Are Being Evaluated" (Needham et al.)
HuggingFace: https://huggingface.co/datasets/jjpn2/eval_awareness

The dataset is an external dependency — it must be obtained separately.
See README for setup instructions.
"""
import json
from pathlib import Path

from src.model import apply_chat_template_with_fallback


SUPPORTED_ROLES = {"system", "user", "assistant"}


def load_needham_dataset(dataset_path: str, tokenizer) -> list[dict]:
    """
    Load the Needham et al. dataset, filter to chat-only transcripts,
    and format conversations using the model's tokenizer.

    Filters:
    - Only entries where metadata.scaffold is null (chat transcripts)
    - Skips tool messages within each conversation
    - Skips entries whose formatted prompt exceeds tokenizer.model_max_length

    Args:
        dataset_path: Path to dataset.json
        tokenizer: HuggingFace tokenizer (used to apply chat template)

    Returns:
        List of dicts with keys:
          - prompt: str (formatted via tokenizer.apply_chat_template)
          - is_eval: bool (ground truth from metadata.eval_category)
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. "
            "See README for setup instructions: clone the Needham et al. (eval_awareness) repo "
            "and run scripts/decrypt.sh, then pass --eval-dataset <path>."
        )

    with open(path) as f:
        raw = json.load(f)

    # Filter to chat-only (scaffold is null)
    chat_entries = [e for e in raw if e["metadata"]["scaffold"] is None]

    n_eval = sum(1 for e in chat_entries if e["metadata"]["eval_category"])
    n_organic = len(chat_entries) - n_eval
    print(
        f"Loaded {len(chat_entries)} chat entries "
        f"({n_eval} eval, {n_organic} organic) from Needham et al. dataset"
    )

    result = []
    n_skipped = 0

    for entry in chat_entries:
        # Strip unsupported roles (e.g. tool)
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in entry["input"]
            if m["role"] in SUPPORTED_ROLES and m.get("content") is not None
        ]

        if not messages:
            n_skipped += 1
            continue

        prompt = apply_chat_template_with_fallback(tokenizer, messages)

        # Skip entries that exceed the model's context window
        token_ids = tokenizer.encode(prompt)
        if len(token_ids) > tokenizer.model_max_length:
            n_skipped += 1
            continue

        result.append({
            "prompt": prompt,
            "is_eval": bool(entry["metadata"]["eval_category"]),
        })

    if n_skipped > 0:
        print(f"Skipped {n_skipped} entries (empty after filtering or over context limit)")

    return result
