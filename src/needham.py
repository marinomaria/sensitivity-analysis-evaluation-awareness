"""
Needham et al. dataset loading and formatting for probe evaluation.

Dataset: "Large Language Models Often Know They Are Being Evaluated" (Needham et al.)
HuggingFace: https://huggingface.co/datasets/jjpn2/eval_awareness

The dataset is an external dependency — it must be obtained separately.
See README for setup instructions.
"""
import csv
import json
from pathlib import Path

from src.model import apply_chat_template_with_fallback


_DEFAULT_CURATED_CSV = Path(__file__).resolve().parent.parent / "datasets" / "needham_curated.csv"


def _conversation_ids_with_keep_true(curated_csv_path: Path) -> set[str]:
    """Load conversation_id values where the curated spreadsheet marks keep=TRUE."""
    ids: set[str] = set()
    with open(curated_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("keep", "").strip().upper() != "TRUE":
                continue
            cid = (row.get("conversation_id") or "").strip()
            if cid:
                ids.add(cid)
    return ids


SUPPORTED_ROLES = {"system", "user", "assistant"}


def _normalize_content(content):
    """Flatten OpenAI multi-part content (list of dicts) into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
        )
    return str(content)


def load_needham_dataset(
    dataset_path: str,
    tokenizer,
    *,
    curated_csv_path: str | Path | None = None,
) -> list[dict]:
    """
    Load the Needham et al. dataset, filter to chat-only transcripts,
    and format conversations using the model's tokenizer.

    Filters:
    - Only entries where metadata.scaffold is null (chat transcripts)
    - Only entries whose id appears in datasets/needham_curated.csv with keep=TRUE
    - Skips tool messages within each conversation
    - Skips entries whose formatted prompt exceeds tokenizer.model_max_length

    Args:
        dataset_path: Path to dataset.json
        tokenizer: HuggingFace tokenizer (used to apply chat template)
        curated_csv_path: Path to needham_curated.csv (default: datasets/needham_curated.csv)

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

    curated_path = Path(curated_csv_path) if curated_csv_path is not None else _DEFAULT_CURATED_CSV
    if not curated_path.exists():
        raise FileNotFoundError(
            f"Curated allowlist not found at '{curated_path}'. "
            "Expected datasets/needham_curated.csv with conversation_id and keep columns."
        )
    keep_ids = _conversation_ids_with_keep_true(curated_path)
    before_curated = len(chat_entries)
    chat_entries = [e for e in chat_entries if e["id"] in keep_ids]
    print(
        f"Curated filter (keep=TRUE): {before_curated} chat entries -> {len(chat_entries)} "
        f"({len(keep_ids)} ids in spreadsheet)"
    )

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
            {"role": m["role"], "content": _normalize_content(m["content"])}
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
