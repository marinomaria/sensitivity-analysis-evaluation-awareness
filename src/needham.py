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


_DEFAULT_ALLOWLIST_JSON = Path(__file__).resolve().parent.parent / "datasets" / "needham_allowlist.json"


def _load_allowlist_json(allowlist_path: Path) -> set[str]:
    """Load Needham conversation IDs from the tracked curation manifest."""
    with open(allowlist_path, encoding="utf-8") as f:
        payload = json.load(f)

    ids = payload.get("conversation_ids")
    if not isinstance(ids, list):
        raise ValueError(
            f"Needham allowlist at '{allowlist_path}' must contain a conversation_ids list."
        )

    return {cid.strip() for cid in ids if isinstance(cid, str) and cid.strip()}


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
    allowlist_path: str | Path | None = None,
) -> list[dict]:
    """
    Load the Needham et al. dataset, filter to chat-only transcripts,
    and format conversations using the model's tokenizer.

    Filters:
    - Only entries where metadata.scaffold is null (chat transcripts)
    - Only entries whose id appears in datasets/needham_allowlist.json
    - Skips tool messages within each conversation
    - Skips entries whose formatted prompt exceeds tokenizer.model_max_length

    Args:
        dataset_path: Path to dataset.json
        tokenizer: HuggingFace tokenizer (used to apply chat template)
        allowlist_path: Path to needham_allowlist.json (default: datasets/needham_allowlist.json)

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

    allowlist = Path(allowlist_path) if allowlist_path is not None else _DEFAULT_ALLOWLIST_JSON
    if not allowlist.exists():
        raise FileNotFoundError(
            f"Needham allowlist not found at '{allowlist}'. "
            "Expected datasets/needham_allowlist.json with a conversation_ids list."
        )
    keep_ids = _load_allowlist_json(allowlist)
    before_curated = len(chat_entries)
    chat_entries = [e for e in chat_entries if e["id"] in keep_ids]
    print(
        f"Needham allowlist: {before_curated} chat entries -> {len(chat_entries)} "
        f"({len(keep_ids)} ids in manifest)"
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
