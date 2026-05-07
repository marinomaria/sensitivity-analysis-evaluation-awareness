"""
Model alias registry — kept torch-free so bash wrappers and lightweight tools
can import it without paying the cost of loading torch / transformer_lens.

src/model.py re-exports these names; downstream code should keep importing
from src.model.
"""

# Known model aliases for convenience.
# TransformerLens compatibility:
#   - Qwen2.5 and DeepSeek-R1-Distill-Qwen-* use the Qwen2 architecture (supported).
#   - DeepSeek-R1-Distill-Llama-* use the Llama architecture (supported).
#   - DeepSeek-V2/V3/R1 (full) use MoE and are NOT supported by TransformerLens.
MODEL_ALIASES = {
    # Llama
    "llama-1b":  "meta-llama/Llama-3.2-1B-Instruct",
    "llama-8b":  "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.3-70B-Instruct",
    # Qwen2.5
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-7b":   "Qwen/Qwen2.5-7B-Instruct",
    "qwen-32b":  "Qwen/Qwen2.5-32B-Instruct",
    "qwen-72b":  "Qwen/Qwen2.5-72B-Instruct",
    # Gemma 2
    "gemma-9b":  "google/gemma-2-9b-it",
    "gemma-27b": "google/gemma-2-27b-it",
    # QwQ (Qwen 2.5 32B reasoning fine-tune)
    "qwq-32b":   "Qwen/QwQ-32B",
    # DeepSeek-R1-Distill (Qwen base)
    "deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-r1-7b":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    # DeepSeek-R1-Distill (Llama base)
    "deepseek-r1-8b":   "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-r1-70b":  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}

# Maps models not in TransformerLens's whitelist to a compatible base architecture.
# DeepSeek-R1-Distill models are architectural clones of their base models;
# QwQ-32B is a Qwen 2.5 32B fine-tune.
TL_ARCHITECTURE_MAP = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":   "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":   "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B":  "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/QwQ-32B":                              "Qwen/Qwen2.5-32B-Instruct",
}


def resolve_model_name(name):
    """Look up an alias; fall back to the input when not registered."""
    return MODEL_ALIASES.get(name, name)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python -m src.aliases <alias_or_repo_id>\n")
        sys.exit(2)
    print(resolve_model_name(sys.argv[1]))
