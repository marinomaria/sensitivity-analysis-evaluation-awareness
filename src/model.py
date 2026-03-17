"""
Model loading and text generation for Llama (and other HuggingFace open-weights models).
Designed to make switching between models easy.
"""
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# Known model aliases for convenience
MODEL_ALIASES = {
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-70b": "meta-llama/Llama-3.3-70B-Instruct",
}


def resolve_model_name(name):
    return MODEL_ALIASES.get(name, name)


def get_device(requested=None):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name, device=None, dtype=torch.bfloat16):
    """
    Load a HookedTransformer model and its tokenizer.

    Args:
        model_name: HuggingFace model ID or alias (see MODEL_ALIASES)
        device: Target device (defaults to best available)
        dtype: torch dtype

    Returns:
        tuple: (model, tokenizer)
    """
    model_name = resolve_model_name(model_name)
    device = get_device(device)

    print(f"Loading model '{model_name}' on {device} ({dtype})...")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    print(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=200):
    """
    Generate a text response for a prompt.

    Args:
        model: HookedTransformer model
        tokenizer: tokenizer
        prompt: input string
        max_new_tokens: maximum tokens to generate

    Returns:
        str: generated text (excluding the prompt)
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.cfg.device)

    model.reset_hooks()
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)

    # Decode only the newly generated tokens
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
