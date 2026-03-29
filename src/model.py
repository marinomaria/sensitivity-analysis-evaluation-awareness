"""
Model loading and text generation for Llama (and other HuggingFace open-weights models).
Designed to make switching between models easy.
"""
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Known model aliases for convenience.
# TransformerLens compatibility:
#   - Qwen2.5 and DeepSeek-R1-Distill-Qwen-* use the Qwen2 architecture (supported).
#   - DeepSeek-R1-Distill-Llama-* use the Llama architecture (supported).
#   - DeepSeek-V2/V3/R1 (full) use MoE and are NOT supported by TransformerLens.
MODEL_ALIASES = {
    # Llama
    "llama-1b":  "meta-llama/Llama-3.2-1B-Instruct",
    "llama-70b": "meta-llama/Llama-3.3-70B-Instruct",
    # Qwen2.5
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-7b":   "Qwen/Qwen2.5-7B-Instruct",
    "qwen-72b":  "Qwen/Qwen2.5-72B-Instruct",
    # DeepSeek-R1-Distill (Qwen base)
    "deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-r1-7b":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    # DeepSeek-R1-Distill (Llama base)
    "deepseek-r1-8b":   "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-r1-70b":  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}

# Maps models not in TransformerLens's whitelist to a compatible base architecture.
# DeepSeek-R1-Distill models are architectural clones of their base models.
_TL_ARCHITECTURE_MAP = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":   "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":   "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B":  "meta-llama/Llama-3.3-70B-Instruct",
}


def apply_chat_template_with_fallback(tokenizer, messages):
    """
    Apply the model's chat template.

    Falls back to prepending the system message to the first user turn for
    models whose template does not support a system role.
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as original_exc:
        flat = []
        sys_content = None
        for m in messages:
            if m["role"] == "system":
                sys_content = m["content"]
            elif m["role"] == "user" and sys_content is not None:
                flat.append({"role": "user", "content": f"{sys_content}\n\n{m['content']}"})
                sys_content = None
            else:
                flat.append(m)
        if sys_content is not None:
            raise ValueError(
                "System message found with no subsequent user message; cannot apply fallback."
            ) from original_exc
        try:
            return tokenizer.apply_chat_template(
                flat, tokenize=False, add_generation_prompt=True
            )
        except Exception as fallback_exc:
            raise original_exc from fallback_exc


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
    tl_name = _TL_ARCHITECTURE_MAP.get(model_name)
    if tl_name:
        # Model not in TransformerLens whitelist — load HF weights and map via base architecture
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        model = HookedTransformer.from_pretrained(tl_name, hf_model=hf_model, device=device, dtype=dtype)
        del hf_model
    else:
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
