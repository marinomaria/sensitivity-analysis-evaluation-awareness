"""
Linear probe training and inference via steering vectors.
Mirrors the approach in https://github.com/Jordine/evaluation-awareness-probing
"""
import gc
import torch
from tqdm import tqdm


def _extract_activation_at_token(model, tokens, token_type, layer, tokenizer):
    """Extract residual stream activation at the last occurrence of token_type."""
    cached = {}

    def hook_fn(activation, hook):
        positions = (tokens == tokenizer.convert_tokens_to_ids(token_type)).nonzero()
        last_pos = positions[-1, -1]
        cached["act"] = activation[:, last_pos, :].clone().detach()
        return activation

    model.reset_hooks()
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook_fn)]):
        with torch.no_grad():
            model(tokens)

    return cached["act"]


def _extract_last_token_activation(model, tokens, layer):
    """Extract residual stream activation at the last token position."""
    cached = {}

    def hook_fn(activation, hook):
        cached["act"] = activation[:, -1, :].clone().detach()
        return activation

    model.reset_hooks()
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook_fn)]):
        with torch.no_grad():
            model(tokens)

    return cached["act"]


def train_probe(model, contrastive_dataset, layer):
    """
    Train a linear probe as a steering vector (mean_positive - mean_negative activations).

    Args:
        model: HookedTransformer model
        contrastive_dataset: List of dicts with 'question', 'positive', 'negative' keys
        layer: Which layer to extract activations from

    Returns:
        dict with 'vector' (normalized probe), 'threshold', 'raw_vector'
    """
    tokenizer = model.tokenizer
    pos_acts = []
    neg_acts = []

    print(f"Training probe on layer {layer} with {len(contrastive_dataset)} examples...")
    for item in tqdm(contrastive_dataset, desc="Extracting activations"):
        question = item["question"]
        positive = item["positive"]
        negative = item["negative"]

        pos_token_type = positive[1]  # e.g. "(A)" -> "A"
        neg_token_type = negative[1]

        pos_prompt = f"{question}\n\n {positive}"
        neg_prompt = f"{question}\n\n {negative}"

        pos_tokens = model.to_tokens(pos_prompt)
        neg_tokens = model.to_tokens(neg_prompt)

        pos_act = _extract_activation_at_token(model, pos_tokens, pos_token_type, layer, tokenizer)
        neg_act = _extract_activation_at_token(model, neg_tokens, neg_token_type, layer, tokenizer)

        pos_acts.append(pos_act)
        neg_acts.append(neg_act)

        del pos_tokens, neg_tokens, pos_act, neg_act
        gc.collect()

    pos_mean = torch.stack(pos_acts).mean(dim=0)
    neg_mean = torch.stack(neg_acts).mean(dim=0)

    raw_vector = pos_mean - neg_mean
    probe_vector = raw_vector / raw_vector.norm()

    # Compute threshold from training data projections
    all_projections = []
    all_labels = []
    for act in pos_acts:
        proj = torch.dot(act[0], probe_vector[0]).item()
        all_projections.append(proj)
        all_labels.append(1)
    for act in neg_acts:
        proj = torch.dot(act[0], probe_vector[0]).item()
        all_projections.append(proj)
        all_labels.append(0)

    threshold = _find_threshold(all_projections, all_labels)
    print(f"Probe trained. Threshold: {threshold:.4f}")

    return {
        "vector": probe_vector,
        "raw_vector": raw_vector,
        "threshold": threshold,
        "layer": layer,
    }


def _find_threshold(projections, labels):
    """Find the threshold that maximizes accuracy on training data."""
    sorted_projs = sorted(set(projections))
    best_threshold = 0.0
    best_acc = -1

    for candidate in sorted_projs:
        preds = [1 if p >= candidate else 0 for p in projections]
        acc = sum(p == l for p, l in zip(preds, labels)) / len(labels) # noqa: E741
        if acc > best_acc:
            best_acc = acc
            best_threshold = candidate

    return best_threshold


def get_projection(model, probe, prompt):
    """
    Project the last-token activation of prompt onto the probe vector.

    Args:
        model: HookedTransformer model
        probe: dict returned by train_probe
        prompt: input string

    Returns:
        float: projection value
    """
    layer = probe["layer"]
    probe_vector = probe["vector"]

    tokens = model.to_tokens(prompt)
    act = _extract_last_token_activation(model, tokens, layer)
    projection = torch.dot(act[0], probe_vector[0]).item()

    return projection


def get_verdict(projection, probe):
    """Return True if projection >= threshold (evaluation-aware)."""
    return projection >= probe["threshold"]


def save_probe(probe, path):
    torch.save({
        "vector": probe["vector"],
        "raw_vector": probe["raw_vector"],
        "threshold": probe["threshold"],
        "layer": probe["layer"],
    }, path)
    print(f"Probe saved to {path}")


def load_probe(path):
    data = torch.load(path)
    return data
