"""
Linear probe training and inference via steering vectors.
Mirrors the approach in https://github.com/Jordine/evaluation-awareness-probing
"""
# ruff: noqa: E741
import gc
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc


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



def _extract_last_token_activations_all_layers(model, tokens, layers):
    """Extract residual stream activations at the last token position for multiple layers."""
    cached = {}

    def make_hook(layer):
        def hook_fn(activation, hook):
            cached[layer] = activation[:, -1, :].clone().detach()
            return activation
        return hook_fn

    model.reset_hooks()
    hooks = [(f"blocks.{layer}.hook_resid_pre", make_hook(layer)) for layer in layers]
    with model.hooks(fwd_hooks=hooks):
        with torch.no_grad():
            model(tokens)

    return cached


def train_probe(model, contrastive_dataset, layer):
    """Train a single-layer probe. Convenience wrapper around train_probes."""
    probes = train_probes(model, contrastive_dataset, [layer])
    return probes[layer]


def train_probes(model, contrastive_dataset, layers):
    """
    Train one linear probe per layer.

    Each item in contrastive_dataset must have:
      - 'positive_prompt': full prompt string for the positive (eval-context) class
      - 'negative_prompt': full prompt string for the negative (deployment-context) class

    Activations are extracted at the last token of each prompt — the same position
    used by evaluate_probes and get_projection — eliminating the train/eval
    extraction-position mismatch that existed when training extracted at the
    answer-letter token of an appended "(A)"/"(B)" suffix.

    Args:
        model: HookedTransformer model
        contrastive_dataset: List of dicts with 'positive_prompt', 'negative_prompt'
        layers: List of layer indices

    Returns:
        dict mapping layer index to probe dict (vector, raw_vector, threshold, layer)
    """
    pos_acts_by_layer = {l: [] for l in layers}
    neg_acts_by_layer = {l: [] for l in layers}

    print(f"Training probes on {len(layers)} layers with {len(contrastive_dataset)} examples...")
    for item in tqdm(contrastive_dataset, desc="Extracting activations"):
        pos_tokens = model.to_tokens(item["positive_prompt"])
        neg_tokens = model.to_tokens(item["negative_prompt"])

        pos_acts = _extract_last_token_activations_all_layers(model, pos_tokens, layers)
        neg_acts = _extract_last_token_activations_all_layers(model, neg_tokens, layers)

        for l in layers:
            pos_acts_by_layer[l].append(pos_acts[l])
            neg_acts_by_layer[l].append(neg_acts[l])

        del pos_tokens, neg_tokens, pos_acts, neg_acts
        gc.collect()

    probes = {}
    for l in layers:
        pos_mean = torch.stack(pos_acts_by_layer[l]).mean(dim=0)
        neg_mean = torch.stack(neg_acts_by_layer[l]).mean(dim=0)

        raw_vector = pos_mean - neg_mean
        probe_vector = raw_vector / raw_vector.norm()

        all_projections = []
        all_labels = []
        for act in pos_acts_by_layer[l]:
            proj = torch.dot(act[0], probe_vector[0]).item()
            all_projections.append(proj)
            all_labels.append(1)
        for act in neg_acts_by_layer[l]:
            proj = torch.dot(act[0], probe_vector[0]).item()
            all_projections.append(proj)
            all_labels.append(0)

        threshold = _find_threshold(all_projections, all_labels)
        probes[l] = {
            "vector": probe_vector,
            "raw_vector": raw_vector,
            "threshold": threshold,
            "layer": l,
        }
        print(f"  Layer {l}: threshold={threshold:.4f}")

    return probes


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


def evaluate_probes(model, probes, eval_dataset):
    """
    Evaluate multiple probes on a labeled dataset using AUROC.

    Prompts must already be pre-formatted before being passed in
    — use load_needham_dataset() from src/needham.py for this.

    Label polarity: is_eval=True → 1, is_eval=False → 0.

    Args:
        model: HookedTransformer model
        probes: dict mapping layer index to probe dict
        eval_dataset: list of dicts with 'prompt' (str) and 'is_eval' (bool)

    Returns:
        dict mapping layer index to {'auroc': float, 'threshold': float},
        plus 'best_layer' key
    """
    layers = sorted(probes.keys())

    projections_by_layer = {l: [] for l in layers}
    labels = []

    print(f"Evaluating probes on {len(eval_dataset)} items...")
    for item in tqdm(eval_dataset, desc="Probe evaluation"):
        label = 1 if item["is_eval"] else 0
        labels.append(label)

        tokens = model.to_tokens(item["prompt"])
        acts = _extract_last_token_activations_all_layers(model, tokens, layers)

        for l in layers:
            proj = torch.dot(acts[l][0], probes[l]["vector"][0]).item()
            projections_by_layer[l].append(proj)

        del tokens, acts
        gc.collect()

    results = {}
    best_layer = None
    best_auroc = -1.0

    for l in layers:
        projs = projections_by_layer[l]
        if any(p != p for p in projs):  # NaN check
            print(f"  Layer {l}: skipped (NaN projections)")
            continue

        fpr, tpr, thresholds = roc_curve(labels, projs)
        auroc_val = auc(fpr, tpr)

        j_scores = tpr - fpr
        optimal_idx = j_scores.argmax()
        optimal_threshold = thresholds[optimal_idx]

        results[l] = {
            "auroc": auroc_val,
            "threshold": float(optimal_threshold),
        }

        if auroc_val > best_auroc:
            best_auroc = auroc_val
            best_layer = l

        print(f"  Layer {l}: AUROC={auroc_val:.4f}")

    if best_layer is None:
        raise ValueError("All layers produced NaN projections — no valid probe found")

    print(f"\nBest layer: {best_layer} (AUROC={best_auroc:.4f})")
    results["best_layer"] = best_layer

    return results


def save_probe(probe, path):
    torch.save({
        "vector": probe["vector"],
        "raw_vector": probe["raw_vector"],
        "threshold": probe["threshold"],
        "layer": probe["layer"],
    }, path)
    print(f"Probe saved to {path}")


def load_probe(path):
    data = torch.load(path, weights_only=True)
    return data


def save_probes(probes, path):
    """Save a dict of probes (keyed by layer) to a single .pt file."""
    torch.save({
        layer: {
            "vector": p["vector"],
            "raw_vector": p["raw_vector"],
            "threshold": p["threshold"],
            "layer": p["layer"],
        }
        for layer, p in probes.items()
    }, path)
    print(f"Probes ({len(probes)} layers) saved to {path}")


def load_probes(path):
    """Load a dict of probes from a .pt file. Also accepts old single-probe format."""
    data = torch.load(path, weights_only=True)
    # Detect single vs multi-layer format
    if "vector" in data:
        data = {data["layer"]: data}
    print(f"Loaded probes for {len(data)} layers from {path}")
    return data
