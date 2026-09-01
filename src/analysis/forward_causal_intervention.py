"""
M4 - Step 2: Forward Intervention (Layer-Level) -> M_fwd
===========================================================

WHAT THIS DOES
---------------
For each source layer i in {0..5}:
    1. Run a batch of sentences through DistilBERT CLEAN (no intervention).
       Record effective_rank / stable_rank / mean_cosine_sim at every layer's output.
    2. Run the SAME batch again, but this time force layer i's output to
       collapse to a low rank (SVD truncation) via a forward hook.
       Record the same three metrics at every layer's output.
    3. For every downstream layer j >= i, compute the damage:
           M_fwd[i][j] = clean_metric[j] - intervened_metric[j]
       (a positive number = geometric diversity was LOST at j because i broke)

Run this once per task (SST-2 / MNLI / CoNLL-2003) by pointing MODEL_CHECKPOINT_PATH
at that task's fine-tuned model.

OUTPUT
------
A JSON file per (task, metric) matching the schema agreed in the base reference doc:
    {
        "task": ...,
        "direction": "forward",
        "granularity": "layer",
        "matrix": [[...6x6...]],
        "metric": ...,
        "target_rank_used": ...,
    }
"""

import json
import copy
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# -----------------------------------------------------------------------
# CONFIG - edit these per run
# -----------------------------------------------------------------------
MODEL_CHECKPOINT_PATH = "distilbert-base-uncased"  # swap for your fine-tuned SST-2/MNLI/CoNLL checkpoint
TASK_NAME = "sst2"                                  # "sst2" | "mnli" | "conll2003"
TARGET_RANK = 32                                    # how low we force the intervened layer's rank
NUM_LAYERS = 6                                       # DistilBERT has 6 transformer blocks
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A small batch of example sentences to probe with.
# Swap this out for a real batch pulled from your SST-2 / MNLI / CoNLL dataloader.
EXAMPLE_SENTENCES = [
    "The movie was surprisingly delightful and full of heart.",
    "I would not recommend this restaurant to anyone.",
    "The committee approved the new budget after lengthy debate.",
    "She quickly realized the plan would not work as intended.",
]


# -----------------------------------------------------------------------
# GEOMETRIC METRICS (same three metrics used throughout M3 / the baseline)
# -----------------------------------------------------------------------
def effective_rank(H: torch.Tensor) -> float:
    """
    H: [seq_len, hidden_dim] token matrix for ONE sentence.
    Effective rank = exp(entropy of normalized singular values).
    This is a continuous, "soft" measure of how many dimensions are
    actually being used - not just a hard rank count.
    """
    H = H.detach().cpu().to(torch.float32)
    s = torch.linalg.svdvals(H)
    s = s[s > 1e-12]
    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy).item()


def stable_rank(H: torch.Tensor) -> float:
    """
    Stable rank = ||H||_F^2 / ||H||_2^2 (Frobenius norm sq / top singular value sq).
    Cheaper, more robust-to-noise cousin of effective rank.
    """
    H = H.detach().cpu().to(torch.float32)
    s = torch.linalg.svdvals(H)
    frob_sq = (s ** 2).sum()
    top_sq = s[0] ** 2
    return (frob_sq / top_sq).item()


def mean_token_cosine_sim(H: torch.Tensor) -> float:
    """
    Average pairwise cosine similarity between all token vectors.
    High value = tokens are becoming identical (collapse / homogenization).
    """
    H = H.detach().cpu().to(torch.float32)
    H_norm = torch.nn.functional.normalize(H, dim=-1)
    sim_matrix = H_norm @ H_norm.T
    seq_len = H.shape[0]
    mask = ~torch.eye(seq_len, dtype=torch.bool)
    return sim_matrix[mask].mean().item()


METRIC_FUNCS = {
    "effective_rank": effective_rank,
    "stable_rank": stable_rank,
    "mean_cosine_sim": mean_token_cosine_sim,
}


# -----------------------------------------------------------------------
# SVD LOW-RANK COLLAPSE (the actual "damage" we inject)
# -----------------------------------------------------------------------
def collapse_to_rank(H: torch.Tensor, target_rank: int) -> torch.Tensor:
    """
    H: [batch, seq_len, hidden_dim]
    Truncates each sentence's token matrix to `target_rank` via SVD and
    reconstructs. This is what "forcing a layer to collapse" means concretely.
    """
    orig_dtype = H.dtype
    orig_device = H.device
    out = torch.empty_like(H)
    for b in range(H.shape[0]):
        mat = H[b].detach().to(torch.float32)
        U, S, Vt = torch.linalg.svd(mat, full_matrices=False)
        k = min(target_rank, S.shape[0])
        S_trunc = S.clone()
        S_trunc[k:] = 0.0
        recon = U @ torch.diag(S_trunc) @ Vt
        out[b] = recon.to(orig_dtype)
    return out.to(orig_device)


# -----------------------------------------------------------------------
# HOOK MACHINERY
# -----------------------------------------------------------------------
class LayerRecorder:
    """
    Registers a forward hook on every transformer layer that just RECORDS
    the output hidden state (no modification). Used for both the clean
    pass and the intervened pass, to capture per-layer activations.
    """
    def __init__(self, model):
        self.activations = {}  # layer_idx -> tensor [batch, seq_len, hidden]
        self.handles = []
        for i, layer in enumerate(model.transformer.layer):
            handle = layer.register_forward_hook(self._make_hook(i))
            self.handles.append(handle)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            # DistilBERT layer output is a tuple; hidden state is output[0]
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden.detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


class LayerCollapser:
    """
    Registers a forward hook on ONE specific layer (source_layer) that
    REPLACES its output with a rank-collapsed version, then lets everything
    downstream run on the corrupted activation.
    """
    def __init__(self, model, source_layer, target_rank):
        self.handle = None
        self.source_layer = source_layer
        self.target_rank = target_rank
        layer = model.transformer.layer[source_layer]
        self.handle = layer.register_forward_hook(self._collapse_hook)

    def _collapse_hook(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            collapsed = collapse_to_rank(hidden, self.target_rank)
            return (collapsed,) + output[1:]
        else:
            return collapse_to_rank(output, self.target_rank)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()


# -----------------------------------------------------------------------
# MAIN EXPERIMENT LOOP
# -----------------------------------------------------------------------
def run_clean_pass(model, tokenizer, sentences):
    """Returns dict: layer_idx -> list of per-sentence [seq_len, hidden] activations."""
    recorder = LayerRecorder(model)
    per_sentence_acts = {i: [] for i in range(NUM_LAYERS)}

    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model(**inputs)
        for layer_idx, act in recorder.activations.items():
            # act: [1, seq_len, hidden] -> [seq_len, hidden]
            per_sentence_acts[layer_idx].append(act[0])

    recorder.remove()
    return per_sentence_acts


def run_intervened_pass(model, tokenizer, sentences, source_layer, target_rank):
    """Same as run_clean_pass but with layer `source_layer` forced to collapse."""
    recorder = LayerRecorder(model)
    collapser = LayerCollapser(model, source_layer, target_rank)
    per_sentence_acts = {i: [] for i in range(NUM_LAYERS)}

    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model(**inputs)
        for layer_idx, act in recorder.activations.items():
            per_sentence_acts[layer_idx].append(act[0])

    collapser.remove()
    recorder.remove()
    return per_sentence_acts


def compute_metric_per_layer(per_sentence_acts, metric_fn):
    """Averages a metric across all probe sentences, for each layer."""
    layer_scores = {}
    for layer_idx, acts_list in per_sentence_acts.items():
        scores = [metric_fn(act) for act in acts_list]
        layer_scores[layer_idx] = float(np.mean(scores))
    return layer_scores


def build_forward_matrix(model, tokenizer, sentences, metric_name):
    metric_fn = METRIC_FUNCS[metric_name]

    print(f"\n[1/2] Running CLEAN baseline pass ({metric_name})...")
    clean_acts = run_clean_pass(model, tokenizer, sentences)
    clean_scores = compute_metric_per_layer(clean_acts, metric_fn)
    print(f"      Clean per-layer {metric_name}: {clean_scores}")

    M_fwd = np.zeros((NUM_LAYERS, NUM_LAYERS))

    for source_layer in range(NUM_LAYERS):
        print(f"\n[2/2] Intervening at layer {source_layer} "
              f"(collapsing to rank {TARGET_RANK})...")
        intervened_acts = run_intervened_pass(
            model, tokenizer, sentences, source_layer, TARGET_RANK
        )
        intervened_scores = compute_metric_per_layer(intervened_acts, metric_fn)

        for j in range(NUM_LAYERS):
            if j < source_layer:
                # Layers before the intervention point are untouched -
                # damage can't flow backward in a feedforward stack.
                M_fwd[source_layer][j] = 0.0
            else:
                damage = clean_scores[j] - intervened_scores[j]
                M_fwd[source_layer][j] = damage

        row_avg = M_fwd[source_layer].mean()
        print(f"      Row {source_layer} avg damage: {row_avg:.4f}  "
              f"(per-layer: {[f'{v:.3f}' for v in M_fwd[source_layer]]})")

    return M_fwd, clean_scores


def main():
    print(f"Loading model from: {MODEL_CHECKPOINT_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_PATH)
    model = AutoModel.from_pretrained(MODEL_CHECKPOINT_PATH).to(DEVICE)
    model.eval()

    for metric_name in METRIC_FUNCS:
        print(f"\n{'='*70}\nMETRIC: {metric_name}\n{'='*70}")
        M_fwd, clean_scores = build_forward_matrix(
            model, tokenizer, EXAMPLE_SENTENCES, metric_name
        )

        result = {
            "task": TASK_NAME,
            "direction": "forward",
            "granularity": "layer",
            "matrix": M_fwd.tolist(),
            "metric": metric_name,
            "target_rank_used": TARGET_RANK,
        }

        out_path = f"M_fwd_{TASK_NAME}_{metric_name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved -> {out_path}")

        # Quick readout: which row (source layer) did the most average damage?
        row_avgs = M_fwd.mean(axis=1)
        top_layer = int(np.argmax(row_avgs))
        print(f"Highest average-damage source layer for {metric_name}: "
              f"Layer {top_layer} (avg damage = {row_avgs[top_layer]:.4f})")


if __name__ == "__main__":
    main()