"""
Shared SST-2 intervention utilities.

This module is the single source of truth for:
- model/checkpoint loading
- probe sentence selection
- representation metrics
- damage/recovery sign convention
- SVD low-rank collapse
- clean/intervened/restored passes
- bootstrap confidence intervals

Run forward and backward experiments on the SAME probe set and checkpoint.
"""

from pathlib import Path
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
TASK_NAME = "sst2"
MODEL_CHECKPOINT_PATH = Path(
    "/content/drive/MyDrive/Project_Transformers/checkpoints/sst2_finetuned"
)

N_LAYERS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use one intervention strength for the first SST-2 pipeline.
# Later, sweep this after the pipeline is validated.
RANK_FRACTION = 0.30
MIN_RANK = 1

N_PROBE_SENTENCES = 40
PROBE_SPLIT = "validation"
PROBE_SEED = 42
MIN_SEQ_LEN = 4

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 123

# ---------------------------------------------------------------------
# MODEL / PROBE SET
# ---------------------------------------------------------------------
def load_model_and_tokenizer():
    if not Path(MODEL_CHECKPOINT_PATH).exists():
        raise FileNotFoundError(
            f"SST-2 checkpoint not found:\n{MODEL_CHECKPOINT_PATH}\n"
            "Run finetune_sst2.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_CHECKPOINT_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_CHECKPOINT_PATH)
    ).to(DEVICE)
    model.eval()

    layers = get_transformer_layers(model)
    if len(layers) != N_LAYERS:
        raise RuntimeError(
            f"Expected {N_LAYERS} DistilBERT layers, found {len(layers)}."
        )

    return model, tokenizer


def load_probe_sentences(tokenizer, n_sentences=N_PROBE_SENTENCES,
                         split=PROBE_SPLIT, seed=PROBE_SEED,
                         min_seq_len=MIN_SEQ_LEN):
    """
    Deterministically select the same SST-2 validation sentences for both
    forward and backward experiments.
    """
    dataset = load_dataset("nyu-mll/glue", "sst2", split=split)
    rng = np.random.default_rng(seed)

    sentences = []
    example_indices = []

    for idx in rng.permutation(len(dataset)):
        sentence = dataset[int(idx)]["sentence"].strip()
        n_tok = len(tokenizer(sentence)["input_ids"])
        if n_tok < min_seq_len:
            continue

        sentences.append(sentence)
        example_indices.append(int(idx))

        if len(sentences) >= n_sentences:
            break

    if len(sentences) < n_sentences:
        raise RuntimeError(
            f"Could only find {len(sentences)} valid probe sentences, "
            f"requested {n_sentences}."
        )

    seq_lens = np.array(
        [len(tokenizer(s)["input_ids"]) for s in sentences], dtype=int
    )

    print(
        f"Probe set: {len(sentences)} SST-2 validation sentences | "
        f"seed={seed} | seq_len min={seq_lens.min()}, "
        f"max={seq_lens.max()}, mean={seq_lens.mean():.1f}"
    )

    return sentences, example_indices


def save_probe_set(sentences, example_indices, output_path):
    payload = {
        "task": TASK_NAME,
        "split": PROBE_SPLIT,
        "seed": PROBE_SEED,
        "n_sentences": len(sentences),
        "dataset_indices": example_indices,
        "sentences": sentences,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# REPRESENTATION METRICS
# ---------------------------------------------------------------------
def effective_rank(H):
    H = H.detach().float().cpu()
    if H.shape[0] < 2:
        return float("nan")

    s = torch.linalg.svdvals(H)
    s = s[s > 1e-12]
    if s.numel() == 0:
        return float("nan")

    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy).item())


def stable_rank(H):
    H = H.detach().float().cpu()
    if H.shape[0] < 2:
        return float("nan")

    s = torch.linalg.svdvals(H)
    if s.numel() == 0 or s[0] < 1e-12:
        return float("nan")

    return float(((s ** 2).sum() / (s[0] ** 2)).item())


def mean_cosine_sim(H):
    H = H.detach().float().cpu()
    if H.shape[0] < 2:
        return float("nan")

    H_norm = torch.nn.functional.normalize(H, dim=-1)
    sim = H_norm @ H_norm.T
    n = H.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    return float(sim[mask].mean().item())


METRIC_FUNCS = {
    "effective_rank": effective_rank,
    "stable_rank": stable_rank,
    "mean_cosine_sim": mean_cosine_sim,
}

# Positive damage/recovery always means "more collapse was caused/recovered".
METRIC_DIRECTION = {
    "effective_rank": "lower_is_worse",
    "stable_rank": "lower_is_worse",
    "mean_cosine_sim": "higher_is_worse",
}


def compute_damage(clean_val, intervened_val, metric_name):
    direction = METRIC_DIRECTION[metric_name]

    if direction == "lower_is_worse":
        return clean_val - intervened_val

    if direction == "higher_is_worse":
        return intervened_val - clean_val

    raise ValueError(f"Unknown metric direction: {metric_name}")


# ---------------------------------------------------------------------
# DISTILBERT LAYER ACCESS
# ---------------------------------------------------------------------
def get_transformer_layers(model):
    if hasattr(model, "distilbert"):
        return model.distilbert.transformer.layer

    if hasattr(model, "transformer"):
        return model.transformer.layer

    raise AttributeError(
        "Could not locate DistilBERT transformer layers."
    )


# ---------------------------------------------------------------------
# SVD LOW-RANK COLLAPSE
# ---------------------------------------------------------------------
def collapse_to_rank(H, rank_fraction=RANK_FRACTION, min_rank=MIN_RANK):
    """
    H: [batch, seq_len, hidden_dim]

    Each sentence is projected to a rank equal to a fraction of
    min(seq_len, hidden_dim), then reconstructed in the ORIGINAL
    hidden dimension.

    IMPORTANT:
    This is a low-rank intervention, not a literal hidden-dimension change.
    """
    out = torch.empty_like(H)

    for b in range(H.shape[0]):
        mat = H[b].detach().float()
        U, S, Vt = torch.linalg.svd(mat, full_matrices=False)

        max_rank = S.shape[0]
        k = max(min_rank, int(max_rank * rank_fraction))
        k = min(k, max_rank)

        recon = (U[:, :k] * S[:k]) @ Vt[:k, :]
        out[b] = recon.to(H.dtype)

    return out


# ---------------------------------------------------------------------
# HOOKS
# ---------------------------------------------------------------------
class LayerRecorder:
    """Record post-hook output from every transformer block."""

    def __init__(self, model):
        self.activations = {}
        self.handles = []

        for i, layer in enumerate(get_transformer_layers(model)):
            handle = layer.register_forward_hook(self._make_hook(i))
            self.handles.append(handle)

    def _make_hook(self, layer_idx):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden.detach().clone()

        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class LayerCollapser:
    """Collapse exactly one transformer-layer output."""

    def __init__(self, model, source_layer, rank_fraction, min_rank):
        layer = get_transformer_layers(model)[source_layer]
        self.handle = layer.register_forward_hook(self._hook)
        self.rank_fraction = rank_fraction
        self.min_rank = min_rank

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            collapsed = collapse_to_rank(
                hidden, self.rank_fraction, self.min_rank
            )
            return (collapsed,) + output[1:]

        return collapse_to_rank(
            output, self.rank_fraction, self.min_rank
        )

    def remove(self):
        self.handle.remove()


class LayerRestorer:
    """Force one layer's output to the corresponding CLEAN activation."""

    def __init__(self, model, layer_idx, clean_tensor):
        layer = get_transformer_layers(model)[layer_idx]
        self.clean_tensor = clean_tensor
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        patched = self.clean_tensor.unsqueeze(0)

        if isinstance(output, tuple):
            return (patched,) + output[1:]

        return patched

    def remove(self):
        self.handle.remove()


# ---------------------------------------------------------------------
# FORWARD / BACKWARD PASSES
# ---------------------------------------------------------------------
def run_clean_pass(model, tokenizer, sentences):
    per_sentence_acts = {i: [] for i in range(N_LAYERS)}

    for sentence in sentences:
        inputs = tokenizer(
            sentence, return_tensors="pt", truncation=True
        ).to(DEVICE)

        recorder = LayerRecorder(model)

        with torch.inference_mode():
            model(**inputs)

        for layer_idx, act in recorder.activations.items():
            per_sentence_acts[layer_idx].append(act[0])

        recorder.remove()

    return per_sentence_acts


def run_forward_intervention(
    model, tokenizer, sentences, source_layer,
    rank_fraction=RANK_FRACTION, min_rank=MIN_RANK
):
    """
    Collapse source_layer only. All other layers remain untouched.
    """
    per_sentence_acts = {i: [] for i in range(N_LAYERS)}

    collapser = LayerCollapser(
        model, source_layer, rank_fraction, min_rank
    )

    for sentence in sentences:
        inputs = tokenizer(
            sentence, return_tensors="pt", truncation=True
        ).to(DEVICE)

        recorder = LayerRecorder(model)

        with torch.inference_mode():
            model(**inputs)

        for layer_idx, act in recorder.activations.items():
            per_sentence_acts[layer_idx].append(act[0])

        recorder.remove()

    collapser.remove()
    return per_sentence_acts


def run_backward_intervention(
    model, tokenizer, sentences, clean_acts, restore_layer,
    rank_fraction=RANK_FRACTION, min_rank=MIN_RANK
):
    """
    Every layer is collapsed EXCEPT restore_layer, whose output is patched
    to its corresponding clean activation.
    """
    per_sentence_acts = {i: [] for i in range(N_LAYERS)}

    for sent_idx, sentence in enumerate(sentences):
        inputs = tokenizer(
            sentence, return_tensors="pt", truncation=True
        ).to(DEVICE)

        mutators = []

        # Register mutators BEFORE recorder so recorder sees final output.
        for layer_idx in range(N_LAYERS):
            if layer_idx == restore_layer:
                mutators.append(
                    LayerRestorer(
                        model,
                        layer_idx,
                        clean_acts[layer_idx][sent_idx],
                    )
                )
            else:
                mutators.append(
                    LayerCollapser(
                        model,
                        layer_idx,
                        rank_fraction,
                        min_rank,
                    )
                )

        recorder = LayerRecorder(model)

        with torch.inference_mode():
            model(**inputs)

        for layer_idx, act in recorder.activations.items():
            per_sentence_acts[layer_idx].append(act[0])

        recorder.remove()
        for mutator in mutators:
            mutator.remove()

    return per_sentence_acts


def run_fully_collapsed_pass(
    model, tokenizer, sentences,
    rank_fraction=RANK_FRACTION, min_rank=MIN_RANK
):
    """
    Collapse ALL transformer layers.
    """
    per_sentence_acts = {i: [] for i in range(N_LAYERS)}

    for sentence in sentences:
        inputs = tokenizer(
            sentence, return_tensors="pt", truncation=True
        ).to(DEVICE)

        mutators = [
            LayerCollapser(
                model, i, rank_fraction, min_rank
            )
            for i in range(N_LAYERS)
        ]

        recorder = LayerRecorder(model)

        with torch.inference_mode():
            model(**inputs)

        for layer_idx, act in recorder.activations.items():
            per_sentence_acts[layer_idx].append(act[0])

        recorder.remove()
        for mutator in mutators:
            mutator.remove()

    return per_sentence_acts


# ---------------------------------------------------------------------
# METRIC / STATISTICS HELPERS
# ---------------------------------------------------------------------
def compute_metric_per_sentence(per_sentence_acts, metric_name):
    metric_fn = METRIC_FUNCS[metric_name]

    return {
        layer_idx: np.asarray(
            [metric_fn(H) for H in acts],
            dtype=np.float64,
        )
        for layer_idx, acts in per_sentence_acts.items()
    }


def bootstrap_mean_ci(values, n_bootstrap=N_BOOTSTRAP,
                      seed=BOOTSTRAP_SEED, ci=0.95):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return [float("nan"), float("nan")]

    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    boot_means = values[samples].mean(axis=1)

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])

    return [float(lo), float(hi)]


def row_score_from_matrix(matrix, source_layer):
    """
    Downstream-only row score:
        mean over j > source_layer

    Layer 5 has no downstream layers, so its score is NaN.
    """
    if source_layer >= N_LAYERS - 1:
        return float("nan")

    return float(
        np.mean(matrix[source_layer, source_layer + 1:])
    )
