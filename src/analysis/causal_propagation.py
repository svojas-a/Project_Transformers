"""
Layerwise Causal Propagation Analysis
=================================================

Goal
----
the baseline diagnostics stage found a correlational pattern: attention entropy has a V-shaped
minimum at layer 3 (candidate "source" of collapse). This script tests
whether collapse at a given source layer *causally* propagates to
downstream layers, by forcing an artificial rank collapse at layer l
(via SVD low-rank projection on that layer's hidden states) and measuring
how much the four baseline diagnostic metrics shift at layers l+1 ... 5, relative to
a clean (uncompressed, unmodified) run.

This isolates causal propagation from confounds: the model weights are
never changed, only the layer-l activations are intervened on for a single
forward pass, so any downstream metric shift is directly attributable to
that intervention.

Design
------
For each task in {SST-2, MNLI, CoNLL-2003}:
  For each source_layer l in {0, 1, 2, 3, 4, 5}:
    For each target_rank r in {1, 2, 4, 8, 16, 32}:
      1. Run clean forward pass -> record metrics at every layer (baseline)
      2. Attach forward hook at layer l that replaces its output hidden
         states with their rank-r SVD reconstruction
      3. Run intervened forward pass -> record metrics at every layer
      4. delta[l][k][r] = metric(intervened, l+k) - metric(clean, l+k)
         for k = 1 ... (5 - l)

Output
------
results/causal_effects.csv   -- long-format table of all deltas
results/raw_metrics.pkl      -- raw metric dicts (clean + intervened)

Metrics computed per layer (same as the baseline diagnostics stage, for direct comparability):
  - effective_rank        (entropy of normalized singular values)
  - stable_rank            (||A||_F^2 / ||A||_2^2)
  - mean_pairwise_cosine   (mean off-diagonal cosine similarity of tokens)
  - attention_entropy      (mean entropy of attention distributions)

Environment notes carried over from the baseline diagnostics stage debugging:
  - Load model with attn_implementation="eager" or attention entropy will
    be empty (fused SDPA backend doesn't materialize attention tensors).
  - Use "nyu-mll/glue" for SST-2 / MNLI (plain "glue" raises HfUriError).
  - Use "BramVanroy/conll2003" (original loading script is deprecated).
"""

import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DistilBertModel

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_LAYERS = 6
SOURCE_LAYERS = [0, 1, 2, 3, 4, 5]
TARGET_RANKS = [1, 2, 4, 8, 16, 32]
N_SAMPLES_PER_TASK = 64  # keep modest; this is O(layers x ranks x tasks) forward passes
MAX_LENGTH = 64
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

TASKS = {
    "sst2": dict(
        hf_name="nyu-mll/glue",
        hf_config="sst2",
        text_cols=["sentence"],
        split="validation",
    ),
    # MNLI has no single "validation" split -- "matched" = same domain as
    # training, "mismatched" = different domain. We use matched here since
    # it's the standard dev-set choice; switch to validation_mismatched if
    # you specifically want the cross-domain variant.
    "mnli": dict(
        hf_name="nyu-mll/glue",
        hf_config="mnli",
        text_cols=["premise", "hypothesis"],
        split="validation_matched",
    ),
    "conll2003": dict(
        hf_name="BramVanroy/conll2003",
        hf_config=None,
        text_cols=["tokens"],
        split="validation",
    ),
}


# --------------------------------------------------------------------------
# Metric functions (mirroring the baseline diagnostics stage so results are directly comparable)
# --------------------------------------------------------------------------


def effective_rank(hidden: torch.Tensor) -> float:
    """Entropy-based effective rank of the token representation matrix.
    hidden: (seq_len, dim) for a single example, or (batch*seq, dim) pooled.
    """
    h = hidden.detach().float().cpu().numpy()
    if h.shape[0] < 2:
        return float("nan")
    s = np.linalg.svd(h, compute_uv=False)
    s = s[s > 1e-12]
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def stable_rank(hidden: torch.Tensor) -> float:
    h = hidden.detach().float().cpu().numpy()
    if h.shape[0] < 2:
        return float("nan")
    s = np.linalg.svd(h, compute_uv=False)
    if s.max() < 1e-12:
        return float("nan")
    return float(np.sum(s**2) / (s.max() ** 2))


def mean_pairwise_cosine(hidden: torch.Tensor) -> float:
    h = torch.nn.functional.normalize(hidden.detach().float(), dim=-1)
    sim = h @ h.T
    n = sim.shape[0]
    if n < 2:
        return float("nan")
    off_diag_sum = sim.sum() - torch.diagonal(sim).sum()
    return float(off_diag_sum / (n * (n - 1)))


def attention_entropy(attn: torch.Tensor) -> float:
    """attn: (num_heads, seq_len, seq_len) attention probabilities for one example."""
    a = attn.detach().float().cpu().numpy()
    a = np.clip(a, 1e-12, 1.0)
    ent = -np.sum(a * np.log(a), axis=-1)  # entropy per query position, per head
    return float(ent.mean())


def compute_all_metrics(hidden_states: torch.Tensor, attentions: torch.Tensor) -> dict:
    """hidden_states: (seq_len, dim); attentions: (num_heads, seq_len, seq_len)."""
    return {
        "effective_rank": effective_rank(hidden_states),
        "stable_rank": stable_rank(hidden_states),
        "mean_pairwise_cosine": mean_pairwise_cosine(hidden_states),
        "attention_entropy": attention_entropy(attentions),
    }


# --------------------------------------------------------------------------
# Intervention: forced low-rank collapse via SVD truncation
# --------------------------------------------------------------------------


class LowRankCollapseHook:
    """Forward hook that replaces a transformer layer's output hidden states
    with their rank-r SVD reconstruction, per example in the batch.
    Registered on `model.transformer.layer[l]`.
    """

    def __init__(self, target_rank: int):
        self.target_rank = target_rank
        self.active = False

    def __call__(self, module, inputs, output):
        if not self.active:
            return output

        # TransformerBlock output is either a bare tensor or a tuple that may
        # include attention weights (4D: batch, heads, seq, seq) alongside the
        # hidden state (3D: batch, seq, dim). Select by ndim rather than
        # position, since ordering has changed across transformers versions.
        if isinstance(output, tuple):
            hidden_idx = next(i for i, t in enumerate(output) if t.dim() == 3)
            hidden = output[hidden_idx]
        else:
            hidden_idx = None
            hidden = output

        collapsed = hidden.clone()
        for b in range(hidden.shape[0]):
            h = hidden[b].detach().float()  # (seq_len, dim)
            r = min(self.target_rank, h.shape[0], h.shape[1])
            U, S, Vt = torch.linalg.svd(h, full_matrices=False)
            recon = (U[:, :r] * S[:r]) @ Vt[:r, :]
            collapsed[b] = recon.to(hidden.dtype)

        if isinstance(output, tuple):
            out_list = list(output)
            out_list[hidden_idx] = collapsed
            return tuple(out_list)
        return collapsed


# --------------------------------------------------------------------------
# Data loading (small fixed sample per task, consistent with the baseline diagnostics stage sizes)
# --------------------------------------------------------------------------


def load_task_texts(task_name: str, n_samples: int) -> list:
    cfg = TASKS[task_name]
    if cfg["hf_config"]:
        ds = load_dataset(cfg["hf_name"], cfg["hf_config"], split=cfg["split"])
    else:
        ds = load_dataset(cfg["hf_name"], split=cfg["split"])
    ds = ds.select(range(min(n_samples, len(ds))))

    texts = []
    for row in ds:
        if task_name == "conll2003":
            texts.append(" ".join(row["tokens"]))
        elif task_name == "mnli":
            texts.append(row["premise"] + " " + row["hypothesis"])
        else:
            texts.append(row["sentence"])
    return texts


# --------------------------------------------------------------------------
# Core experiment: one (task, source_layer, rank) cell
# --------------------------------------------------------------------------


def run_forward_collect_metrics(
    model,
    tokenizer,
    text: str,
    hook: LowRankCollapseHook = None,
    collapse_active: bool = False,
) -> dict:
    """Runs a single forward pass, returns {layer_idx: metrics_dict} for layers 0..5
    (layer_idx here means AFTER transformer layer idx, i.e. its output)."""
    if hook is not None:
        hook.active = collapse_active

    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
    ).to(DEVICE)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, output_attentions=True)

    # out.hidden_states: tuple of (embedding, layer1_out, ..., layer6_out) -> len 7
    # out.attentions: tuple of length 6, each (batch, heads, seq, seq)
    per_layer = {}
    for layer in range(N_LAYERS):
        hidden = out.hidden_states[layer + 1][0]  # (seq_len, dim), skip batch dim
        attn = out.attentions[layer][0]  # (heads, seq_len, seq_len)
        per_layer[layer] = compute_all_metrics(hidden, attn)

    if hook is not None:
        hook.active = False
    return per_layer


def run_experiment():
    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertModel.from_pretrained(MODEL_NAME, attn_implementation="eager").to(
        DEVICE
    )
    model.eval()

    records = []
    raw_store = {}

    for task_name in TASKS:
        print(f"\n=== Task: {task_name} ===")
        texts = load_task_texts(task_name, N_SAMPLES_PER_TASK)

        # ---- clean baseline metrics (no hook attached at all) ----
        clean_metrics_per_example = []
        for text in texts:
            clean_metrics_per_example.append(
                run_forward_collect_metrics(model, tokenizer, text, hook=None)
            )
        # average clean metrics across examples, per layer
        clean_avg = {
            layer: {
                m: float(np.nanmean([ex[layer][m] for ex in clean_metrics_per_example]))
                for m in [
                    "effective_rank",
                    "stable_rank",
                    "mean_pairwise_cosine",
                    "attention_entropy",
                ]
            }
            for layer in range(N_LAYERS)
        }
        raw_store[(task_name, "clean")] = clean_avg

        # ---- intervened runs: for each source layer, each rank ----
        for source_layer, rank in itertools.product(SOURCE_LAYERS, TARGET_RANKS):
            hook_fn = LowRankCollapseHook(target_rank=rank)
            handle = model.transformer.layer[source_layer].register_forward_hook(
                hook_fn
            )

            intervened_metrics_per_example = []
            for text in texts:
                intervened_metrics_per_example.append(
                    run_forward_collect_metrics(
                        model, tokenizer, text, hook=hook_fn, collapse_active=True
                    )
                )
            handle.remove()

            intervened_avg = {
                layer: {
                    m: float(
                        np.nanmean([ex[layer][m] for ex in intervened_metrics_per_example])
                    )
                    for m in [
                        "effective_rank",
                        "stable_rank",
                        "mean_pairwise_cosine",
                        "attention_entropy",
                    ]
                }
                for layer in range(N_LAYERS)
            }
            raw_store[(task_name, source_layer, rank)] = intervened_avg

            # record deltas for every downstream layer (l+1 ... 5)
            for downstream_layer in range(source_layer + 1, N_LAYERS):
                distance = downstream_layer - source_layer
                for metric_name in [
                    "effective_rank",
                    "stable_rank",
                    "mean_pairwise_cosine",
                    "attention_entropy",
                ]:
                    clean_val = clean_avg[downstream_layer][metric_name]
                    intervened_val = intervened_avg[downstream_layer][metric_name]
                    records.append(
                        {
                            "task": task_name,
                            "source_layer": source_layer,
                            "target_rank": rank,
                            "downstream_layer": downstream_layer,
                            "distance": distance,
                            "metric": metric_name,
                            "clean_value": clean_val,
                            "intervened_value": intervened_val,
                            "delta": intervened_val - clean_val,
                            "abs_delta": abs(intervened_val - clean_val),
                        }
                    )
            print(f"  source_layer={source_layer}, rank={rank} done")

    df = pd.DataFrame.from_records(records)
    df.to_csv(OUTPUT_DIR / "causal_effects.csv", index=False)
    with open(OUTPUT_DIR / "raw_metrics.pkl", "wb") as f:
        pickle.dump(raw_store, f)

    print(f"\nSaved {len(df)} rows to {OUTPUT_DIR / 'causal_effects.csv'}")
    return df


if __name__ == "__main__":
    run_experiment()
