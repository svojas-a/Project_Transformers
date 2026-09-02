"""
M4 - Step 2: Forward Intervention (Layer-Level) -> M_fwd
===========================================================

WHAT THIS DOES
---------------
For each source layer i in {0..5}:
    1. Run a batch of REAL SST-2 sentences through DistilBERT CLEAN (no
       intervention). Record effective_rank / stable_rank / mean_cosine_sim
       at every layer's output.
    2. Run the SAME batch again, but this time force layer i's output to
       collapse to a low rank (SVD truncation) via a forward hook.
       Record the same three metrics at every layer's output.
    3. For every downstream layer j >= i, compute a DAMAGE score where
       POSITIVE ALWAYS MEANS "more collapse", regardless of the metric's
       raw direction (see compute_damage() below).
    4. Row averages are computed ONLY over the valid downstream cells
       (j >= i) - NOT across all 6 columns. Averaging in the zero-padded
       j < i cells would unfairly shrink the average for layers deeper in
       the stack (they simply have fewer valid cells to average), making
       early layers look artificially more damaging than they really are.
    5. Row averages get a bootstrap 95% CI computed by resampling probe
       SENTENCES (not individual matrix cells) with replacement. A small
       numeric gap between two row averages (e.g. 12.09 vs 12.12) is not
       by itself evidence of a real ordering between two source layers -
       only a CI check across sentences can tell you that.

Run this once per task (SST-2 / MNLI / CoNLL-2003) by pointing MODEL_CHECKPOINT_PATH
at that task's fine-tuned model.

RANK IS RELATIVE, NOT ABSOLUTE
-------------------------------
SVD rank of a per-sentence activation matrix [seq_len, hidden_dim] is capped at
min(seq_len, hidden_dim). Short sentences (SST-2 is often ~10-15 tokens) have
seq_len well under any fixed absolute rank like 32 - so "collapsing to rank 32"
on a 12-token sentence truncates nothing and silently no-ops.

RANK_FRACTION fixes this by expressing the collapse target as a fraction of
each sentence's own min(seq_len, hidden_dim), so the intervention scales
down for short sequences instead of vanishing. This also keeps this script
consistent with the dimension-sweep finding that relative compression
(rank / hidden_dim) is the generalizable signal, not raw rank.

WHY COLLAPSER MUST BE REGISTERED BEFORE RECORDER
---------------------------------------------------
PyTorch calls multiple forward hooks on the same module in REGISTRATION
ORDER, and each hook after the first receives whatever the PREVIOUS hook
returned as `output` - not the module's original output. If the recorder
were registered first, it would snapshot the PRE-collapse activation at
the source layer itself (stale), even though the collapse hook still
correctly propagates the collapsed value downstream to later layers. This
would silently corrupt the diagonal cell of M_fwd (the source layer's own
self-damage) while leaving downstream cells correct. Registering the
collapser first ensures the recorder always captures the post-intervention
value, everywhere.

OUTPUT
------
A JSON file per (task, metric) matching the schema agreed in the base
reference doc, with damage values already sign-corrected so that positive
always means "more collapse" for every metric. Also includes per-sentence
damage arrays (not just the averaged matrix) and a bootstrap 95% CI per
row average, so downstream analysis or a write-up can quote significance
without rerunning the forward passes:
    {
        "task": ...,
        "direction": "forward",
        "granularity": "layer",
        "matrix": [[...6x6...]],
        "metric": ...,
        "rank_fraction_used": ...,
        "min_rank_used": ...,
        "row_avg_damage": [...6 values...],
        "row_avg_bootstrap_ci95": [[lo, hi], ...6 pairs...],
        "per_sentence_row_avg_damage": [[...per-sentence values...], ...6 rows...],
    }
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_dataset

# -----------------------------------------------------------------------
# CONFIG - edit these per run
# -----------------------------------------------------------------------
MODEL_CHECKPOINT_PATH = "/content/drive/MyDrive/Project_Transformers/checkpoints/sst2_finetuned"
TASK_NAME = "sst2"                                  # "sst2" | "mnli" | "conll2003"
IS_CLASSIFICATION_CHECKPOINT = True                 # True for fine-tuned checkpoints saved via
                                                     # AutoModelForSequenceClassification (e.g. finetune_sst2.py output)
RANK_FRACTION = 0.3                                 # collapse target = rank_fraction * min(seq_len, hidden_dim)
MIN_RANK = 1                                         # never collapse below this, even on very short sentences
NUM_LAYERS = 6                                       # DistilBERT has 6 transformer blocks
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOTSTRAP = 2000                                   # resamples for the row-average CI
BOOTSTRAP_SEED = 123

# -----------------------------------------------------------------------
# PROBE SENTENCES - real batch pulled from SST-2 validation split
# -----------------------------------------------------------------------
N_PROBE_SENTENCES = 40      # how many sentences to probe with
PROBE_SPLIT = "validation"  # use validation, not train (never touched during fine-tuning)
PROBE_SEED = 42              # fixed seed -> same probe batch every run, for reproducibility
MIN_SEQ_LEN = 4               # drop pathologically short examples that leave
                              # almost nothing for SVD to work with


def load_probe_sentences(tokenizer, n_sentences, split=PROBE_SPLIT, seed=PROBE_SEED,
                          min_seq_len=MIN_SEQ_LEN):
    """
    Pulls a random, reproducible batch of real SST-2 sentences spanning a
    range of lengths, instead of a handful of hardcoded (and coincidentally
    same-length) example sentences.

    Length variation matters here: RANK_FRACTION scales the collapse target
    off each sentence's own seq_len, so a probe set with only one seq_len
    can't reveal whether the layer-wise damage pattern is stable across
    sentence lengths or an artifact of that one length.
    """
    print(f"Loading {n_sentences} probe sentences from SST-2 [{split}] (seed={seed})...")
    dataset = load_dataset("nyu-mll/glue", "sst2", split=split)
    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(len(dataset))

    sentences, seq_lens = [], []
    for idx in shuffled_idx:
        sent = dataset[int(idx)]["sentence"].strip()
        n_tok = len(tokenizer(sent)["input_ids"])
        if n_tok < min_seq_len:
            continue
        sentences.append(sent)
        seq_lens.append(n_tok)
        if len(sentences) >= n_sentences:
            break

    seq_lens = np.array(seq_lens)
    print(f"      Got {len(sentences)} sentences. "
          f"seq_len: min={seq_lens.min()}, max={seq_lens.max()}, "
          f"mean={seq_lens.mean():.1f}, median={int(np.median(seq_lens))}")
    return sentences


# -----------------------------------------------------------------------
# GEOMETRIC METRICS (same three metrics used throughout M3 / the baseline)
# -----------------------------------------------------------------------
def effective_rank(H: torch.Tensor) -> float:
    """
    H: [seq_len, hidden_dim] token matrix for ONE sentence.
    Effective rank = exp(entropy of normalized singular values).
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

# Direction each metric moves in when collapse gets WORSE.
# effective_rank / stable_rank: collapse makes these go DOWN (less diversity).
# mean_cosine_sim: collapse makes this go UP (tokens become more similar).
METRIC_DIRECTION = {
    "effective_rank": "lower_is_worse",
    "stable_rank": "lower_is_worse",
    "mean_cosine_sim": "higher_is_worse",
}


def compute_damage(clean_val, intervened_val, metric_name):
    """
    Returns a damage score where POSITIVE ALWAYS MEANS "more collapse",
    regardless of which raw direction the underlying metric moves in.
    Works elementwise on numpy arrays as well as on plain floats, since
    the bootstrap needs it applied across whole per-sentence arrays.
    """
    direction = METRIC_DIRECTION[metric_name]
    if direction == "lower_is_worse":
        return clean_val - intervened_val
    elif direction == "higher_is_worse":
        return intervened_val - clean_val
    else:
        raise ValueError(f"Unknown direction for metric {metric_name}")


def get_transformer_layers(model):
    """
    Locates the list of 6 DistilBERT transformer blocks regardless of
    whether `model` is a plain AutoModel (layers at model.transformer.layer)
    or a classification checkpoint like AutoModelForSequenceClassification
    (layers nested one level deeper, at model.distilbert.transformer.layer).
    """
    if hasattr(model, "distilbert"):
        return model.distilbert.transformer.layer
    elif hasattr(model, "transformer"):
        return model.transformer.layer
    else:
        raise AttributeError(
            "Could not find transformer layers on this model. "
            "Expected model.distilbert.transformer.layer or model.transformer.layer - "
            "print(model) to inspect the structure and adjust get_transformer_layers()."
        )


# -----------------------------------------------------------------------
# SVD LOW-RANK COLLAPSE (the actual "damage" we inject)
# -----------------------------------------------------------------------
def collapse_to_rank(H: torch.Tensor, rank_fraction: float, min_rank: int = 1) -> torch.Tensor:
    """
    H: [batch, seq_len, hidden_dim]
    Truncates each sentence's token matrix to a rank that is a FRACTION of
    that sentence's own max possible rank (min(seq_len, hidden_dim)), then
    reconstructs via SVD. Scales correctly regardless of sentence length -
    unlike a fixed absolute rank, which silently no-ops once seq_len drops
    below the target.
    """
    orig_dtype = H.dtype
    orig_device = H.device
    out = torch.empty_like(H)
    for b in range(H.shape[0]):
        mat = H[b].detach().to(torch.float32)
        U, S, Vt = torch.linalg.svd(mat, full_matrices=False)
        max_rank = S.shape[0]  # = min(seq_len, hidden_dim)
        k = max(min_rank, int(max_rank * rank_fraction))
        k = min(k, max_rank)
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
        for i, layer in enumerate(get_transformer_layers(model)):
            handle = layer.register_forward_hook(self._make_hook(i))
            self.handles.append(handle)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
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
    def __init__(self, model, source_layer, rank_fraction, min_rank=1):
        self.handle = None
        self.source_layer = source_layer
        self.rank_fraction = rank_fraction
        self.min_rank = min_rank
        layer = get_transformer_layers(model)[source_layer]
        self.handle = layer.register_forward_hook(self._collapse_hook)

    def _collapse_hook(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            collapsed = collapse_to_rank(hidden, self.rank_fraction, self.min_rank)
            return (collapsed,) + output[1:]
        else:
            return collapse_to_rank(output, self.rank_fraction, self.min_rank)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()


# -----------------------------------------------------------------------
# SANITY CHECK - confirms the intervention actually perturbs the source
# layer's own output before we bother measuring anything downstream.
# -----------------------------------------------------------------------
def sanity_check_intervention(model, tokenizer, sentences, source_layer, rank_fraction, min_rank,
                               n_check=3):
    """
    Spot-checks a handful of sentences (not just sentences[0]) to confirm the
    intervention is actually perturbing the source layer's own output before
    trusting anything measured downstream.
    """
    check_idxs = list(range(min(n_check, len(sentences))))
    for i in check_idxs:
        sentence = sentences[i]
        inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
        seq_len = inputs["input_ids"].shape[1]

        recorder = LayerRecorder(model)
        with torch.no_grad():
            model(**inputs)
        clean_hidden = recorder.activations[source_layer][0].clone()
        recorder.remove()

        # Collapser registered BEFORE recorder - see module docstring for why
        # this ordering is required for the recorder to capture the
        # POST-collapse value at the source layer itself.
        collapser = LayerCollapser(model, source_layer, rank_fraction, min_rank)
        recorder = LayerRecorder(model)
        with torch.no_grad():
            model(**inputs)
        collapsed_hidden = recorder.activations[source_layer][0].clone()
        collapser.remove()
        recorder.remove()

        max_rank = min(seq_len, clean_hidden.shape[-1])
        k_used = min(max(min_rank, int(max_rank * rank_fraction)), max_rank)
        diff = (collapsed_hidden - clean_hidden).abs().max().item()

        print(f"      [sanity {i}] seq_len={seq_len}, max_rank={max_rank}, "
              f"k_used={k_used}, max abs diff at source layer={diff:.6f}")
        if diff < 1e-6:
            print(f"      [sanity {i}] WARNING: intervention produced ~zero change at the "
                  "source layer itself - check RANK_FRACTION / MIN_RANK.")


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
            per_sentence_acts[layer_idx].append(act[0])

    recorder.remove()
    return per_sentence_acts


def run_intervened_pass(model, tokenizer, sentences, source_layer, rank_fraction, min_rank):
    """Same as run_clean_pass but with layer `source_layer` forced to collapse."""
    # Collapser must be registered BEFORE the recorder - see module docstring.
    collapser = LayerCollapser(model, source_layer, rank_fraction, min_rank)
    recorder = LayerRecorder(model)
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


def compute_metric_per_sentence(per_sentence_acts, metric_fn):
    """
    Returns dict: layer_idx -> np.array of per-sentence metric values
    (NOT averaged). The mean of this array is what the old
    compute_metric_per_layer returned; keeping the raw per-sentence values
    is what makes the bootstrap CI possible.
    """
    layer_scores = {}
    for layer_idx, acts_list in per_sentence_acts.items():
        scores = np.array([metric_fn(act) for act in acts_list], dtype=np.float64)
        layer_scores[layer_idx] = scores
    return layer_scores


def bootstrap_row_ci(per_sentence_damage_row, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
                      ci=0.95):
    """
    per_sentence_damage_row: np.array [n_sentences] - the per-sentence row
    average damage for ONE source layer (already averaged across the valid
    j >= source_layer columns for that sentence, see build_forward_matrix).

    Resamples SENTENCES with replacement (not matrix cells - the sentences
    are the actual independent unit here) and recomputes the row mean each
    time, to get a 95% CI on the row average. Returns (lo, hi).
    """
    rng = np.random.default_rng(seed)
    n = len(per_sentence_damage_row)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        boot_means[b] = per_sentence_damage_row[sample_idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(lo), float(hi)


def build_forward_matrix(model, tokenizer, sentences, metric_name):
    metric_fn = METRIC_FUNCS[metric_name]

    print(f"\n[1/2] Running CLEAN baseline pass ({metric_name})...")
    clean_acts = run_clean_pass(model, tokenizer, sentences)
    clean_per_sentence = compute_metric_per_sentence(clean_acts, metric_fn)
    clean_scores = {i: float(v.mean()) for i, v in clean_per_sentence.items()}
    print(f"      Clean per-layer {metric_name}: {clean_scores}")

    M_fwd = np.zeros((NUM_LAYERS, NUM_LAYERS))
    # per_sentence_row_avg[source_layer] = np.array of shape [n_sentences],
    # the per-sentence average damage across valid columns j >= source_layer.
    # This is the array the bootstrap resamples.
    per_sentence_row_avg = [None] * NUM_LAYERS
    row_ci95 = [None] * NUM_LAYERS

    for source_layer in range(NUM_LAYERS):
        print(f"\n[2/2] Intervening at layer {source_layer} "
              f"(collapsing to rank_fraction={RANK_FRACTION})...")

        sanity_check_intervention(
            model, tokenizer, sentences, source_layer, RANK_FRACTION, MIN_RANK
        )

        intervened_acts = run_intervened_pass(
            model, tokenizer, sentences, source_layer, RANK_FRACTION, MIN_RANK
        )
        intervened_per_sentence = compute_metric_per_sentence(intervened_acts, metric_fn)
        intervened_scores = {i: float(v.mean()) for i, v in intervened_per_sentence.items()}

        # Per-sentence damage matrix for this source layer's valid columns
        # (j >= source_layer): shape [n_sentences, n_valid_cols].
        valid_js = list(range(source_layer, NUM_LAYERS))
        per_sentence_damage_cols = np.stack([
            compute_damage(clean_per_sentence[j], intervened_per_sentence[j], metric_name)
            for j in valid_js
        ], axis=1)  # [n_sentences, n_valid_cols]

        for col_idx, j in enumerate(valid_js):
            M_fwd[source_layer][j] = per_sentence_damage_cols[:, col_idx].mean()
        # j < source_layer stays 0.0 from np.zeros() - untouched, no damage possible.

        # Row average PER SENTENCE (mean across the valid columns, for each
        # sentence individually) is the unit the bootstrap resamples.
        per_sentence_row_avg[source_layer] = per_sentence_damage_cols.mean(axis=1)
        row_ci95[source_layer] = bootstrap_row_ci(per_sentence_row_avg[source_layer])

        row_avg = per_sentence_row_avg[source_layer].mean()
        lo, hi = row_ci95[source_layer]
        print(f"      Row {source_layer} avg damage (valid cells only): {row_avg:.4f} "
              f"[95% CI: {lo:.4f}, {hi:.4f}]  "
              f"(per-layer: {[f'{v:.3f}' for v in M_fwd[source_layer]]})")

    return M_fwd, clean_scores, per_sentence_row_avg, row_ci95


def compare_top_layers(row_avgs, row_ci95, metric_name):
    """
    Flags whether the top-1 and top-2 source layers by row average damage
    are actually distinguishable, or just a close call within noise. Two
    layers are called "not distinguishable" if their 95% CIs overlap.
    """
    order = np.argsort(row_avgs)[::-1]  # descending
    top1, top2 = int(order[0]), int(order[1])
    lo1, hi1 = row_ci95[top1]
    lo2, hi2 = row_ci95[top2]
    overlap = not (hi2 < lo1 or hi1 < lo2)  # CIs overlap if neither is strictly above the other
    gap_pct = 100 * (row_avgs[top1] - row_avgs[top2]) / max(abs(row_avgs[top1]), 1e-12)

    print(f"\n[significance check - {metric_name}] "
          f"Top layer: {top1} ({row_avgs[top1]:.4f}, CI [{lo1:.4f},{hi1:.4f}])  "
          f"Runner-up: {top2} ({row_avgs[top2]:.4f}, CI [{lo2:.4f},{hi2:.4f}])  "
          f"gap={gap_pct:.1f}%")
    if overlap:
        print(f"[significance check - {metric_name}] CIs OVERLAP - "
              f"Layer {top1} being 'the top layer' is NOT statistically distinguishable "
              f"from Layer {top2} on this probe batch. Treat the ranking as a tie, not a finding.")
    else:
        print(f"[significance check - {metric_name}] CIs do not overlap - "
              f"Layer {top1} is distinguishably higher-damage than Layer {top2}.")
    return top1, top2, overlap


def main():
    print(f"Loading model from: {MODEL_CHECKPOINT_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_PATH)
    if IS_CLASSIFICATION_CHECKPOINT:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT_PATH).to(DEVICE)
    else:
        model = AutoModel.from_pretrained(MODEL_CHECKPOINT_PATH).to(DEVICE)
    model.eval()
    print(f"Layers found: {len(get_transformer_layers(model))} "
          f"(should be 6 for DistilBERT)")

    probe_sentences = load_probe_sentences(tokenizer, N_PROBE_SENTENCES)

    for metric_name in METRIC_FUNCS:
        print(f"\n{'='*70}\nMETRIC: {metric_name}\n{'='*70}")
        M_fwd, clean_scores, per_sentence_row_avg, row_ci95 = build_forward_matrix(
            model, tokenizer, probe_sentences, metric_name
        )

        row_avgs = np.array([v.mean() for v in per_sentence_row_avg])

        result = {
            "task": TASK_NAME,
            "direction": "forward",
            "granularity": "layer",
            "matrix": M_fwd.tolist(),
            "metric": metric_name,
            "rank_fraction_used": RANK_FRACTION,
            "min_rank_used": MIN_RANK,
            "row_avg_damage": row_avgs.tolist(),
            "row_avg_bootstrap_ci95": [list(ci) for ci in row_ci95],
            "per_sentence_row_avg_damage": [v.tolist() for v in per_sentence_row_avg],
        }

        out_path = f"M_fwd_{TASK_NAME}_{metric_name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved -> {out_path}")

        top_layer = int(np.argmax(row_avgs))
        print(f"Highest average-damage source layer for {metric_name}: "
              f"Layer {top_layer} (avg damage = {row_avgs[top_layer]:.4f})")
        print(f"All row averages: {[f'{v:.3f}' for v in row_avgs]}")

        compare_top_layers(row_avgs, row_ci95, metric_name)


if __name__ == "__main__":
    main()