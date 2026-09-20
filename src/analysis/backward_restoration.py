"""
M4 - Step 3: Backward Restoration (layer-level) -> M_bwd
=========================================================
Supports all three tasks from the M4 doc: SST-2, MNLI and CoNLL-2003.

This file is SELF-CONTAINED: it does not import anything from Person B's
forward_causal_intervention.py. The metric definitions, the damage sign
convention, the SVD collapse, the hook classes and the bootstrap CI below are
copied from B's script one-to-one, so M_fwd and M_bwd are still computed under
the same definitions. The shared contract is the output JSON schema (doc
Sec. 4), not a shared import.

    python -m src.analysis.backward_restoration --tasks sst2,mnli,conll2003

WHAT THIS DOES
---------------
For each task, and for each candidate "healing" layer i in {0..5}:
    1. CLEAN pass: run the probe batch through the fine-tuned model untouched
       and record every layer's output.
    2. FULLY COLLAPSED baseline: every layer's output is rank-collapsed with
       the SVD truncation (RANK_FRACTION). Nothing is restored.
    3. RESTORED pass: every layer EXCEPT i is still collapsed; layer i's
       output is force-replaced by its CLEAN activation (ROME-style
       single-point restoration). Everything downstream runs on the healed
       activation.
    4. recovery[i][j] (j >= i) = damage(clean, fully_collapsed)[j]
                                - damage(clean, restored_at_i)[j]
       Positive = restoring layer i undid collapse at layer j.

WHAT CHANGED VS THE PREVIOUS VERSION
--------------------------------------
* All 3 tasks: task registry (SST-2 single sentence, MNLI premise/hypothesis
  pair, CoNLL-2003 word tokens with a token-classification head), per-task
  probe loading, and the right AutoModel class per task.
* Forward passes run once per task, and all three metrics are computed on the
  same stored activations (before, every metric re-ran every pass).
* Doc Step 4 additions: recovery at the FINAL layer ("highest downstream
  recovery across the final layers"), the fixed-distance check
  M_bwd[i][i+1], and a propagation slope per row.
* Sanity check now also reports task performance (clean vs fully collapsed
  vs fully restored). Full restoration matches clean by construction, so on
  its own that check only proves the wiring; the collapsed-vs-restored gap
  shows the collapse actually hurts and restoration actually helps.
* Output JSON now includes target_rank_used (doc schema), records which model
  produced the numbers, and writes null instead of NaN (valid JSON).
* The M_fwd cross-check reads the M_fwd matrix and scores it the same way as
  M_bwd (downstream cells only, diagonal excluded), and is optional.

RANK IS RELATIVE, NOT ABSOLUTE
-------------------------------
The doc says "start from d = 32". Like B, this collapses each example's
activation to RANK_FRACTION * min(seq_len, hidden_dim), because a fixed rank
like 32 silently no-ops on short sequences. target_rank_used in the output is
the median rank actually used across the probe examples.

WHY MUTATORS ARE REGISTERED BEFORE THE RECORDER
------------------------------------------------
PyTorch runs forward hooks on a module in registration order, and each hook
receives what the previous one returned. If the recorder ran first it would
snapshot the pre-patch value at the patched layer.

OUTPUT
------
One JSON file per (task, metric): M_bwd_{task}_{metric}.json. Required doc
fields: task, direction, granularity, matrix, metric, target_rank_used. The
lower triangle (j < i) of the matrix is structurally 0.0 (not reachable by the
patch), same as M_fwd.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# -----------------------------------------------------------------------
# CONFIG - defaults match Person B's script so M_fwd and M_bwd are comparable
# -----------------------------------------------------------------------
NUM_LAYERS = 6  # DistilBERT has 6 transformer blocks
RANK_FRACTION = 0.3  # collapse target = fraction * min(seq_len, hidden_dim)
MIN_RANK = 1
N_PROBE_EXAMPLES = 40
PROBE_SEED = 42
MIN_SEQ_LEN = 4  # drop pathologically short examples (in tokens)
MAX_SEQ_LEN = 128
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 123
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_NAME = "distilbert-base-uncased"
SANITY_TOL = 1e-5


# -----------------------------------------------------------------------
# TASK REGISTRY
# -----------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    name: str
    hf_sources: tuple  # ((dataset_path, config_name), ...) tried in order
    split: str
    input_kind: str  # "single" | "pair" | "tokens"
    text_fields: tuple
    label_field: str


TASKS = {
    "sst2": TaskSpec(
        name="sst2",
        hf_sources=(("nyu-mll/glue", "sst2"),),
        split="validation",
        input_kind="single",
        text_fields=("sentence",),
        label_field="label",
    ),
    "mnli": TaskSpec(
        name="mnli",
        hf_sources=(("nyu-mll/glue", "mnli"),),
        split="validation_matched",
        input_kind="pair",
        text_fields=("premise", "hypothesis"),
        label_field="label",
    ),
    "conll2003": TaskSpec(
        name="conll2003",
        # Script-based datasets stopped loading in newer `datasets` versions,
        # so try the namespaced copy first and the legacy name second.
        hf_sources=(("eriktks/conll2003", None), ("conll2003", None)),
        split="validation",
        input_kind="tokens",
        text_fields=("tokens",),
        label_field="ner_tags",
    ),
}


@dataclass
class ProbeExample:
    text: object  # str (single) | (str, str) (pair) | list[str] (tokens)
    label: object  # int, or list[int] (one per word) for tokens
    seq_len: int  # length in model tokens, incl. [CLS]/[SEP]
    word_ids: Optional[list] = None  # tokens task only: subword -> word index


def tokenize(tokenizer, spec, text, **kwargs):
    kwargs.setdefault("truncation", True)
    kwargs.setdefault("max_length", MAX_SEQ_LEN)
    if spec.input_kind == "pair":
        return tokenizer(text[0], text[1], **kwargs)
    if spec.input_kind == "tokens":
        return tokenizer(text, is_split_into_words=True, **kwargs)
    return tokenizer(text, **kwargs)


def encode(tokenizer, spec, example):
    """Model inputs for one example, on DEVICE. Batch size is always 1."""
    enc = tokenize(tokenizer, spec, example.text, return_tensors="pt")
    # DistilBERT has no segment embeddings; even for MNLI the [SEP] token
    # marks the boundary, so token_type_ids must not be passed.
    enc.pop("token_type_ids", None)
    return {k: v.to(DEVICE) for k, v in enc.items()}


def row_to_example(spec, row):
    """Turns one raw HF row into (text, label), or None if it should be skipped."""
    fields = [row[f] for f in spec.text_fields]
    if spec.input_kind == "single":
        return fields[0].strip(), row[spec.label_field]
    if spec.input_kind == "pair":
        return (fields[0].strip(), fields[1].strip()), row[spec.label_field]
    words = list(fields[0])
    if not words or words == ["-DOCSTART-"]:  # document separator, not a sentence
        return None
    return words, list(row[spec.label_field])


def load_hf_split(spec):
    from datasets import load_dataset

    errors = []
    for path, config in spec.hf_sources:
        try:
            return load_dataset(path,config,split=spec.split,trust_remote_code=True,)
        except Exception as exc:  # noqa: BLE001 - report every failed source
            errors.append(f"  {path}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"Could not load task '{spec.name}'. Tried:\n" + "\n".join(errors)
    )


def load_probe_examples(
    tokenizer,
    spec,
    n_examples,
    seed=PROBE_SEED,
    min_seq_len=MIN_SEQ_LEN,
    dataset=None,
):
    """
    Reproducible probe batch. Same recipe as B's SST-2 loader (seeded shuffle,
    drop examples shorter than min_seq_len tokens, take the first n), so for
    SST-2 this selects the same sentences as B's script.
    """
    print(f"Loading {n_examples} probe examples from {spec.name} (seed={seed})...")
    if dataset is None:
        dataset = load_hf_split(spec)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))

    examples = []
    for idx in order:
        parsed = row_to_example(spec, dataset[int(idx)])
        if parsed is None:
            continue
        text, label = parsed
        enc = tokenize(tokenizer, spec, text)
        seq_len = len(enc["input_ids"])
        if seq_len < min_seq_len:
            continue
        word_ids = enc.word_ids(0) if spec.input_kind == "tokens" else None
        examples.append(ProbeExample(text, label, seq_len, word_ids))
        if len(examples) >= n_examples:
            break

    lens = np.array([e.seq_len for e in examples])
    print(
        f"      Got {len(examples)} examples. seq_len: min={lens.min()}, "
        f"max={lens.max()}, mean={lens.mean():.1f}, median={int(np.median(lens))}"
    )
    return examples


# -----------------------------------------------------------------------
# GEOMETRIC METRICS (identical to forward_causal_intervention.py)
# -----------------------------------------------------------------------
def effective_rank(H: torch.Tensor) -> float:
    """H: [seq_len, hidden]. exp(entropy of normalized singular values)."""
    H = H.detach().cpu().to(torch.float32)
    s = torch.linalg.svdvals(H)
    s = s[s > 1e-12]
    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy).item()


def stable_rank(H: torch.Tensor) -> float:
    """||H||_F^2 / ||H||_2^2."""
    H = H.detach().cpu().to(torch.float32)
    s = torch.linalg.svdvals(H)
    return ((s**2).sum() / s[0] ** 2).item()


def mean_token_cosine_sim(H: torch.Tensor) -> float:
    """Mean pairwise cosine similarity between distinct token vectors."""
    H = H.detach().cpu().to(torch.float32)
    H_norm = torch.nn.functional.normalize(H, dim=-1)
    sim = H_norm @ H_norm.T
    mask = ~torch.eye(H.shape[0], dtype=torch.bool)
    return sim[mask].mean().item()


METRIC_FUNCS = {
    "effective_rank": effective_rank,
    "stable_rank": stable_rank,
    "mean_cosine_sim": mean_token_cosine_sim,
}

METRIC_DIRECTION = {
    "effective_rank": "lower_is_worse",
    "stable_rank": "lower_is_worse",
    "mean_cosine_sim": "higher_is_worse",
}


def compute_damage(clean_val, intervened_val, metric_name):
    """Positive always means 'more collapse', whatever the metric's raw direction."""
    direction = METRIC_DIRECTION[metric_name]
    if direction == "lower_is_worse":
        return clean_val - intervened_val
    if direction == "higher_is_worse":
        return intervened_val - clean_val
    raise ValueError(f"Unknown direction for metric {metric_name}")


# -----------------------------------------------------------------------
# COLLAPSE + HOOKS
# -----------------------------------------------------------------------
def collapse_to_rank(
    H: torch.Tensor, rank_fraction: float, min_rank: int = 1
) -> torch.Tensor:
    """H: [batch, seq_len, hidden]. SVD-truncate each example to a rank that is
    a fraction of its own min(seq_len, hidden)."""
    out = torch.empty_like(H)
    for b in range(H.shape[0]):
        mat = H[b].detach().to(torch.float32)
        U, S, Vt = torch.linalg.svd(mat, full_matrices=False)
        max_rank = S.shape[0]
        k = min(max(min_rank, int(max_rank * rank_fraction)), max_rank)
        S_trunc = S.clone()
        S_trunc[k:] = 0.0
        out[b] = (U @ torch.diag(S_trunc) @ Vt).to(H.dtype)
    return out


def get_transformer_layers(model):
    """Works for AutoModel, ...ForSequenceClassification and ...ForTokenClassification."""
    if hasattr(model, "distilbert"):
        return model.distilbert.transformer.layer
    if hasattr(model, "transformer"):
        return model.transformer.layer
    raise AttributeError("Could not find DistilBERT transformer layers on this model.")


def model_has_head(model) -> bool:
    return hasattr(model, "classifier")


class LayerRecorder:
    """Records every layer's output hidden state (no modification)."""

    def __init__(self, model):
        self.activations = {}
        self.handles = [
            layer.register_forward_hook(self._make_hook(i))
            for i, layer in enumerate(get_transformer_layers(model))
        ]

    def _make_hook(self, layer_idx):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden.detach()

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


class LayerCollapser:
    """Replaces ONE layer's output with a rank-collapsed version."""

    def __init__(self, model, source_layer, rank_fraction, min_rank=1):
        self.rank_fraction = rank_fraction
        self.min_rank = min_rank
        layer = get_transformer_layers(model)[source_layer]
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            collapsed = collapse_to_rank(output[0], self.rank_fraction, self.min_rank)
            return (collapsed,) + output[1:]
        return collapse_to_rank(output, self.rank_fraction, self.min_rank)

    def remove(self):
        self.handle.remove()


class LayerRestorer:
    """
    ROME-style single-point restoration: discards what ONE layer just computed
    (possibly from a corrupted input) and force-replaces it with the
    precomputed CLEAN activation for this exact example.
    """

    def __init__(self, model, layer_idx, clean_tensor):
        self.clean_tensor = clean_tensor  # [seq_len, hidden], on DEVICE
        layer = get_transformer_layers(model)[layer_idx]
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        patched = self.clean_tensor.unsqueeze(0)
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    def remove(self):
        self.handle.remove()


# -----------------------------------------------------------------------
# FORWARD PASSES
# -----------------------------------------------------------------------
@dataclass
class PassResult:
    acts: dict  # layer_idx -> list of [seq_len, hidden] CPU tensors
    logits: Optional[list]  # per-example logits on CPU, None for a headless model


def run_pass(
    model,
    tokenizer,
    spec,
    examples,
    clean_acts=None,
    restore_layers=frozenset(),
    collapse_layers=frozenset(),
    rank_fraction=RANK_FRACTION,
    min_rank=MIN_RANK,
):
    """
    General pass. Each layer is restored to its clean value (restore_layers),
    rank-collapsed (collapse_layers) or left alone.

        nothing / nothing                   -> clean pass
        collapse=all                        -> fully collapsed baseline
        restore={i}, collapse=all - {i}     -> single-point restoration at i
        restore=all                         -> full restoration (sanity check)
    """
    restore_layers = set(restore_layers)
    collapse_layers = set(collapse_layers)
    if restore_layers & collapse_layers:
        raise ValueError("A layer cannot be both restored and collapsed.")
    if restore_layers and clean_acts is None:
        raise ValueError("clean_acts is required when restoring layers.")

    has_head = model_has_head(model)
    acts = {i: [] for i in range(NUM_LAYERS)}
    logits = []

    for idx, example in enumerate(examples):
        inputs = encode(tokenizer, spec, example)

        # Mutators BEFORE the recorder (see module docstring).
        mutators = []
        for layer_idx in range(NUM_LAYERS):
            if layer_idx in restore_layers:
                clean_t = clean_acts[layer_idx][idx].to(DEVICE)
                mutators.append(LayerRestorer(model, layer_idx, clean_t))
            elif layer_idx in collapse_layers:
                mutators.append(
                    LayerCollapser(model, layer_idx, rank_fraction, min_rank)
                )
        recorder = LayerRecorder(model)
        try:
            with torch.no_grad():
                out = model(**inputs)
        finally:
            recorder.remove()
            for m in mutators:
                m.remove()

        for layer_idx, act in recorder.activations.items():
            acts[layer_idx].append(act[0].cpu())
        if has_head:
            logits.append(out.logits[0].detach().cpu())

    return PassResult(acts, logits if has_head else None)


def compute_metric_per_example(acts, metric_fn):
    """layer_idx -> np.array of per-example metric values (not averaged)."""
    return {
        layer_idx: np.array([metric_fn(a) for a in acts_list], dtype=np.float64)
        for layer_idx, acts_list in acts.items()
    }


# -----------------------------------------------------------------------
# TASK PERFORMANCE + SANITY CHECK
# -----------------------------------------------------------------------
def score_task(spec, examples, logits):
    """
    Accuracy of `logits` against gold labels. Sequence tasks: argmax over
    classes. CoNLL: accuracy over words, using each word's first subword.
    """
    correct = total = 0
    for example, lg in zip(examples, logits):
        if spec.input_kind == "tokens":
            pred = lg.argmax(-1)
            seen = set()
            for pos, word in enumerate(example.word_ids):
                if word is None or word in seen:
                    continue
                seen.add(word)
                total += 1
                correct += int(pred[pos].item() == example.label[word])
        else:
            total += 1
            correct += int(lg.argmax(-1).item() == example.label)
    return correct / max(total, 1)


def check_full_restoration(clean: PassResult, restored: PassResult, tol=SANITY_TOL):
    """
    Full restoration forces every layer to its clean value, so it must
    reproduce the clean pass exactly. A nonzero diff is a wiring bug (hook
    leaked across examples, wrong layer index), not a research finding.
    """
    max_act_diff = 0.0
    for layer_idx in range(NUM_LAYERS):
        for c, r in zip(clean.acts[layer_idx], restored.acts[layer_idx]):
            max_act_diff = max(max_act_diff, (c - r).abs().max().item())

    result = {"max_activation_diff": max_act_diff, "passed": max_act_diff <= tol}
    if clean.logits is not None:
        max_logit_diff = max(
            (c - r).abs().max().item() for c, r in zip(clean.logits, restored.logits)
        )
        same_pred = all(
            bool((c.argmax(-1) == r.argmax(-1)).all())
            for c, r in zip(clean.logits, restored.logits)
        )
        result["max_logit_diff"] = max_logit_diff
        result["predictions_identical"] = same_pred
        result["passed"] = result["passed"] and same_pred
    return result


def task_performance_summary(spec, examples, clean, collapsed, restored_by_layer, full):
    """Accuracy for clean / fully collapsed / each single-layer restoration /
    full restoration. None if the model has no task head."""
    if clean.logits is None:
        return None

    def acc(res):
        return score_task(spec, examples, res.logits)

    return {
        "metric": "token_accuracy" if spec.input_kind == "tokens" else "accuracy",
        "clean": acc(clean),
        "fully_collapsed": acc(collapsed),
        "restored_by_layer": [acc(restored_by_layer[i]) for i in range(NUM_LAYERS)],
        "fully_restored": acc(full),
    }


# -----------------------------------------------------------------------
# M_bwd + ROW SUMMARIES
# -----------------------------------------------------------------------
def bootstrap_ci(per_example_values, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """95% CI of the mean, resampling EXAMPLES (not matrix cells) with replacement."""
    rng = np.random.default_rng(seed)
    n = len(per_example_values)
    means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        means[b] = per_example_values[rng.integers(0, n, size=n)].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def compute_recovery_tensor(clean_m, collapsed_m, restored_m, metric_name):
    """
    clean_m / collapsed_m: layer -> array[n_examples]
    restored_m: source_layer -> (layer -> array[n_examples])
    Returns R[i][j] = per-example recovery at layer j from restoring layer i,
    shape [L, L, n_examples], NaN where j < i (not causally reachable).
    """
    n = len(clean_m[0])
    R = np.full((NUM_LAYERS, NUM_LAYERS, n), np.nan)
    for i in range(NUM_LAYERS):
        for j in range(i, NUM_LAYERS):
            baseline = compute_damage(clean_m[j], collapsed_m[j], metric_name)
            remaining = compute_damage(clean_m[j], restored_m[i][j], metric_name)
            R[i, j] = baseline - remaining
    return R


def matrix_from_recovery(R):
    """Mean over examples; unreachable cells (j < i) are stored as 0.0."""
    M = np.zeros((NUM_LAYERS, NUM_LAYERS))
    for i in range(NUM_LAYERS):
        for j in range(i, NUM_LAYERS):
            M[i, j] = R[i, j].mean()
    return M


VIEW_KEYS = {
    "row_avg": "row_avg_recovery",
    "adjacent": "adjacent_recovery",
    "final_layer": "final_layer_recovery",
}


def summarize_rows(R):
    """
    Healing-site scores from DOWNSTREAM cells only (j > i). The diagonal
    M_bwd[i][i] is excluded: layer i is patched to its own clean value, so
    that cell is just the baseline damage at i, not evidence it heals anything
    else. The last layer has no downstream cells, so its scores are NaN.

    Three views, because a plain row average over j > i gives higher layers
    fewer, nearer cells (a column-count artifact):
      row_avg      mean over all j > i             (doc Step 4 row impact)
      adjacent     the single cell j = i + 1       (fixed-distance check)
      final_layer  the single cell j = L - 1       (doc: "across the final layers")
    """
    nan_row = [float("nan")] * NUM_LAYERS
    nan_ci = (float("nan"), float("nan"))
    out = {
        "propagation_slope": list(nan_row),
        "per_sentence_row_avg_recovery": [None] * NUM_LAYERS,
    }
    for view, val_key in VIEW_KEYS.items():
        out[val_key] = list(nan_row)
        out[f"{view}_bootstrap_ci95"] = [nan_ci] * NUM_LAYERS

    for i in range(NUM_LAYERS - 1):
        per_ex_row = R[i, i + 1 :].mean(axis=0)
        views = {
            "row_avg": per_ex_row,
            "adjacent": R[i, i + 1],
            "final_layer": R[i, NUM_LAYERS - 1],
        }
        out["per_sentence_row_avg_recovery"][i] = per_ex_row
        for view, values in views.items():
            out[VIEW_KEYS[view]][i] = float(values.mean())
            out[f"{view}_bootstrap_ci95"][i] = bootstrap_ci(values)

        # Slope of mean recovery vs. layer index over j > i (needs >= 2 cells).
        if NUM_LAYERS - 1 - i >= 2:
            js = np.arange(i + 1, NUM_LAYERS)
            col_means = R[i, i + 1 :].mean(axis=1)
            out["propagation_slope"][i] = float(np.polyfit(js, col_means, 1)[0])
    return out


def top_layer(values):
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return None
    return int(np.nanargmax(values))


def compare_top_layers(values, cis, label):
    """Is the top healing site distinguishable from the runner-up (CI overlap)?"""
    valid = [i for i, v in enumerate(values) if not np.isnan(v)]
    if len(valid) < 2:
        return None
    order = sorted(valid, key=lambda i: values[i], reverse=True)
    t1, t2 = order[0], order[1]
    (lo1, hi1), (lo2, hi2) = cis[t1], cis[t2]
    overlap = not (hi2 < lo1 or hi1 < lo2)
    verdict = "CIs OVERLAP (not distinguishable)" if overlap else "CIs do not overlap"
    print(
        f"      [{label}] top: L{t1} ({values[t1]:.4f}) vs runner-up: "
        f"L{t2} ({values[t2]:.4f}) -> {verdict}"
    )
    return {"top": t1, "runner_up": t2, "ci_overlap": overlap}


def cross_check_with_forward(fwd_dir, task, metric_name, top_recovery_layer):
    """
    Joint Step 4 cross-check: highest-damage row of M_fwd vs highest-recovery
    row of M_bwd. A mismatch is a finding to report to the group, not a bug to
    resolve quietly. M_fwd is scored the same way as M_bwd (downstream cells
    only, diagonal excluded) so the comparison is like for like. Skipped if
    M_fwd is not available.
    """
    path = os.path.join(fwd_dir, f"M_fwd_{task}_{metric_name}.json")
    if not os.path.exists(path):
        print(f"      [cross-check] {path} not found - skipping.")
        return None
    with open(path) as f:
        fwd = json.load(f)
    mat = np.array(fwd["matrix"], dtype=float)
    down = [
        mat[i, i + 1 :].mean() if i < NUM_LAYERS - 1 else np.nan
        for i in range(NUM_LAYERS)
    ]
    top_down = top_layer(down)
    reported = fwd.get("row_avg_damage")
    top_reported = top_layer(reported) if reported is not None else None
    match = top_down == top_recovery_layer
    print(
        f"      [cross-check] M_fwd top-damage layer (downstream-only): {top_down}"
        f" | as reported in file: {top_reported} | M_bwd top healing layer: "
        f"{top_recovery_layer} -> {'MATCH' if match else 'MISMATCH - flag to the group'}"
    )
    return {
        "top_damage_layer_downstream_only": top_down,
        "top_damage_layer_as_reported": top_reported,
        "top_healing_layer": top_recovery_layer,
        "match": match,
    }


def _jsonable(obj):
    """NaN -> None so files are valid JSON; numpy -> python."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def build_result(
    task,
    metric_name,
    M,
    summary,
    target_rank_used,
    rank_fraction,
    min_rank,
    model_source,
    has_head,
    n_examples,
    baseline_damage,
    task_perf,
    sanity,
):
    return _jsonable(
        {
            "task": task,
            "direction": "backward",
            "granularity": "layer",
            "matrix": M,
            "metric": metric_name,
            "target_rank_used": target_rank_used,
            "rank_fraction_used": rank_fraction,
            "min_rank_used": min_rank,
            "model_source": model_source,
            "has_task_head": has_head,
            "n_probe_examples": n_examples,
            **summary,
            "top_healing_layer": {
                view: top_layer(summary[val_key]) for view, val_key in VIEW_KEYS.items()
            },
            "baseline_damage_per_layer": baseline_damage,
            "task_performance": task_perf,
            "sanity_check": sanity,
        }
    )


# -----------------------------------------------------------------------
# MODEL LOADING + PER-TASK DRIVER
# -----------------------------------------------------------------------
def load_model_and_tokenizer(spec, checkpoint_dir, allow_base_model):
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    if checkpoint_dir and os.path.isdir(checkpoint_dir):
        source = checkpoint_dir
        cls = (
            AutoModelForTokenClassification
            if spec.input_kind == "tokens"
            else AutoModelForSequenceClassification
        )
    elif allow_base_model:
        print(
            f"WARNING: no fine-tuned checkpoint at '{checkpoint_dir}'. Using "
            f"'{BASE_MODEL_NAME}' with NO task head: geometry metrics only, "
            "no task performance, and not comparable to M_fwd from a "
            "fine-tuned model."
        )
        source, cls = BASE_MODEL_NAME, AutoModel
    else:
        raise FileNotFoundError(
            f"No fine-tuned checkpoint at '{checkpoint_dir}' for task "
            f"'{spec.name}'. Fine-tune one first, or pass --allow_base_model "
            "to run geometry-only on the pretrained model."
        )
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = cls.from_pretrained(source).to(DEVICE)
    model.eval()
    return model, tokenizer, source


def target_rank_median(examples, hidden_dim, rank_fraction, min_rank):
    ranks = []
    for e in examples:
        max_rank = min(e.seq_len, hidden_dim)
        ranks.append(min(max(min_rank, int(max_rank * rank_fraction)), max_rank))
    return int(np.median(ranks))


def run_task(
    task,
    checkpoint_dir,
    out_dir,
    fwd_dir,
    rank_fraction=RANK_FRACTION,
    min_rank=MIN_RANK,
    n_probe=N_PROBE_EXAMPLES,
    allow_base_model=False,
):
    spec = TASKS[task]
    print(f"\n{'=' * 70}\nTASK: {task}\n{'=' * 70}")
    model, tokenizer, source = load_model_and_tokenizer(
        spec, checkpoint_dir, allow_base_model
    )
    has_head = model_has_head(model)
    examples = load_probe_examples(tokenizer, spec, n_probe)
    all_layers = set(range(NUM_LAYERS))
    kw = dict(rank_fraction=rank_fraction, min_rank=min_rank)

    print("[1/4] clean pass")
    clean = run_pass(model, tokenizer, spec, examples)
    print(f"[2/4] fully collapsed pass (rank_fraction={rank_fraction})")
    collapsed = run_pass(
        model, tokenizer, spec, examples, collapse_layers=all_layers, **kw
    )
    restored = {}
    for i in range(NUM_LAYERS):
        print(f"[3/4] restoring layer {i}, collapsing the rest")
        restored[i] = run_pass(
            model,
            tokenizer,
            spec,
            examples,
            clean_acts=clean.acts,
            restore_layers={i},
            collapse_layers=all_layers - {i},
            **kw,
        )
    print("[4/4] full restoration sanity check")
    full = run_pass(
        model,
        tokenizer,
        spec,
        examples,
        clean_acts=clean.acts,
        restore_layers=all_layers,
    )
    sanity = check_full_restoration(clean, full)
    print(f"      {sanity}")
    if not sanity["passed"]:
        raise RuntimeError(
            "Full restoration does not reproduce the clean pass - fix the "
            "hook wiring before trusting any M_bwd numbers."
        )

    task_perf = task_performance_summary(
        spec, examples, clean, collapsed, restored, full
    )
    if task_perf is not None:
        print(
            f"      {task_perf['metric']}: clean={task_perf['clean']:.3f} "
            f"fully_collapsed={task_perf['fully_collapsed']:.3f} "
            f"fully_restored={task_perf['fully_restored']:.3f} "
            f"restored_by_layer={[round(a, 3) for a in task_perf['restored_by_layer']]}"
        )

    hidden_dim = clean.acts[0][0].shape[-1]
    target_rank = target_rank_median(examples, hidden_dim, rank_fraction, min_rank)
    os.makedirs(out_dir, exist_ok=True)
    results = []

    for metric_name, metric_fn in METRIC_FUNCS.items():
        print(f"\n--- {task} / {metric_name} ---")
        clean_m = compute_metric_per_example(clean.acts, metric_fn)
        collapsed_m = compute_metric_per_example(collapsed.acts, metric_fn)
        restored_m = {
            i: compute_metric_per_example(restored[i].acts, metric_fn)
            for i in range(NUM_LAYERS)
        }
        baseline_damage = [
            float(compute_damage(clean_m[j], collapsed_m[j], metric_name).mean())
            for j in range(NUM_LAYERS)
        ]
        print(
            f"      baseline damage per layer: {[round(d, 3) for d in baseline_damage]}"
        )
        if max(abs(d) for d in baseline_damage) < 1e-6:
            print("      WARNING: the collapse changed nothing - check RANK_FRACTION.")

        R = compute_recovery_tensor(clean_m, collapsed_m, restored_m, metric_name)
        M = matrix_from_recovery(R)
        summary = summarize_rows(R)
        result = build_result(
            task,
            metric_name,
            M,
            summary,
            target_rank,
            rank_fraction,
            min_rank,
            source,
            has_head,
            len(examples),
            baseline_damage,
            task_perf,
            sanity,
        )
        path = os.path.join(out_dir, f"M_bwd_{task}_{metric_name}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"      saved -> {path}")

        for key, val_key in VIEW_KEYS.items():
            vals = summary[val_key]
            print(f"      {key:11s}: {[round(v, 3) for v in vals]}")
            compare_top_layers(vals, summary[f"{key}_bootstrap_ci95"], key)
        print(
            f"      slopes     : {[round(v, 3) for v in summary['propagation_slope']]}"
        )
        cross_check_with_forward(
            fwd_dir, task, metric_name, result["top_healing_layer"]["row_avg"]
        )
        results.append(result)

    del model
    return results


def main():
    parser = argparse.ArgumentParser(description="M4 Step 3: backward restoration")
    parser.add_argument(
        "--tasks", default="sst2", help="comma list of sst2,mnli,conll2003"
    )
    parser.add_argument(
        "--checkpoint_root",
        default="./checkpoints",
        help="expects {root}/{task}_finetuned (same as finetune_sst2.py)",
    )
    parser.add_argument("--allow_base_model", action="store_true")
    parser.add_argument("--out_dir", default="results/m4")
    parser.add_argument("--fwd_dir", default=None, help="where M_fwd_*.json live")
    parser.add_argument("--rank_fraction", type=float, default=RANK_FRACTION)
    parser.add_argument("--min_rank", type=int, default=MIN_RANK)
    parser.add_argument("--n_probe", type=int, default=N_PROBE_EXAMPLES)
    args = parser.parse_args()

    for task in args.tasks.split(","):
        task = task.strip()
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Available: {list(TASKS)}")
        run_task(
            task,
            os.path.join(args.checkpoint_root, f"{task}_finetuned"),
            args.out_dir,
            args.fwd_dir or args.out_dir,
            rank_fraction=args.rank_fraction,
            min_rank=args.min_rank,
            n_probe=args.n_probe,
            allow_base_model=args.allow_base_model,
        )


if __name__ == "__main__":
    main()