
"""
SST-2 Backward Restoration
==========================

Backward restoration experiment corresponding to the forward
low-rank intervention.

For each source layer i:
    - collapse every layer
    - restore layer i to its clean activation
    - measure downstream recovery

The diagonal is excluded from the healing-site row average.
The final layer has no downstream layers and therefore receives NaN.
"""

import json
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ....sst2.sst2 import common

from ..forward.forward_causal_intervention import (
    METRIC_FUNCS,
    MIN_RANK,
    N_LAYERS,
    N_PROBE_SENTENCES,
    RANK_FRACTION,
    TASK_NAME,
    compute_damage,
    compute_metric_per_sentence,
    load_probe_sentences,
    run_clean_pass,
)


# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------

MODEL_CHECKPOINT_PATH = common.MODEL_CHECKPOINT_PATH
DEVICE = common.DEVICE

NUM_LAYERS = N_LAYERS

IS_CLASSIFICATION_CHECKPOINT = True


# -----------------------------------------------------------------------
# MODEL LAYER ACCESS
# -----------------------------------------------------------------------

def get_transformer_layers(model):
    """
    Return the DistilBERT transformer layers.
    """

    return model.distilbert.transformer.layer


# -----------------------------------------------------------------------
# LAYER RECORDER
# -----------------------------------------------------------------------

class LayerRecorder:

    def __init__(self, model):

        self.activations = {}
        self.handles = []

        layers = get_transformer_layers(model)

        for layer_idx, layer in enumerate(layers):

            handle = layer.register_forward_hook(
                self._make_hook(layer_idx)
            )

            self.handles.append(handle)

    def _make_hook(self, layer_idx):

        def hook(module, inputs, output):

            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output

            self.activations[layer_idx] = activation.detach()

        return hook

    def remove(self):

        for handle in self.handles:
            handle.remove()

        self.handles = []


# -----------------------------------------------------------------------
# LAYER COLLAPSER
# -----------------------------------------------------------------------

class LayerCollapser:

    def __init__(
        self,
        model,
        layer_idx,
        rank_fraction,
        min_rank,
    ):

        self.handle = None

        self.rank_fraction = rank_fraction
        self.min_rank = min_rank

        layer = get_transformer_layers(model)[layer_idx]

        self.handle = layer.register_forward_hook(
            self._collapse_hook
        )

    def _collapse_hook(
        self,
        module,
        inputs,
        output,
    ):

        if isinstance(output, tuple):

            hidden = output[0]

        else:

            hidden = output

        # hidden:
        # [batch, seq_len, hidden_dim]

        x = hidden

        batch_size, seq_len, hidden_dim = x.shape

        patched = []

        for b in range(batch_size):

            H = x[b]

            # SVD rank is bounded by min(seq_len, hidden_dim)
            max_rank = min(
                H.shape[0],
                H.shape[1],
            )

            rank = max(
                self.min_rank,
                int(
                    np.floor(
                        self.rank_fraction * max_rank
                    )
                ),
            )

            rank = min(
                rank,
                max_rank,
            )

            # SVD
            U, S, Vh = torch.linalg.svd(
                H,
                full_matrices=False,
            )

            # Rank-truncated reconstruction
            H_low_rank = (
                U[:, :rank]
                @ torch.diag(S[:rank])
                @ Vh[:rank, :]
            )

            patched.append(
                H_low_rank
            )

        patched = torch.stack(
            patched,
            dim=0,
        )

        if isinstance(output, tuple):

            return (
                patched,
            ) + output[1:]

        return patched

    def remove(self):

        if self.handle is not None:
            self.handle.remove()

        self.handle = None


# -----------------------------------------------------------------------
# RESTORER
# -----------------------------------------------------------------------

class LayerRestorer:

    def __init__(
        self,
        model,
        layer_idx,
        clean_tensor,
    ):

        self.handle = None

        self.clean_tensor = clean_tensor

        layer = get_transformer_layers(model)[layer_idx]

        self.handle = layer.register_forward_hook(
            self._restore_hook
        )

    def _restore_hook(
        self,
        module,
        inputs,
        output,
    ):

        patched = self.clean_tensor.unsqueeze(0)

        if isinstance(output, tuple):

            return (
                patched,
            ) + output[1:]

        return patched

    def remove(self):

        if self.handle is not None:
            self.handle.remove()

        self.handle = None


# -----------------------------------------------------------------------
# BOOTSTRAP 95% CI
# -----------------------------------------------------------------------

def bootstrap_row_ci(values, n_boot=1000, seed=42):
    """
    Bootstrap 95% confidence interval for a row-average recovery score.

    Resamples probe sentences, not matrix cells.
    """
    values = np.asarray(values, dtype=float)

    rng = np.random.default_rng(seed)

    n = len(values)

    boot_means = np.empty(n_boot, dtype=float)

    for b in range(n_boot):

        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        boot_means[b] = sample.mean()

    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))

    return lo, hi


# -----------------------------------------------------------------------
# RESTORED PASS
# -----------------------------------------------------------------------

def run_restored_pass(
    model,
    tokenizer,
    sentences,
    clean_acts,
    patch_layers,
    rank_fraction,
    min_rank,
):

    per_sentence_acts = {
        i: []
        for i in range(NUM_LAYERS)
    }

    for sent_idx, sent in enumerate(sentences):

        inputs = tokenizer(
            sent,
            return_tensors="pt",
        ).to(DEVICE)

        mutators = []

        for layer_idx in range(NUM_LAYERS):

            if layer_idx in patch_layers:

                clean_tensor = (
                    clean_acts[layer_idx][sent_idx]
                )

                mutators.append(
                    LayerRestorer(
                        model,
                        layer_idx,
                        clean_tensor,
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

        # IMPORTANT:
        # Mutators before recorder.
        recorder = LayerRecorder(model)

        with torch.no_grad():

            model(**inputs)

        for layer_idx, activation in recorder.activations.items():

            per_sentence_acts[layer_idx].append(
                activation[0]
            )

        recorder.remove()

        for mutator in mutators:
            mutator.remove()

    return per_sentence_acts


# -----------------------------------------------------------------------
# FULL RESTORATION SANITY CHECK
# -----------------------------------------------------------------------

def sanity_check_full_restoration(
    model,
    tokenizer,
    sentences,
):

    print(
        f"Running clean + fully-restored passes on "
        f"{len(sentences)} sentences..."
    )

    clean_acts = run_clean_pass(
        model,
        tokenizer,
        sentences,
    )

    restored_acts = run_restored_pass(
        model,
        tokenizer,
        sentences,
        clean_acts,
        patch_layers=set(
            range(NUM_LAYERS)
        ),
        rank_fraction=RANK_FRACTION,
        min_rank=MIN_RANK,
    )

    max_diff = 0.0

    for layer_idx in range(NUM_LAYERS):

        for clean_t, restored_t in zip(
            clean_acts[layer_idx],
            restored_acts[layer_idx],
        ):

            diff = (
                clean_t - restored_t
            ).abs().max().item()

            max_diff = max(
                max_diff,
                diff,
            )

    print(
        f"Max abs diff, clean vs fully-restored "
        f"hidden states: {max_diff:.8f}"
    )

    if max_diff <= 1e-5:

        print(
            "OK: patching mechanism reproduces "
            "the clean pass exactly."
        )

    else:

        print(
            "WARNING: fully-restored pass does not "
            "reproduce the clean pass."
        )

    # ---------------------------------------------------------------
    # Classification prediction check
    # ---------------------------------------------------------------

    matches = 0

    for sent_idx, sent in enumerate(sentences):

        inputs = tokenizer(
            sent,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():

            clean_pred = (
                model(**inputs)
                .logits
                .argmax(-1)
                .item()
            )

        mutators = []

        for layer_idx in range(NUM_LAYERS):

            mutators.append(
                LayerRestorer(
                    model,
                    layer_idx,
                    clean_acts[layer_idx][sent_idx],
                )
            )

        with torch.no_grad():

            restored_pred = (
                model(**inputs)
                .logits
                .argmax(-1)
                .item()
            )

        for mutator in mutators:
            mutator.remove()

        if clean_pred == restored_pred:
            matches += 1

    percentage = (
        100.0 * matches / len(sentences)
    )

    print(
        f"Classification agreement: "
        f"{matches}/{len(sentences)} "
        f"= {percentage:.1f}%"
    )


# -----------------------------------------------------------------------
# BUILD BACKWARD MATRIX
# -----------------------------------------------------------------------

def build_backward_matrix(
    model,
    tokenizer,
    sentences,
    metric_name,
):

    metric_fn = METRIC_FUNCS[metric_name]

    # ---------------------------------------------------------------
    # CLEAN
    # ---------------------------------------------------------------

    print(
        f"\n[1/3] CLEAN baseline ({metric_name})..."
    )

    clean_acts = run_clean_pass(
        model,
        tokenizer,
        sentences,
    )

    clean_metrics = compute_metric_per_sentence(
        clean_acts,
        metric_name,
    )

    # ---------------------------------------------------------------
    # FULLY COLLAPSED
    # ---------------------------------------------------------------

    print(
        f"\n[2/3] FULLY COLLAPSED baseline..."
    )

    collapsed_acts = run_restored_pass(
        model,
        tokenizer,
        sentences,
        clean_acts,
        patch_layers=set(),
        rank_fraction=RANK_FRACTION,
        min_rank=MIN_RANK,
    )

    collapsed_metrics = compute_metric_per_sentence(
        collapsed_acts,
        metric_name,
    )

    # ---------------------------------------------------------------
    # MATRIX
    # ---------------------------------------------------------------

    M_bwd = np.full(
        (NUM_LAYERS, NUM_LAYERS),
        np.nan,
    )

    per_sentence_rows = [
        None
        for _ in range(NUM_LAYERS)
    ]

    row_ci95 = [
        None
        for _ in range(NUM_LAYERS)
    ]

    # ---------------------------------------------------------------
    # RESTORE EACH LAYER
    # ---------------------------------------------------------------

    for source_layer in range(NUM_LAYERS):

        print(
            f"\n[3/3] Restoring layer "
            f"{source_layer}..."
        )

        restored_acts = run_restored_pass(
            model,
            tokenizer,
            sentences,
            clean_acts,
            patch_layers={source_layer},
            rank_fraction=RANK_FRACTION,
            min_rank=MIN_RANK,
        )

        restored_metrics = compute_metric_per_sentence(
            restored_acts,
            metric_name,
        )

        valid_layers = range(
            source_layer,
            NUM_LAYERS,
        )

        recoveries = []

        for j in valid_layers:

            recovery = (
                compute_damage(
                    clean_metrics[j],
                    collapsed_metrics[j],
                    metric_name,
                )
                -
                compute_damage(
                    clean_metrics[j],
                    restored_metrics[j],
                    metric_name,
                )
            )

            M_bwd[source_layer][j] = (
                recovery.mean()
            )

            recoveries.append(
                recovery
            )

        # -----------------------------------------------------------
        # DOWNSTREAM ONLY
        # -----------------------------------------------------------

        downstream_layers = range(
            source_layer + 1,
            NUM_LAYERS,
        )

        if len(list(downstream_layers)) > 0:

            # First recovery corresponds to diagonal.
            downstream_recoveries = np.stack(
                recoveries[1:],
                axis=1,
            )

            per_sentence_rows[source_layer] = (
                downstream_recoveries.mean(
                    axis=1
                )
            )

            row_ci95[source_layer] = (
                bootstrap_row_ci(
                    per_sentence_rows[source_layer]
                )
            )

            avg = (
                per_sentence_rows[source_layer]
                .mean()
            )

            lo, hi = row_ci95[source_layer]

            print(
                f"Layer {source_layer}: "
                f"downstream-only recovery = "
                f"{avg:.5f} "
                f"CI=[{lo:.5f}, {hi:.5f}]"
            )

        else:

            print(
                f"Layer {source_layer}: "
                f"no downstream layers -> NaN"
            )

    return (
        M_bwd,
        per_sentence_rows,
        row_ci95,
    )


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():

    print("=" * 70)
    print("SST-2 BACKWARD RESTORATION")
    print("=" * 70)

    print(
        f"Loading model from: "
        f"{MODEL_CHECKPOINT_PATH}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CHECKPOINT_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT_PATH
    ).to(DEVICE)

    model.eval()

    print(
        f"Layers found: "
        f"{len(get_transformer_layers(model))}"
    )

    probe_sentences, probe_indices = (
        load_probe_sentences(
            tokenizer,
            N_PROBE_SENTENCES,
        )
    )

    print(
        f"Probe set: "
        f"{len(probe_sentences)} SST-2 validation "
        f"sentences | seed=42"
    )

    # ---------------------------------------------------------------
    # SANITY CHECK
    # ---------------------------------------------------------------

    print("=" * 70)
    print(
        "Running full-restoration wiring test..."
    )
    print("=" * 70)

    sanity_check_full_restoration(
        model,
        tokenizer,
        probe_sentences[:10],
    )

    # ---------------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------------

    out_dir = (
        Path(__file__).resolve().parent /
        "bwd_results"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for metric_name in METRIC_FUNCS:

        print("=" * 70)
        print(
            f"METRIC: {metric_name}"
        )
        print("=" * 70)

        (
            M_bwd,
            per_sentence_rows,
            row_ci95,
        ) = build_backward_matrix(
            model,
            tokenizer,
            probe_sentences,
            metric_name,
        )

        row_avgs = np.array([
            values.mean()
            if values is not None
            else float("nan")
            for values in per_sentence_rows
        ])

        result = {
            "task": TASK_NAME,
            "direction": "backward",
            "granularity": "layer",
            "metric": metric_name,
            "rank_fraction_used": RANK_FRACTION,
            "min_rank_used": MIN_RANK,
            "probe_split": "validation",
            "probe_seed": 42,
            "n_probe_sentences": len(
                probe_sentences
            ),
            "matrix": M_bwd.tolist(),
            "row_avg_recovery_downstream_only":
                row_avgs.tolist(),
            "row_avg_bootstrap_ci95": [
                list(ci)
                if ci is not None
                else [
                    float("nan"),
                    float("nan"),
                ]
                for ci in row_ci95
            ],
            "interpretation":
                "matrix[i][j] is recovery at "
                "downstream layer j caused by "
                "restoring source layer i; "
                "j < i is undefined and the "
                "diagonal is excluded from "
                "row summaries.",
        }

        out_path = (
            out_dir /
            f"M_bwd_{TASK_NAME}_{metric_name}.json"
        )

        with open(out_path, "w") as f:

            json.dump(
                result,
                f,
                indent=2,
            )

        print(
            f"SAVED: {out_path}"
        )

        valid = ~np.isnan(row_avgs)

        if np.any(valid):

            top_layer = int(
                np.nanargmax(row_avgs)
            )

            print(
                f"Highest downstream-only "
                f"recovery: Layer {top_layer} "
                f"({row_avgs[top_layer]:.5f})"
            )


if __name__ == "__main__":
    main()
