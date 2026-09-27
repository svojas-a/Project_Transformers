"""
SST-2 Backward Restoration
==========================

Baseline:
    collapse ALL 6 layers.

Intervention:
    restore exactly ONE layer i to its clean activation,
    while all other layers remain collapsed.

Measure:
    downstream recovery at j > i.

The same checkpoint, probe set, metrics, rank fraction and sign convention
are used as the forward intervention.
"""

import json
from pathlib import Path
import numpy as np

from common import (
    TASK_NAME,
    RANK_FRACTION,
    MIN_RANK,
    N_LAYERS,
    N_PROBE_SENTENCES,
    load_model_and_tokenizer,
    load_probe_sentences,
    save_probe_set,
    run_clean_pass,
    run_fully_collapsed_pass,
    run_backward_intervention,
    compute_metric_per_sentence,
    compute_damage,
    bootstrap_mean_ci,
    METRIC_FUNCS,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "bwd_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROBE_PATH = ROOT.parent / "probe_set_sst2.json"


def sanity_check_full_restoration(model, tokenizer, sentences, clean_acts):
    """
    Restore every layer to its clean activation simultaneously.

    This should reproduce the clean hidden states exactly up to floating-point
    tolerance. It validates the restoration hook before single-layer recovery
    results are trusted.
    """
    # We implement the all-layer restoration directly here so the test is
    # independent of the single-layer experiment.
    from common import LayerRestorer, LayerRecorder

    max_diff = 0.0
    n_match = 0

    for sent_idx, sentence in enumerate(sentences):
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
        ).to("cuda" if next(model.parameters()).is_cuda else "cpu")

        restorers = [
            LayerRestorer(
                model,
                layer_idx,
                clean_acts[layer_idx][sent_idx],
            )
            for layer_idx in range(N_LAYERS)
        ]

        recorder = LayerRecorder(model)

        import torch
        with torch.inference_mode():
            model(**inputs)

        for layer_idx in range(N_LAYERS):
            diff = (
                recorder.activations[layer_idx][0]
                - clean_acts[layer_idx][sent_idx]
            ).abs().max().item()

            max_diff = max(max_diff, diff)

        recorder.remove()
        for restorer in restorers:
            restorer.remove()

        # Classification prediction agreement.
        # Re-run clean and restored predictions for this sentence.
        with torch.inference_mode():
            clean_out = model(**inputs)

        restorers = [
            LayerRestorer(
                model,
                layer_idx,
                clean_acts[layer_idx][sent_idx],
            )
            for layer_idx in range(N_LAYERS)
        ]

        with torch.inference_mode():
            restored_out = model(**inputs)

        for restorer in restorers:
            restorer.remove()

        clean_pred = int(clean_out.logits.argmax(dim=-1).item())
        restored_pred = int(
            restored_out.logits.argmax(dim=-1).item()
        )
        n_match += int(clean_pred == restored_pred)

    agreement = n_match / len(sentences)

    print(
        f"Full restoration hidden-state max diff: {max_diff:.8g}"
    )
    print(
        f"Full restoration prediction agreement: "
        f"{n_match}/{len(sentences)} = {agreement:.1%}"
    )

    if max_diff > 1e-5:
        raise RuntimeError(
            "Full restoration failed hidden-state equality. "
            "Do NOT trust backward results."
        )

    if agreement < 1.0:
        raise RuntimeError(
            "Full restoration did not reproduce clean predictions. "
            "Do NOT trust backward results."
        )


def build_backward_matrix(
    model, tokenizer, sentences, metric_name, clean_acts
):
    print(f"\n--- BACKWARD: {metric_name} ---")

    clean = compute_metric_per_sentence(
        clean_acts, metric_name
    )

    print("Running fully collapsed baseline...")
    fully_collapsed_acts = run_fully_collapsed_pass(
        model,
        tokenizer,
        sentences,
        RANK_FRACTION,
        MIN_RANK,
    )
    fully_collapsed = compute_metric_per_sentence(
        fully_collapsed_acts,
        metric_name,
    )

    matrix = np.full((N_LAYERS, N_LAYERS), np.nan)
    row_per_sentence = [None] * N_LAYERS
    row_ci95 = [None] * N_LAYERS

    for restore_layer in range(N_LAYERS):
        if restore_layer == N_LAYERS - 1:
            print(
                "Layer 5: no downstream layers; "
                "healing score = undefined."
            )
            continue

        print(f"\nRestoring layer {restore_layer}...")

        restored_acts = run_backward_intervention(
            model,
            tokenizer,
            sentences,
            clean_acts,
            restore_layer,
            RANK_FRACTION,
            MIN_RANK,
        )

        restored = compute_metric_per_sentence(
            restored_acts,
            metric_name,
        )

        downstream_recovery = []

        for target_layer in range(
            restore_layer + 1, N_LAYERS
        ):
            baseline_damage = compute_damage(
                clean[target_layer],
                fully_collapsed[target_layer],
                metric_name,
            )

            remaining_damage = compute_damage(
                clean[target_layer],
                restored[target_layer],
                metric_name,
            )

            recovery = baseline_damage - remaining_damage

            matrix[restore_layer, target_layer] = float(
                np.mean(recovery)
            )
            downstream_recovery.append(recovery)

        downstream_recovery = np.stack(
            downstream_recovery, axis=1
        )

        row_per_sentence[restore_layer] = (
            downstream_recovery.mean(axis=1)
        )
        row_ci95[restore_layer] = bootstrap_mean_ci(
            row_per_sentence[restore_layer]
        )

        print(
            f"  downstream-only recovery = "
            f"{row_per_sentence[restore_layer].mean():.5f} "
            f"CI={row_ci95[restore_layer]}"
        )

    row_avg = [
        None if x is None else float(np.mean(x))
        for x in row_per_sentence
    ]

    return matrix, row_avg, row_ci95


def main():
    print("=" * 72)
    print("SST-2 BACKWARD RESTORATION")
    print("=" * 72)

    model, tokenizer = load_model_and_tokenizer()

    sentences, indices = load_probe_sentences(
        tokenizer,
        N_PROBE_SENTENCES,
    )

    # IMPORTANT: overwrite/save the same probe definition used by forward.
    save_probe_set(sentences, indices, PROBE_PATH)

    print(f"Shared probe set -> {PROBE_PATH}")

    print("\nComputing clean activations...")
    clean_acts = run_clean_pass(
        model,
        tokenizer,
        sentences,
    )

    print("\nRunning full-restoration wiring test...")
    sanity_check_full_restoration(
        model,
        tokenizer,
        sentences[:10],
        {
            layer: acts[:10]
            for layer, acts in clean_acts.items()
        },
    )

    for metric_name in METRIC_FUNCS:
        matrix, row_avg, row_ci95 = build_backward_matrix(
            model,
            tokenizer,
            sentences,
            metric_name,
            clean_acts,
        )

        result = {
            "task": TASK_NAME,
            "direction": "backward",
            "granularity": "layer",
            "metric": metric_name,
            "rank_fraction_used": RANK_FRACTION,
            "min_rank_used": MIN_RANK,
            "probe_split": "validation",
            "probe_seed": 42,
            "n_probe_sentences": len(sentences),
            "matrix": matrix.tolist(),
            "row_avg_recovery_downstream_only": row_avg,
            "row_avg_bootstrap_ci95": row_ci95,
            "interpretation": (
                "matrix[i][j] is mean downstream recovery at layer j "
                "when only layer i is restored to its clean activation "
                "while all other layers remain collapsed; j <= i is "
                "excluded from recovery summaries."
            ),
        }

        out_path = OUT_DIR / f"M_bwd_{TASK_NAME}_{metric_name}.json"
        out_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        print(f"\nSAVED: {out_path}")


if __name__ == "__main__":
    main()
