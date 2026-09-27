"""
SST-2 Forward Causal Intervention
=================================

For each source layer i:
    clean model
        ->
    collapse ONLY layer i by SVD rank truncation
        ->
    measure downstream representation damage.

IMPORTANT:
- The diagonal is NOT used for the source-layer summary.
- Only j > i is used.
- Layer 5 therefore has no downstream score and is NaN.
- Results are written to forward/fwd_results/.
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
    run_forward_intervention,
    compute_metric_per_sentence,
    compute_damage,
    bootstrap_mean_ci,
    METRIC_FUNCS,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "fwd_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROBE_PATH = ROOT.parent / "probe_set_sst2.json"


def sanity_check(model, tokenizer, sentences, source_layer):
    """Confirm that source-layer intervention actually changes activations."""
    # One sentence is enough for a wiring check; the real experiment uses all.
    sentence = sentences[0]

    clean = run_clean_pass(model, tokenizer, [sentence])
    damaged = run_forward_intervention(
        model, tokenizer, [sentence],
        source_layer,
        RANK_FRACTION,
        MIN_RANK,
    )

    diff = (
        clean[source_layer][0] -
        damaged[source_layer][0]
    ).abs().max().item()

    print(
        f"  sanity layer={source_layer}: "
        f"max source activation difference={diff:.6g}"
    )

    if diff < 1e-6:
        raise RuntimeError(
            f"Intervention at layer {source_layer} produced essentially "
            "zero change. Check rank fraction / hook wiring."
        )


def build_forward_matrix(model, tokenizer, sentences, metric_name):
    print(f"\n--- FORWARD: {metric_name} ---")

    clean_acts = run_clean_pass(model, tokenizer, sentences)
    clean = compute_metric_per_sentence(clean_acts, metric_name)

    matrix = np.full((N_LAYERS, N_LAYERS), np.nan)
    row_per_sentence = [None] * N_LAYERS
    row_ci95 = [None] * N_LAYERS

    for source_layer in range(N_LAYERS):
        print(f"\nSource layer {source_layer}")
        sanity_check(model, tokenizer, sentences, source_layer)

        damaged_acts = run_forward_intervention(
            model,
            tokenizer,
            sentences,
            source_layer,
            RANK_FRACTION,
            MIN_RANK,
        )
        damaged = compute_metric_per_sentence(
            damaged_acts, metric_name
        )

        # ONLY downstream cells: j > source_layer.
        downstream_layers = range(source_layer + 1, N_LAYERS)

        per_sentence_damage = []

        for target_layer in downstream_layers:
            damage = compute_damage(
                clean[target_layer],
                damaged[target_layer],
                metric_name,
            )

            matrix[source_layer, target_layer] = float(
                np.mean(damage)
            )
            per_sentence_damage.append(damage)

        if per_sentence_damage:
            per_sentence_damage = np.stack(
                per_sentence_damage, axis=1
            )  # [sentences, downstream layers]

            row_per_sentence[source_layer] = (
                per_sentence_damage.mean(axis=1)
            )

            row_ci95[source_layer] = bootstrap_mean_ci(
                row_per_sentence[source_layer]
            )

            print(
                f"  downstream-only damage = "
                f"{row_per_sentence[source_layer].mean():.5f} "
                f"CI={row_ci95[source_layer]}"
            )
        else:
            # Layer 5: no downstream layer exists.
            row_per_sentence[source_layer] = None
            row_ci95[source_layer] = None

    row_avg = [
        None if x is None else float(np.mean(x))
        for x in row_per_sentence
    ]

    return matrix, row_avg, row_ci95


def main():
    print("=" * 72)
    print("SST-2 FORWARD CAUSAL INTERVENTION")
    print("=" * 72)
    print(f"Device: checkpoint-loaded model")
    print(f"Rank fraction: {RANK_FRACTION}")
    print(f"Min rank: {MIN_RANK}")

    model, tokenizer = load_model_and_tokenizer()

    sentences, indices = load_probe_sentences(
        tokenizer,
        N_PROBE_SENTENCES,
    )
    save_probe_set(sentences, indices, PROBE_PATH)

    print(f"Saved shared probe set -> {PROBE_PATH}")

    for metric_name in METRIC_FUNCS:
        matrix, row_avg, row_ci95 = build_forward_matrix(
            model,
            tokenizer,
            sentences,
            metric_name,
        )

        result = {
            "task": TASK_NAME,
            "direction": "forward",
            "granularity": "layer",
            "metric": metric_name,
            "rank_fraction_used": RANK_FRACTION,
            "min_rank_used": MIN_RANK,
            "probe_split": "validation",
            "probe_seed": 42,
            "n_probe_sentences": len(sentences),
            "matrix": matrix.tolist(),
            "row_avg_damage_downstream_only": row_avg,
            "row_avg_bootstrap_ci95": row_ci95,
            "interpretation": (
                "matrix[i][j] is mean directional damage at downstream "
                "layer j caused by collapsing source layer i; j <= i "
                "is intentionally excluded from row summaries."
            ),
        }

        out_path = OUT_DIR / f"M_fwd_{TASK_NAME}_{metric_name}.json"
        out_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        print(f"\nSAVED: {out_path}")


if __name__ == "__main__":
    main()
