"""
M4 - Step 3: Backward Restoration (Layer-Level) -> M_bwd
===========================================================

COMPATIBILITY NOTE (read this first)
--------------------------------------
This file imports directly from forward_causal_intervention.py (Person B's
script) rather than reimplementing the metrics, hooks, probe-sentence
loading, damage sign-convention, or bootstrap CI. That's deliberate: the
whole point of the shared contract in the M4 doc (Sec. 4) is that M_fwd and
M_bwd have to be produced under the IDENTICAL probe batch, rank_fraction,
min_rank, and metric definitions, or the Step 4 synthesis (comparing rows
across the two matrices) is comparing apples to oranges. Importing the real
functions guarantees zero drift instead of two copies quietly diverging.

    Put this file in the same directory as forward_causal_intervention.py
    before running it.

ONE DELIBERATE DEVIATION FROM THE DOC, MATCHING PERSON B'S ACTUAL CODE
-------------------------------------------------------------------------
The M4 doc describes Person C starting from "a fully compressed model,
d = 32". Person B's actual implementation does NOT collapse to a fixed
absolute rank -- it collapses each sentence's activation to a FRACTION of
its own min(seq_len, hidden_dim) (RANK_FRACTION), because a fixed rank like
32 silently no-ops on short SST-2 sentences (see B's module docstring,
"RANK IS RELATIVE, NOT ABSOLUTE"). "Fully compressed, d=32" is reinterpreted
here as: every layer's output collapsed to RANK_FRACTION via the same SVD
truncation B uses. This is what makes M_bwd comparable to M_fwd -- both use
the exact same corruption mechanism, just applied to different (all vs. one)
layers.

WHAT THIS DOES
---------------
For each candidate "healing" layer i in {0..5}:
    1. Reuse the CLEAN pass (per-sentence, per-layer activations) -- this is
       B's run_clean_pass(), called fresh here (or you can thread through a
       cached copy if you're calling both scripts from a shared driver).
    2. Build the FULLY COLLAPSED baseline: every layer's output rank-
       collapsed via B's LayerCollapser, none restored. This is the
       "no restoration at all" reference point recovery is measured against.
    3. Build the RESTORED pass: every layer EXCEPT layer i is still
       collapsed, but layer i's output is force-replaced with the precomputed
       CLEAN activation for that exact sentence (ROME-style single-point
       restoration) via the new LayerRestorer hook below. Everything
       downstream of i then runs on the "healed" activation.
    4. For every downstream layer j >= i, recovery[i][j] = how much of the
       fully-collapsed damage at j got undone by restoring i, using B's own
       compute_damage() so the sign convention is identical:
           recovery = damage(clean, fully_collapsed) - damage(clean, restored)
       Positive recovery = restoring layer i measurably un-does collapse at
       layer j. recovery[i][i] is the "self-recovery" and is always the full
       baseline damage at i (since layer i is patched to exactly the clean
       value), analogous to M_fwd's diagonal self-damage cell.
    5. Row averages + bootstrap 95% CI over probe sentences, same procedure
       as B (resample sentences, not matrix cells).

A sanity check (sanity_check_full_restoration) restores ALL 6 layers at
once and confirms this reproduces the clean pass -- both at the hidden-state
level and, for classification checkpoints, at the predicted-label level.
This is the "confirm the patching mechanism actually works before trusting
single-point results" check called for in the doc.

OUTPUT
------
One JSON file per metric, schema matching B's file field-for-field except
direction="backward" and damage-shaped fields renamed to recovery-shaped
ones (row_avg_recovery instead of row_avg_damage, etc.) per the doc's Sec.4
output-format table:
    {
        "task": ...,
        "direction": "backward",
        "granularity": "layer",
        "matrix": [[...6x6...]],
        "metric": ...,
        "rank_fraction_used": ...,
        "min_rank_used": ...,
        "row_avg_recovery": [...6 values...],
        "row_avg_bootstrap_ci95": [[lo, hi], ...6 pairs...],
        "per_sentence_row_avg_recovery": [[...per-sentence values...], ...6 rows...],
    }
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from forward_causal_intervention import (
    MODEL_CHECKPOINT_PATH,
    TASK_NAME,
    IS_CLASSIFICATION_CHECKPOINT,
    RANK_FRACTION,
    MIN_RANK,
    NUM_LAYERS,
    DEVICE,
    N_PROBE_SENTENCES,
    METRIC_FUNCS,
    load_probe_sentences,
    compute_damage,
    get_transformer_layers,
    LayerRecorder,
    LayerCollapser,
    run_clean_pass,
    compute_metric_per_sentence,
    bootstrap_row_ci,
)


# -----------------------------------------------------------------------
# RESTORATION HOOK - the ROME-style single-point patch
# -----------------------------------------------------------------------
class LayerRestorer:
    """
    Registers a forward hook on ONE layer that discards whatever that layer
    just computed (which may itself have run on an already-corrupted input,
    if upstream layers were collapsed) and force-replaces it with a
    precomputed CLEAN activation for this exact sentence.

    Mirrors LayerCollapser's constructor/remove() signature exactly, so the
    two are interchangeable per-layer "mutator" hooks in run_restored_pass()
    below -- swap collapse-this-layer for restore-this-layer without
    touching anything else.

    IMPORTANT: like B's collapser, this must be registered BEFORE any
    LayerRecorder on the same model, or the recorder will capture the
    pre-patch (module's own, uncorrected) value instead of the restored one.
    """

    def __init__(self, model, layer_idx, clean_tensor):
        self.handle = None
        self.clean_tensor = clean_tensor  # [seq_len, hidden], already on DEVICE
        layer = get_transformer_layers(model)[layer_idx]
        self.handle = layer.register_forward_hook(self._patch_hook)

    def _patch_hook(self, module, input, output):
        patched = self.clean_tensor.unsqueeze(0)
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    def remove(self):
        if self.handle is not None:
            self.handle.remove()


# -----------------------------------------------------------------------
# MAIN EXPERIMENT LOOP
# -----------------------------------------------------------------------
def run_restored_pass(model, tokenizer, sentences, clean_acts, patch_layers,
                       rank_fraction, min_rank):
    """
    General-purpose pass: every layer in `patch_layers` gets restored to its
    clean value; every other layer gets rank-collapsed via B's LayerCollapser.

    patch_layers=set()                    -> fully collapsed baseline
    patch_layers={i}                      -> single-point restoration at i
    patch_layers=set(range(NUM_LAYERS))   -> full restoration (sanity check)

    Returns dict: layer_idx -> list of per-sentence [seq_len, hidden] output
    activations, same shape/format as B's run_clean_pass()/run_intervened_pass(),
    so compute_metric_per_sentence() works unchanged on the result.
    """
    per_sentence_acts = {i: [] for i in range(NUM_LAYERS)}

    for sent_idx, sent in enumerate(sentences):
        inputs = tokenizer(sent, return_tensors="pt").to(DEVICE)

        # Mutators (collapse or restore) must be registered BEFORE the
        # recorder -- see LayerRestorer / B's module docstring for why.
        mutators = []
        for layer_idx in range(NUM_LAYERS):
            if layer_idx in patch_layers:
                clean_tensor = clean_acts[layer_idx][sent_idx]
                mutators.append(LayerRestorer(model, layer_idx, clean_tensor))
            else:
                mutators.append(LayerCollapser(model, layer_idx, rank_fraction, min_rank))

        recorder = LayerRecorder(model)
        with torch.no_grad():
            model(**inputs)
        for layer_idx, act in recorder.activations.items():
            per_sentence_acts[layer_idx].append(act[0])

        recorder.remove()
        for m in mutators:
            m.remove()

    return per_sentence_acts


def sanity_check_full_restoration(model, tokenizer, sentences):
    """
    Restores ALL 6 layers at once (patch_layers = every layer) and checks:
      (a) every layer's hidden-state output exactly matches the clean pass.
          This is guaranteed by construction (LayerRestorer just returns the
          clean tensor, ignoring the module's real output) -- so a nonzero
          diff here doesn't mean the restoration "half-worked", it means a
          hook leaked across sentences, a shape mismatched, or a layer index
          is wrong. Treat any diff > ~1e-5 as a wiring bug, not a research
          finding.
      (b) for classification checkpoints, the predicted label under full
          restoration matches the clean model's predicted label on every
          probe sentence -- this is the actual "recovers clean-model
          performance" check the doc's Person-C deliverable calls for.
    """
    print(f"Running clean + fully-restored passes on {len(sentences)} sentences "
          f"for the sanity check...")
    clean_acts = run_clean_pass(model, tokenizer, sentences)
    restored_acts = run_restored_pass(
        model, tokenizer, sentences, clean_acts,
        patch_layers=set(range(NUM_LAYERS)),
        rank_fraction=RANK_FRACTION, min_rank=MIN_RANK,
    )

    max_diff = 0.0
    for layer_idx in range(NUM_LAYERS):
        for clean_t, restored_t in zip(clean_acts[layer_idx], restored_acts[layer_idx]):
            d = (clean_t - restored_t).abs().max().item()
            max_diff = max(max_diff, d)
    print(f"  Max abs diff, clean vs. fully-restored hidden states "
          f"(all layers/sentences): {max_diff:.8f}")
    if max_diff > 1e-5:
        print("  WARNING: fully-restored pass does not reproduce the clean pass -- "
              "check LayerRestorer wiring / hook registration order before "
              "trusting M_bwd.")
    else:
        print("  OK: patching mechanism reproduces the clean pass exactly.")

    if IS_CLASSIFICATION_CHECKPOINT:
        n_match = 0
        for sent_idx, sent in enumerate(sentences):
            inputs = tokenizer(sent, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                clean_pred = model(**inputs).logits.argmax(-1).item()

            mutators = [
                LayerRestorer(model, layer_idx, clean_acts[layer_idx][sent_idx])
                for layer_idx in range(NUM_LAYERS)
            ]
            with torch.no_grad():
                restored_pred = model(**inputs).logits.argmax(-1).item()
            for m in mutators:
                m.remove()

            n_match += int(clean_pred == restored_pred)

        pct = 100 * n_match / len(sentences)
        print(f"  Classification agreement (fully-restored vs. clean): "
              f"{n_match}/{len(sentences)} = {pct:.1f}%")
        if pct < 100.0:
            print("  WARNING: predictions should match exactly under full "
                  "restoration (every layer is forced to its clean value) -- "
                  "investigate before trusting single-point recovery numbers.")


def build_backward_matrix(model, tokenizer, sentences, metric_name):
    metric_fn = METRIC_FUNCS[metric_name]

    print(f"\n[1/3] CLEAN baseline pass ({metric_name})...")
    clean_acts = run_clean_pass(model, tokenizer, sentences)
    clean_per_sentence = compute_metric_per_sentence(clean_acts, metric_fn)
    clean_scores = {i: float(v.mean()) for i, v in clean_per_sentence.items()}
    print(f"      Clean per-layer {metric_name}: {clean_scores}")

    print(f"\n[2/3] FULLY COLLAPSED baseline pass "
          f"(all layers, rank_fraction={RANK_FRACTION})...")
    fully_collapsed_acts = run_restored_pass(
        model, tokenizer, sentences, clean_acts,
        patch_layers=set(), rank_fraction=RANK_FRACTION, min_rank=MIN_RANK,
    )
    fc_per_sentence = compute_metric_per_sentence(fully_collapsed_acts, metric_fn)
    fc_scores = {i: float(v.mean()) for i, v in fc_per_sentence.items()}
    print(f"      Fully-collapsed per-layer {metric_name}: {fc_scores}")

    M_bwd = np.zeros((NUM_LAYERS, NUM_LAYERS))
    per_sentence_row_avg = [None] * NUM_LAYERS
    row_ci95 = [None] * NUM_LAYERS

    for source_layer in range(NUM_LAYERS):
        print(f"\n[3/3] Restoring layer {source_layer} "
              f"(rank_fraction={RANK_FRACTION} elsewhere)...")

        patched_acts = run_restored_pass(
            model, tokenizer, sentences, clean_acts,
            patch_layers={source_layer}, rank_fraction=RANK_FRACTION, min_rank=MIN_RANK,
        )
        patched_per_sentence = compute_metric_per_sentence(patched_acts, metric_fn)

        # Only j >= source_layer is causally meaningful: layers upstream of
        # the patch point were already collapsed before layer i ran and
        # can't be affected by it (same convention as M_fwd's j >= i).
        valid_js = list(range(source_layer, NUM_LAYERS))
        per_sentence_recovery_cols = np.stack([
            compute_damage(clean_per_sentence[j], fc_per_sentence[j], metric_name)
            - compute_damage(clean_per_sentence[j], patched_per_sentence[j], metric_name)
            for j in valid_js
        ], axis=1)  # [n_sentences, n_valid_cols]

        for col_idx, j in enumerate(valid_js):
            M_bwd[source_layer][j] = per_sentence_recovery_cols[:, col_idx].mean()
        # j < source_layer stays 0.0 -- not causally reachable by this patch.

        # Healing-site scoring must use DOWNSTREAM-ONLY cells (j > source_layer),
        # excluding the diagonal M_bwd[i][i]. The diagonal is a tautology: layer
        # i is force-patched to its own clean value, so "recovery" there is just
        # the full baseline damage at i, not evidence that restoring i heals
        # anything else. Including it inflates row averages and, since the
        # diagonal cell contributes a larger share of the row as source_layer
        # increases (fewer downstream cells remain), made row averages
        # spuriously increase with layer index.
        downstream_js = [j for j in valid_js if j > source_layer]
        if downstream_js:
            # valid_js starts at source_layer, so column 0 is always the
            # diagonal cell -- drop it to get downstream-only columns.
            downstream_cols = per_sentence_recovery_cols[:, 1:]
            per_sentence_row_avg[source_layer] = downstream_cols.mean(axis=1)
            row_ci95[source_layer] = bootstrap_row_ci(per_sentence_row_avg[source_layer])

            row_avg = per_sentence_row_avg[source_layer].mean()
            lo, hi = row_ci95[source_layer]
            print(f"      Row {source_layer} downstream-only avg recovery "
                  f"(excl. diagonal M_bwd[{source_layer}][{source_layer}]): "
                  f"{row_avg:.4f} [95% CI: {lo:.4f}, {hi:.4f}]  "
                  f"(per-layer: {[f'{v:.3f}' for v in M_bwd[source_layer]]})")
        else:
            # The last layer has no downstream layers, so its only recovery
            # cell is the diagonal self-recovery -- a tautology, not a
            # healing-site score. Mark it undefined rather than silently
            # falling back to self-recovery, which is what let this layer
            # spuriously rank as the most potent healing site.
            per_sentence_row_avg[source_layer] = None
            row_ci95[source_layer] = (float("nan"), float("nan"))
            print(f"      Row {source_layer}: no downstream layers exist -- "
                  f"downstream-only healing score is undefined (NaN); the "
                  f"diagonal self-recovery cell M_bwd[{source_layer}][{source_layer}] "
                  f"= {M_bwd[source_layer][source_layer]:.4f} is excluded from ranking "
                  f"(per-layer: {[f'{v:.3f}' for v in M_bwd[source_layer]]})")

    return M_bwd, clean_scores, per_sentence_row_avg, row_ci95


def compare_top_recovery_layers(row_avgs, row_ci95, metric_name):
    """
    Same logic as B's compare_top_layers, reworded for recovery instead of
    damage: flags whether the top healing-site candidate is actually
    distinguishable from the runner-up, or just noise (overlapping 95% CIs).
    """
    # Rank only layers with a defined downstream-only score; a layer with no
    # downstream cells (NaN) has no healing-site score to compare.
    valid_layers = [i for i in range(len(row_avgs)) if not np.isnan(row_avgs[i])]
    order = sorted(valid_layers, key=lambda i: row_avgs[i], reverse=True)
    top1, top2 = order[0], order[1]
    lo1, hi1 = row_ci95[top1]
    lo2, hi2 = row_ci95[top2]
    overlap = not (hi2 < lo1 or hi1 < lo2)
    gap_pct = 100 * (row_avgs[top1] - row_avgs[top2]) / max(abs(row_avgs[top1]), 1e-12)

    print(f"\n[significance check - {metric_name}] "
          f"Top healing site: Layer {top1} ({row_avgs[top1]:.4f}, CI [{lo1:.4f},{hi1:.4f}])  "
          f"Runner-up: Layer {top2} ({row_avgs[top2]:.4f}, CI [{lo2:.4f},{hi2:.4f}])  "
          f"gap={gap_pct:.1f}%")
    if overlap:
        print(f"[significance check - {metric_name}] CIs OVERLAP - "
              f"Layer {top1} being 'the healing site' is NOT statistically "
              f"distinguishable from Layer {top2} on this probe batch.")
    else:
        print(f"[significance check - {metric_name}] CIs do not overlap - "
              f"Layer {top1} is distinguishably more restorative than Layer {top2}.")
    return top1, top2, overlap


def maybe_cross_check_with_forward(task_name, metric_name, top_recovery_layer):
    """
    Joint Step 4 cross-check (per the doc): the row in M_fwd with the
    highest average DAMAGE should, ideally, be the same layer as the row in
    M_bwd with the highest average RECOVERY. If Person B has already run
    and saved M_fwd_{task}_{metric}.json in this directory, compare them and
    say so -- a mismatch is "a genuinely interesting finding worth flagging
    to the group, not a bug to quietly resolve" (doc, Person C task 5).
    """
    fwd_path = f"M_fwd_{task_name}_{metric_name}.json"
    try:
        with open(fwd_path) as f:
            fwd_result = json.load(f)
    except FileNotFoundError:
        print(f"\n[cross-check] {fwd_path} not found yet -- run "
              f"forward_causal_intervention.py to enable the M_fwd vs. M_bwd "
              f"top-layer cross-check.")
        return None

    fwd_row_avgs = np.array(fwd_result["row_avg_damage"])
    top_damage_layer = int(np.argmax(fwd_row_avgs))
    print(f"\n[cross-check - {metric_name}] Highest-damage source layer (M_fwd): "
          f"Layer {top_damage_layer}  |  Highest-recovery healing site (M_bwd): "
          f"Layer {top_recovery_layer}")
    if top_damage_layer == top_recovery_layer:
        print(f"[cross-check - {metric_name}] MATCH -- consistent with a single "
              f"Critical Source Layer.")
    else:
        print(f"[cross-check - {metric_name}] MISMATCH -- the layer that does the "
              f"most damage forward isn't the layer that recovers the most "
              f"restored backward. Flag this to the group per the doc.")
    return top_damage_layer == top_recovery_layer


def main():
    print(f"Loading model from: {MODEL_CHECKPOINT_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_PATH)
    if IS_CLASSIFICATION_CHECKPOINT:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT_PATH).to(DEVICE)
    else:
        model = AutoModel.from_pretrained(MODEL_CHECKPOINT_PATH).to(DEVICE)
    model.eval()
    print(f"Layers found: {len(get_transformer_layers(model))} (should be 6 for DistilBERT)")

    # Same probe sentences B uses -- SAME split, seed, count, so M_fwd and
    # M_bwd are evaluated on the identical sentence set.
    probe_sentences = load_probe_sentences(tokenizer, N_PROBE_SENTENCES)

    print(f"\n{'='*70}\nSANITY CHECK: full restoration should reproduce the clean pass\n{'='*70}")
    sanity_check_full_restoration(model, tokenizer, probe_sentences[:10])

    for metric_name in METRIC_FUNCS:
        print(f"\n{'='*70}\nMETRIC: {metric_name}\n{'='*70}")
        M_bwd, clean_scores, per_sentence_row_avg, row_ci95 = build_backward_matrix(
            model, tokenizer, probe_sentences, metric_name
        )

        # NaN marks layers with no downstream cells (currently just the last
        # layer) -- their diagonal self-recovery is a tautology, not a
        # healing-site score, so it must not be treated as a real value.
        row_avgs = np.array([
            v.mean() if v is not None else float("nan")
            for v in per_sentence_row_avg
        ])

        result = {
            "task": TASK_NAME,
            "direction": "backward",
            "granularity": "layer",
            "matrix": M_bwd.tolist(),
            "metric": metric_name,
            "rank_fraction_used": RANK_FRACTION,
            "min_rank_used": MIN_RANK,
            "row_avg_recovery": row_avgs.tolist(),
            "row_avg_bootstrap_ci95": [list(ci) for ci in row_ci95],
            "per_sentence_row_avg_recovery": [
                v.tolist() if v is not None else None for v in per_sentence_row_avg
            ],
        }

        out_path = f"M_bwd_{TASK_NAME}_{metric_name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved -> {out_path}")

        # nanargmax: pick the top layer among DEFINED downstream-only scores
        # only, so the undefined (NaN) last-layer row can never win by
        # default the way its inflated self-recovery row average used to.
        top_layer = int(np.nanargmax(row_avgs))
        print(f"Most potent healing site for {metric_name}: "
              f"Layer {top_layer} (downstream-only avg recovery = {row_avgs[top_layer]:.4f})")
        print(f"All row averages (downstream-only, NaN = undefined): "
              f"{[f'{v:.3f}' for v in row_avgs]}")

        compare_top_recovery_layers(row_avgs, row_ci95, metric_name)
        maybe_cross_check_with_forward(TASK_NAME, metric_name, top_layer)


if __name__ == "__main__":
    main()