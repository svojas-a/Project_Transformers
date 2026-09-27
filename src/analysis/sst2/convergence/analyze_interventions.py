"""
SST-2 Forward/Backward Convergence Analysis
===========================================

Loads:
    forward/fwd_results/M_fwd_sst2_<metric>.json
    backward/bwd_results/M_bwd_sst2_<metric>.json

Produces:
    convergence_results/
        sst2_forward_backward_convergence.csv
        sst2_forward_backward_convergence.json
        convergence_<metric>.png

The goal is NOT to force a single "winner".
It identifies layers that are simultaneously prominent in:
    1. forward downstream damage
    2. backward downstream recovery

Layer 5 is undefined because it has no downstream layer.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FWD_DIR = ROOT / "forward" / "fwd_results"
BWD_DIR = ROOT / "backward" / "bwd_results"
OUT_DIR = ROOT / "convergence" / "convergence_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "effective_rank",
    "stable_rank",
    "mean_cosine_sim",
]


def load_metric(metric):
    fwd_path = FWD_DIR / f"M_fwd_sst2_{metric}.json"
    bwd_path = BWD_DIR / f"M_bwd_sst2_{metric}.json"

    if not fwd_path.exists():
        raise FileNotFoundError(f"Missing: {fwd_path}")

    if not bwd_path.exists():
        raise FileNotFoundError(f"Missing: {bwd_path}")

    fwd = json.loads(fwd_path.read_text())
    bwd = json.loads(bwd_path.read_text())

    return fwd, bwd


def percentile_rank(values):
    """
    Convert valid values into [0, 1] percentile ranks.
    Higher = more prominent.
    """
    values = np.asarray(values, dtype=float)
    result = np.full_like(values, np.nan)

    valid = np.isfinite(values)
    valid_values = values[valid]

    if len(valid_values) == 1:
        result[valid] = 1.0
        return result

    order = np.argsort(valid_values)
    ranks = np.empty(len(valid_values), dtype=float)
    ranks[order] = np.arange(len(valid_values))
    ranks /= len(valid_values) - 1

    result[valid] = ranks
    return result


def analyze_metric(metric):
    fwd, bwd = load_metric(metric)

    fwd_scores = np.asarray(
        fwd["row_avg_damage_downstream_only"],
        dtype=float,
    )
    bwd_scores = np.asarray(
        bwd["row_avg_recovery_downstream_only"],
        dtype=float,
    )

    fwd_rank = percentile_rank(fwd_scores)
    bwd_rank = percentile_rank(bwd_scores)

    # A simple convergence visualization quantity:
    # high on BOTH profiles -> high convergence.
    convergence = np.nanmean(
        np.vstack([fwd_rank, bwd_rank]),
        axis=0,
    )

    df = pd.DataFrame({
        "metric": metric,
        "layer": np.arange(len(fwd_scores)),
        "forward_damage": fwd_scores,
        "backward_recovery": bwd_scores,
        "forward_percentile": fwd_rank,
        "backward_percentile": bwd_rank,
        "convergence_index": convergence,
    })

    # Do not treat the undefined final layer as a candidate.
    df.loc[~np.isfinite(df["convergence_index"]), "convergence_index"] = np.nan

    # "Common top layers" means top-2 in BOTH profiles.
    valid = df.dropna(subset=["forward_percentile", "backward_percentile"])

    top_k = min(2, len(valid))

    top_forward = set(
        valid.nlargest(top_k, "forward_damage")["layer"].astype(int)
    )
    top_backward = set(
        valid.nlargest(top_k, "backward_recovery")["layer"].astype(int)
    )

    common_top_layers = sorted(top_forward & top_backward)

    # Highest convergence is only a candidate indicator, not proof of causality.
    candidate_layer = int(
        valid.loc[
            valid["convergence_index"].idxmax(),
            "layer",
        ]
    )

    print(f"\nMetric: {metric}")
    print(df.to_string(index=False))
    print(f"Top forward layers: {sorted(top_forward)}")
    print(f"Top backward layers: {sorted(top_backward)}")
    print(f"Common top layers: {common_top_layers}")
    print(f"Convergence candidate: layer {candidate_layer}")

    fig, ax1 = plt.subplots(figsize=(9, 5))

    x = df["layer"].to_numpy()

    ax1.plot(
        x,
        df["forward_damage"].to_numpy(),
        marker="o",
        label="Forward downstream damage",
    )
    ax1.plot(
        x,
        df["backward_recovery"].to_numpy(),
        marker="o",
        label="Backward downstream recovery",
    )

    ax1.set_xlabel("Source / restoration layer")
    ax1.set_ylabel("Raw effect magnitude")
    ax1.set_title(
        f"SST-2 forward damage vs backward recovery — {metric}"
    )
    ax1.set_xticks(x)
    ax1.grid(alpha=0.3)
    ax1.legend()

    fig.tight_layout()
    fig.savefig(
        OUT_DIR / f"convergence_{metric}.png",
        dpi=180,
    )
    plt.close(fig)

    return df, common_top_layers, candidate_layer


def main():
    all_tables = []
    summary = []

    for metric in METRICS:
        df, common, candidate = analyze_metric(metric)

        all_tables.append(df)
        summary.append({
            "metric": metric,
            "common_top_layers": common,
            "convergence_candidate": candidate,
        })

    combined = pd.concat(all_tables, ignore_index=True)

    combined.to_csv(
        OUT_DIR / "sst2_forward_backward_convergence.csv",
        index=False,
    )

    summary_path = OUT_DIR / "sst2_forward_backward_convergence.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("SAVED CONVERGENCE RESULTS")
    print("=" * 72)
    print(f"CSV:  {OUT_DIR / 'sst2_forward_backward_convergence.csv'}")
    print(f"JSON: {summary_path}")


if __name__ == "__main__":
    main()
