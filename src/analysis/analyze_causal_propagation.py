"""
Causal Propagation Analysis
============================

Reads results/causal_effects.csv (produced by causal_propagation.py) and
answers the three questions this experiment was designed for:

  1. How does the collapse effect decay with distance from the source layer?
  2. Is layer 3 a disproportionate source of downstream collapse, compared
     to the other layers -- controlling for the fact that layer 0 simply
     has more downstream layers to affect than layer 4 does?
  3. Does the pattern hold across metrics, ranks, and tasks, or is it
     specific to one of them?

Important design point
-----------------------
source_layer 0 has 5 downstream layers (distance 1..5), but source_layer 4
only has 1 (distance 1 only). A naive "sum of abs_delta across all
downstream layers" comparison would make layer 0 look artificially more
"causal" just because it has more layers to sum over. Every analysis below
either (a) compares at a fixed, matched distance, or (b) averages per
downstream layer rather than summing, so source layers are compared fairly.

Outputs (written to results/analysis/):
  - decay_curves_<metric>.png       one figure per metric: abs_delta vs
                                     distance, one line per source layer
  - layer3_test_summary.csv         regression-based significance test of
                                     whether source_layer==3 has an
                                     elevated effect, per metric
  - mean_effect_per_downstream_layer.csv   normalized comparison table
  - composite_propagation_score.csv        cross-metric combined ranking
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

INPUT_CSV = "results/causal_effects.csv"
OUTPUT_DIR = Path("results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["effective_rank", "stable_rank", "mean_pairwise_cosine", "attention_entropy"]


def load_data(path: str = INPUT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# --------------------------------------------------------------------------
# 1. Decay curves: effect size vs distance, one line per source layer
# --------------------------------------------------------------------------


def plot_decay_curves(df: pd.DataFrame):
    """For each metric, plot mean abs_delta vs distance, one line per
    source_layer, averaged across task and target_rank. Separate panels
    also shown for a mild rank (16) and a severe rank (1) so you can see
    whether the pattern only shows up at one collapse intensity."""

    for metric in METRICS:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
        subsets = {
            "All ranks (averaged)": df,
            "Severe collapse (rank=1)": df[df["target_rank"] == 1],
            "Mild collapse (rank=16)": df[df["target_rank"] == 16],
        }
        for ax, (title, sub) in zip(axes, subsets.items()):
            sub_m = sub[sub["metric"] == metric]
            for source_layer in sorted(sub_m["source_layer"].unique()):
                line = (
                    sub_m[sub_m["source_layer"] == source_layer]
                    .groupby("distance")["abs_delta"]
                    .mean()
                    .sort_index()
                )
                ax.plot(
                    line.index,
                    line.values,
                    marker="o",
                    label=f"source layer {source_layer}",
                )
            ax.set_xlabel("distance (downstream layers from source)")
            ax.set_title(title)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(f"mean |delta| ({metric})")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Downstream propagation decay -- {metric}")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"decay_curves_{metric}.png", dpi=150)
        plt.close(fig)
        print(f"saved decay_curves_{metric}.png")


# --------------------------------------------------------------------------
# 2. Layer-3 hypothesis test (matched-distance, regression-controlled)
# --------------------------------------------------------------------------


def layer3_significance_test(df: pd.DataFrame) -> pd.DataFrame:
    """For each metric, restrict to distance == 1 (the one downstream
    layer every source_layer 0-4 has in common -- a fair comparison point),
    then fit:
        abs_delta ~ C(source_layer) + C(task) + target_rank
    The coefficient on C(source_layer)[T.3] tells you whether layer 3's
    immediate downstream effect is significantly different from the
    reference layer, after controlling for task and rank.
    """
    rows = []
    for metric in METRICS:
        sub = df[(df["metric"] == metric) & (df["distance"] == 1)].copy()
        sub["source_layer"] = sub["source_layer"].astype(str)
        model = smf.ols(
            "abs_delta ~ C(source_layer, Treatment(reference='0')) + C(task) + target_rank",
            data=sub,
        ).fit()

        for layer in ["1", "2", "3", "4"]:
            param_name = f"C(source_layer, Treatment(reference='0'))[T.{layer}]"
            if param_name in model.params.index:
                rows.append(
                    {
                        "metric": metric,
                        "source_layer": layer,
                        "coef_vs_layer0": model.params[param_name],
                        "p_value": model.pvalues[param_name],
                        "significant_at_0.05": model.pvalues[param_name] < 0.05,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "layer3_test_summary.csv", index=False)
    print("saved layer3_test_summary.csv")
    return result


# --------------------------------------------------------------------------
# 3. Fair cross-layer comparison: mean effect PER downstream layer
#    (not summed, so layer 0 with 5 downstream layers isn't unfairly
#    favored over layer 4 with only 1)
# --------------------------------------------------------------------------


def mean_effect_per_downstream_layer(df: pd.DataFrame) -> pd.DataFrame:
    result = (
        df.groupby(["metric", "source_layer"])["abs_delta"]
        .mean()
        .reset_index()
        .rename(columns={"abs_delta": "mean_abs_delta_per_downstream_layer"})
    )
    pivot = result.pivot(
        index="source_layer",
        columns="metric",
        values="mean_abs_delta_per_downstream_layer",
    )
    pivot.to_csv(OUTPUT_DIR / "mean_effect_per_downstream_layer.csv")
    print("saved mean_effect_per_downstream_layer.csv")
    return pivot


# --------------------------------------------------------------------------
# 4. Composite propagation score (z-normalized across metrics so effective
#    rank, cosine similarity etc. -- which live on very different scales --
#    can be combined into one comparable number per source layer)
# --------------------------------------------------------------------------


def composite_propagation_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["z_abs_delta"] = df.groupby("metric")["abs_delta"].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    composite = (
        df.groupby("source_layer")["z_abs_delta"]
        .mean()
        .reset_index()
        .rename(columns={"z_abs_delta": "composite_propagation_score"})
        .sort_values("composite_propagation_score", ascending=False)
    )
    composite.to_csv(OUTPUT_DIR / "composite_propagation_score.csv", index=False)
    print("saved composite_propagation_score.csv")
    return composite


# --------------------------------------------------------------------------
# Run everything
# --------------------------------------------------------------------------


def run_analysis(input_csv: str = INPUT_CSV):
    df = load_data(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}\n")

    plot_decay_curves(df)
    print()

    layer3_results = layer3_significance_test(df)
    print(
        "\nLayer-3 significance test (positive coef = layer 3 causes MORE\n"
        "downstream effect than layer 0 at the same distance, controlling\n"
        "for task and rank):"
    )
    print(layer3_results[layer3_results["source_layer"] == "3"].to_string(index=False))
    print()

    per_layer = mean_effect_per_downstream_layer(df)
    print(
        "Mean effect per downstream layer (fair comparison, normalized\nby number of downstream layers each source layer has):"
    )
    print(per_layer.to_string())
    print()

    composite = composite_propagation_score(df)
    print(
        "Composite propagation score (higher = more causally disruptive\nsource layer, averaged across all 4 metrics, z-normalized):"
    )
    print(composite.to_string(index=False))

    return {
        "layer3_test": layer3_results,
        "per_layer": per_layer,
        "composite": composite,
    }


if __name__ == "__main__":
    run_analysis()
