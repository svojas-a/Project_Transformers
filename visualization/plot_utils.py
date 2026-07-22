"""
Plotting utilities. Deliberately dumb: takes a tidy DataFrame (from
ResultStore.to_dataframe) and draws it. No knowledge of how results were
computed -- keeps visualization decoupled from the pipeline (Single
Responsibility).
"""
from typing import List, Optional
import os


def plot_metric_vs_dimension(
    df, metric_column: str, output_path: str, task: Optional[str] = None
) -> str:
    import matplotlib.pyplot as plt
    import pandas as pd

    plot_df = df if task is None else df[df["task"] == task]
    if metric_column not in plot_df.columns:
        raise ValueError(f"'{metric_column}' not found in results. "
                          f"Available: {list(plot_df.columns)}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for t, group in plot_df.groupby("task"):
        agg = group.groupby("hidden_dim")[metric_column].agg(["mean", "std"]).reset_index()
        agg = agg.sort_values("hidden_dim", ascending=False)
        ax.errorbar(
            agg["hidden_dim"], agg["mean"], yerr=agg["std"],
            marker="o", capsize=3, label=t,
        )

    ax.set_xlabel("Hidden dimension")
    ax.set_ylabel(metric_column)
    ax.set_title(f"{metric_column} vs. hidden dimension")
    ax.invert_xaxis()  # left = full size, right = most reduced
    ax.legend()
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_all_metrics(df, metric_columns: List[str], output_dir: str) -> List[str]:
    paths = []
    for col in metric_columns:
        path = os.path.join(output_dir, f"{col}.png")
        try:
            paths.append(plot_metric_vs_dimension(df, col, path))
        except Exception as e:
            print(f"Skipping plot for {col}: {e}")
    return paths
