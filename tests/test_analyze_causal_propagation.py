"""
Tests for src/analysis/analyze_causal_propagation.py

Uses a small synthetic causal_effects-style DataFrame rather than real
experiment output, so these tests are fast and don't depend on having run
the full sweep first.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.analyze_causal_propagation import (
    composite_propagation_score,
    layer3_significance_test,
    mean_effect_per_downstream_layer,
)


@pytest.fixture
def synthetic_effects_df():
    """A small but structurally realistic causal_effects table: 3 tasks,
    5 source layers (0-4) each with a distance=1 row (mirroring the real
    data, where source_layer 4 only ever reaches distance=1), a couple of
    ranks, and all 4 metrics. Deliberately gives source_layer 0 a bigger
    effect than the rest so the significance test has something real to
    detect.
    """
    rows = []
    rng = np.random.default_rng(42)
    tasks = ["sst2", "mnli", "conll2003"]
    metrics = [
        "effective_rank",
        "stable_rank",
        "mean_pairwise_cosine",
        "attention_entropy",
    ]
    ranks = [1, 8]

    for task in tasks:
        for source_layer in range(5):
            max_distance = 5 - source_layer
            for distance in range(1, max_distance + 1):
                for rank in ranks:
                    for metric in metrics:
                        # source_layer 0 gets a deliberately larger effect
                        base_effect = 10.0 if source_layer == 0 else 2.0
                        noise = rng.normal(0, 0.1)
                        clean_value = 20.0
                        intervened_value = clean_value - base_effect + noise
                        delta = intervened_value - clean_value
                        rows.append(
                            {
                                "task": task,
                                "source_layer": source_layer,
                                "target_rank": rank,
                                "downstream_layer": source_layer + distance,
                                "distance": distance,
                                "metric": metric,
                                "clean_value": clean_value,
                                "intervened_value": intervened_value,
                                "delta": delta,
                                "abs_delta": abs(delta),
                            }
                        )
    return pd.DataFrame(rows)


class TestLayer3SignificanceTest:
    def test_returns_expected_columns(self, synthetic_effects_df):
        result = layer3_significance_test(synthetic_effects_df)
        assert set(result.columns) == {
            "metric",
            "source_layer",
            "coef_vs_layer0",
            "p_value",
            "significant_at_0.05",
        }

    def test_detects_layer0_as_stronger_source(self, synthetic_effects_df):
        # Since source_layer 0 was constructed with a much larger effect,
        # every other layer's coefficient relative to layer 0 should be
        # negative (i.e. weaker effect than layer 0).
        result = layer3_significance_test(synthetic_effects_df)
        for _, row in result.iterrows():
            assert row["coef_vs_layer0"] < 0

    def test_all_metrics_present(self, synthetic_effects_df):
        result = layer3_significance_test(synthetic_effects_df)
        assert set(result["metric"].unique()) == {
            "effective_rank",
            "stable_rank",
            "mean_pairwise_cosine",
            "attention_entropy",
        }


class TestMeanEffectPerDownstreamLayer:
    def test_returns_one_row_per_source_layer(self, synthetic_effects_df):
        result = mean_effect_per_downstream_layer(synthetic_effects_df)
        assert sorted(result.index.tolist()) == [0, 1, 2, 3, 4]

    def test_layer0_has_highest_mean_effect(self, synthetic_effects_df):
        result = mean_effect_per_downstream_layer(synthetic_effects_df)
        for metric in result.columns:
            assert result[metric].idxmax() == 0


class TestCompositePropagationScore:
    def test_returns_one_row_per_source_layer(self, synthetic_effects_df):
        result = composite_propagation_score(synthetic_effects_df)
        assert sorted(result["source_layer"].unique().tolist()) == [0, 1, 2, 3, 4]

    def test_layer0_ranked_highest(self, synthetic_effects_df):
        result = composite_propagation_score(synthetic_effects_df)
        top_layer = result.iloc[0]["source_layer"]
        assert top_layer == 0

    def test_sorted_descending(self, synthetic_effects_df):
        result = composite_propagation_score(synthetic_effects_df)
        scores = result["composite_propagation_score"].tolist()
        assert scores == sorted(scores, reverse=True)
