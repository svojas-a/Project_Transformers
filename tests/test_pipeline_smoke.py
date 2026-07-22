"""
Smoke test: runs the full ExperimentRunner with a synthetic MockModelWrapper
instead of a real DistilBERT download. Validates that the wiring (config ->
model -> extractor -> registry -> store) works end-to-end, independent of
network access or real weights. Real integration tests should additionally
run this against the actual DistilBertWrapper.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List

from config.config import ExperimentConfig, DimensionSchedule
from models.model_wrapper import ModelWrapper
from metrics.registry import build_default_registry
from data.dataset_loader import DatasetLoader
from pipeline.result_store import ResultStore
from pipeline.experiment_runner import ExperimentRunner


class MockModelWrapper(ModelWrapper):
    """Deterministic fake model: representation = random projection of a
    per-text hash, shrunk as hidden_dim shrinks, so metrics have something
    real to react to."""

    def __init__(self, base_dim=768, seed=0):
        self._base_dim = base_dim
        self._current_dim = base_dim
        self._rng = np.random.default_rng(seed)
        self._base_vectors = None

    def load(self) -> None:
        self._current_dim = self._base_dim

    def get_hidden_size(self) -> int:
        return self._current_dim

    def set_hidden_size(self, new_dim: int) -> None:
        self._current_dim = new_dim

    def extract_representations(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        # simulate collapse: as dim shrinks, representations get pushed
        # toward a shared low-rank subspace
        full = self._rng.standard_normal((n, self._base_dim))
        rank_cap = max(2, int(self._current_dim * 0.5))
        u, s, vt = np.linalg.svd(full, full_matrices=False)
        s[rank_cap:] *= 0.01  # crush variance outside the "surviving" subspace
        reduced_full = (u * s) @ vt
        return reduced_full[:, : self._current_dim]


class MockDatasetLoader(DatasetLoader):
    def load(self, task_name: str):
        texts = [f"sample sentence {i}" for i in range(120)]
        labels = [i % 3 for i in range(120)]
        return texts, labels


def test_pipeline_runs_end_to_end():
    config = ExperimentConfig(
        model_name="mock-model",
        seeds=[1, 2],
        tasks=["mock_task"],
        dimension_schedule=DimensionSchedule(base_dim=64, reduction_factor=0.75, num_steps=4),
        max_samples_for_metrics=100,
        output_dir="/tmp/phase1_smoke_test",
    )

    runner = ExperimentRunner(
        config=config,
        model_wrapper=MockModelWrapper(base_dim=64),
        metric_registry=build_default_registry(),
        dataset_loader=MockDatasetLoader(),
        result_store=ResultStore(output_dir=config.output_dir),
    )

    store = runner.run()
    df = store.to_dataframe()

    expected_dims = config.dimension_schedule.generate()
    assert set(df["hidden_dim"].unique()) == set(expected_dims)
    assert set(df["seed"].unique()) == {1, 2}
    assert "effective_rank" in df.columns
    assert "stable_rank" in df.columns
    assert df.shape[0] == len(expected_dims) * len(config.seeds)

    # sanity: effective rank should trend down as hidden_dim shrinks
    means = df.groupby("hidden_dim")["effective_rank"].mean().sort_index(ascending=False)
    print(means)
    assert means.iloc[0] >= means.iloc[-1] * 0.9  # loose monotonic-ish check

    print("Smoke test passed. Sample rows:")
    print(df.head())


if __name__ == "__main__":
    test_pipeline_runs_end_to_end()
