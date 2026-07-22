"""
ExperimentRunner: the GRASP "Controller" for this pipeline.

It coordinates collaborators (ModelWrapper, RepresentationExtractor,
MetricRegistry, DatasetLoader, ResultStore, TaskPerformanceEvaluator) but
performs no computation itself -- every actual algorithm lives in its own
class. This keeps the runner readable as pure orchestration logic and means
unit tests for metrics/reduction never need to spin up a full experiment.
"""
from typing import List, Optional
import numpy as np

from config.config import ExperimentConfig
from models.model_wrapper import ModelWrapper
from extraction.representation_extractor import RepresentationExtractor
from metrics.registry import MetricRegistry
from data.dataset_loader import DatasetLoader
from pipeline.result_store import ResultStore
from evaluation.task_performance import TaskPerformanceEvaluator


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        model_wrapper: ModelWrapper,
        metric_registry: MetricRegistry,
        dataset_loader: DatasetLoader,
        result_store: ResultStore,
        task_performance_evaluator: Optional[TaskPerformanceEvaluator] = None,
    ):
        self._config = config
        self._model = model_wrapper
        self._registry = metric_registry
        self._loader = dataset_loader
        self._store = result_store
        # BUGFIX: this now operates on (representations, labels) -- the
        # SAME representations already extracted for the collapse metrics --
        # instead of re-running the model on raw text. Previously this was
        # never wired up from main.py at all, so task_performance was
        # always null; it's also now cheaper (no duplicate forward pass)
        # and guaranteed to be evaluated on the exact same samples as the
        # metrics for that (task, seed, dim).
        self._task_performance_evaluator = task_performance_evaluator

    def run(self) -> ResultStore:
        dims = self._config.dimension_schedule.generate()

        for task in self._config.tasks:
            texts, labels = self._loader.load(task)

            for seed in self._config.seeds:
                self._run_single_seed(task, texts, labels, seed, dims)

        self._store.save()
        return self._store

    def _run_single_seed(
        self, task: str, texts: List[str], labels, seed: int, dims: List[int]
    ) -> None:
        np.random.seed(seed)

        self._model.load()
        # A fresh extractor per seed: it draws its subsample ONCE (on first
        # extract() call below) and reuses that same fixed subsample for
        # every dimension in this seed's sweep. See
        # RepresentationExtractor's docstring for why this matters both for
        # CKA correctness and for genuine seed-to-seed variance.
        extractor = RepresentationExtractor(
            self._model, max_samples=self._config.max_samples_for_metrics, seed=seed
        )

        # full-size representations act as the CKA reference for this seed/task
        reference_reps, reference_labels = extractor.extract(texts, labels)

        for dim in dims:
            if dim != self._model.get_hidden_size():
                self._model.set_hidden_size(dim)

            reps, dim_labels = extractor.extract(texts, labels)

            metrics_result = self._registry.compute_all(
                reps,
                labels=dim_labels,
                reference_representations=reference_reps
                if reference_reps.shape[0] == reps.shape[0]
                else None,
            )

            perf = None
            if self._task_performance_evaluator is not None:
                perf = self._task_performance_evaluator.evaluate(reps, dim_labels)

            self._store.add_record(
                task=task, seed=seed, hidden_dim=dim,
                metrics=metrics_result, task_performance=perf,
            )