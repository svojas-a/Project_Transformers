"""
MetricRegistry: the single place that knows about every available metric.

This is the GRASP "Creator" for metric instances and the reason adding a
new metric never requires editing ExperimentRunner: register it here, and
it's automatically included in every run's computation loop
(Open/Closed Principle).
"""
from typing import Dict, List, Optional
import numpy as np

from .base import CollapseMetric
from .spectral_metrics import (
    EffectiveRankMetric,
    StableRankMetric,
    ConditionNumberMetric,
    SingularValueDecayMetric,
)
from .isotropy_metrics import AverageCosineSimilarityMetric, IsoScoreMetric
from .intrinsic_dimension import TwoNNIntrinsicDimensionMetric
from .neural_collapse import NC1Metric
from .alignment_metrics import LinearCKAMetric


class MetricRegistry:
    def __init__(self):
        self._metrics: Dict[str, CollapseMetric] = {}

    def register(self, metric: CollapseMetric) -> "MetricRegistry":
        self._metrics[metric.name] = metric
        return self  # chainable

    def unregister(self, name: str) -> None:
        self._metrics.pop(name, None)

    def get(self, name: str) -> CollapseMetric:
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not registered. "
                            f"Available: {list(self._metrics.keys())}")
        return self._metrics[name]

    def names(self) -> List[str]:
        return list(self._metrics.keys())

    def compute_all(
        self,
        representations: np.ndarray,
        labels: Optional[np.ndarray] = None,
        reference_representations: Optional[np.ndarray] = None,
        skip_on_error: bool = True,
    ) -> Dict[str, object]:
        """
        Runs every registered metric. Metrics that need labels/reference and
        don't get them will raise ValueError from inside their own
        `compute()` — caught here (if skip_on_error) so one missing
        prerequisite doesn't kill the whole run.
        """
        results = {}
        for name, metric in self._metrics.items():
            try:
                results[name] = metric.compute(
                    representations, labels=labels,
                    reference_representations=reference_representations,
                )
            except Exception as e:
                if skip_on_error:
                    results[name] = {"error": str(e)}
                else:
                    raise
        return results


def build_default_registry() -> MetricRegistry:
    """
    Convenience factory wiring up the six core Phase-1 metrics discussed in
    the research plan. Callers needing a different subset should construct
    their own MetricRegistry and register() only what they want — this
    function is a starting point, not the only way to build one.
    """
    registry = MetricRegistry()
    registry.register(EffectiveRankMetric())
    registry.register(StableRankMetric())
    registry.register(ConditionNumberMetric())
    registry.register(SingularValueDecayMetric())
    registry.register(AverageCosineSimilarityMetric())
    registry.register(IsoScoreMetric())
    registry.register(TwoNNIntrinsicDimensionMetric())
    registry.register(NC1Metric())
    registry.register(LinearCKAMetric())
    return registry
