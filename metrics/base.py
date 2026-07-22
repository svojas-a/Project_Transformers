"""
Base contract for collapse metrics.

Every metric takes a representation matrix (and optionally labels, or a
reference matrix for alignment metrics) and returns a float or a small dict
of floats. Keeping this contract narrow means the MetricRegistry and
ExperimentRunner can treat every metric identically regardless of its
internal math (Liskov Substitution + Interface Segregation).
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Union
import numpy as np


MetricResult = Union[float, Dict[str, float]]


class CollapseMetric(ABC):
    """All collapse metrics implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
        representations: np.ndarray,
        labels: Optional[np.ndarray] = None,
        reference_representations: Optional[np.ndarray] = None,
    ) -> MetricResult:
        """
        representations: (n_samples, hidden_dim) matrix at the CURRENT dim.
        labels: optional, required only by label-aware metrics (e.g. NC1).
        reference_representations: optional, required only by alignment
            metrics (e.g. CKA against the full/unreduced model).
        Implementations that don't need labels/reference should ignore them.
        """
        raise NotImplementedError
