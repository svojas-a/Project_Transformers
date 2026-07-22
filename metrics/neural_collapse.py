"""
Neural Collapse metrics.

Reference:
  Papyan, Han, Donoho (2020), "Prevalence of Neural Collapse during the
  Terminal Phase of Deep Learning Training" -> defines NC1-NC4. We implement
  NC1, the most broadly applicable one for representation analysis: it
  requires labels but not a trained classifier head.

NC1 = tr(Sigma_W) / tr(Sigma_B)
  Sigma_W: within-class covariance (average scatter of samples around their
           own class mean)
  Sigma_B: between-class covariance (scatter of class means around the
           global mean)
  NC1 -> 0 indicates classes are collapsing to single points (their means),
  i.e. within-class variability vanishing relative to between-class spread.
"""
from typing import Optional
import numpy as np
from .base import CollapseMetric, MetricResult


class NC1Metric(CollapseMetric):
    """
    Requires `labels` (integer class ids). Metrics not needing labels can
    ignore the argument, but this one raises clearly if it's missing rather
    than silently returning a meaningless number.
    """

    @property
    def name(self) -> str:
        return "nc1_within_between_ratio"

    def compute(
        self,
        representations: np.ndarray,
        labels: Optional[np.ndarray] = None,
        reference_representations: Optional[np.ndarray] = None,
    ) -> MetricResult:
        if labels is None:
            raise ValueError(
                f"{self.name} requires class labels; none were provided. "
                "Skip this metric for unlabeled/regression tasks."
            )

        X = representations
        y = np.asarray(labels)
        classes = np.unique(y)
        if classes.size < 2:
            return 0.0

        global_mean = X.mean(axis=0)
        d = X.shape[1]

        sigma_w = np.zeros((d, d))
        sigma_b = np.zeros((d, d))
        total_n = X.shape[0]

        for c in classes:
            Xc = X[y == c]
            nc = Xc.shape[0]
            if nc == 0:
                continue
            mean_c = Xc.mean(axis=0)

            diff_w = Xc - mean_c
            sigma_w += diff_w.T @ diff_w

            diff_b = (mean_c - global_mean).reshape(-1, 1)
            sigma_b += nc * (diff_b @ diff_b.T)

        sigma_w /= total_n
        sigma_b /= classes.size

        trace_w = np.trace(sigma_w)
        trace_b = np.trace(sigma_b)

        if trace_b < 1e-12:
            return float("inf")

        return float(trace_w / trace_b)
