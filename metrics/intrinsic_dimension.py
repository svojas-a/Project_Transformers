"""
Intrinsic dimensionality estimation.

Reference:
  Facco et al. (2017), "Estimating the intrinsic dimension of datasets by a
  minimal neighborhood information" -> the TwoNN estimator, which uses only
  the ratio of distances to the first and second nearest neighbors, making
  it robust to non-uniform density (unlike PCA-based estimates).
"""
from typing import Optional
import numpy as np
from .base import CollapseMetric, MetricResult


class TwoNNIntrinsicDimensionMetric(CollapseMetric):
    """
    Estimates the manifold's intrinsic dimension independent of the
    ambient (allocated) hidden size. Comparing this to hidden_dim directly
    tells you how much allocated capacity is actually being used.
    """

    def __init__(self, discard_fraction: float = 0.1):
        # Facco et al. recommend discarding the top ~10% of ratio values as
        # they're the least reliable (long-tail neighbors).
        self._discard_fraction = discard_fraction

    @property
    def name(self) -> str:
        return "intrinsic_dimension_twonn"

    def compute(
        self,
        representations: np.ndarray,
        labels: Optional[np.ndarray] = None,
        reference_representations: Optional[np.ndarray] = None,
    ) -> MetricResult:
        from sklearn.neighbors import NearestNeighbors

        X = representations
        n = X.shape[0]
        if n < 10:
            return 0.0

        nbrs = NearestNeighbors(n_neighbors=3).fit(X)
        distances, _ = nbrs.kneighbors(X)
        # distances[:, 0] is self (0); use 1st and 2nd true neighbors
        r1 = distances[:, 1]
        r2 = distances[:, 2]

        valid = r1 > 1e-12
        mu = r2[valid] / r1[valid]
        mu = mu[mu > 1]  # ratio must exceed 1 by definition

        if mu.size < 5:
            return 0.0

        # sort and discard the unreliable tail
        mu_sorted = np.sort(mu)
        keep = int(len(mu_sorted) * (1 - self._discard_fraction))
        keep = max(keep, 5)
        mu_kept = mu_sorted[:keep]

        # MLE: d = N / sum(log(mu_i))  [Facco et al. eq. 3, empirical CDF fit]
        N = mu_kept.size
        log_mu_sum = np.sum(np.log(mu_kept))
        if log_mu_sum <= 0:
            return 0.0

        d_estimate = N / log_mu_sum
        return float(d_estimate)
