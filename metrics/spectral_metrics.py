"""
Rank & spectral metrics — the linear-algebraic view of collapse.

References:
  Roy & Vetterli (2007), "The effective rank: a measure of effective
  dimensionality" — effective rank via entropy of the normalized
  singular-value distribution.
"""
from typing import Optional
import numpy as np
from .base import CollapseMetric, MetricResult


def _singular_values(representations: np.ndarray) -> np.ndarray:
    # Center first: collapse metrics should reflect variance structure,
    # not be dominated by a nonzero mean.
    centered = representations - representations.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    return s


class EffectiveRankMetric(CollapseMetric):
    """
    erank(X) = exp( H(p) ), where p_i = sigma_i / sum(sigma).
    Ranges from 1 (fully collapsed, one direction) to min(n, d) (isotropic).
    """

    @property
    def name(self) -> str:
        return "effective_rank"

    def compute(
        self,
        representations: np.ndarray,
        labels: Optional[np.ndarray] = None,
        reference_representations: Optional[np.ndarray] = None,
    ) -> MetricResult:
        s = _singular_values(representations)
        s = s[s > 1e-12]
        if s.size == 0:
            return 0.0
        p = s / s.sum()
        entropy = -np.sum(p * np.log(p))
        return float(np.exp(entropy))


class StableRankMetric(CollapseMetric):
    """
    stable_rank(X) = ||X||_F^2 / ||X||_2^2 = sum(sigma_i^2) / max(sigma_i)^2.
    Cheaper and more noise-robust than effective rank; same intuition.
    """

    @property
    def name(self) -> str:
        return "stable_rank"

    def compute(
        self,
        representations: np.ndarray,
        labels: Optional[np.ndarray] = None,
        reference_representations: Optional[np.ndarray] = None,
    ) -> MetricResult:
        s = _singular_values(representations)
        if s.size == 0 or s[0] == 0:
            return 0.0
        return float(np.sum(s ** 2) / (s[0] ** 2))


class ConditionNumberMetric(CollapseMetric):
    """
    sigma_max / sigma_min of the covariance matrix. Rises sharply (toward
    infinity) as the representation covariance becomes near-singular —
    a direct signal of collapse into a lower-dimensional subspace.
    Reported on a log10 scale since raw values can span many orders of
    magnitude.
    """

    @property
    def name(self) -> str:
        return "log10_condition_number"

    def compute(
        self,
        representations: np.ndarray,
        labels: Optional[np.ndarray] = None,
        reference_representations: Optional[np.ndarray] = None,
    ) -> MetricResult:
        s = _singular_values(representations)
        s = s[s > 1e-12]
        if s.size < 2:
            return float("inf")
        cond = s[0] / s[-1]
        return float(np.log10(cond))


class SingularValueDecayMetric(CollapseMetric):
    """
    Fits sigma_i ~ C * i^(-alpha) via log-log linear regression.
    Larger alpha = faster spectral decay = sharper collapse into few
    directions. Returned alongside the fit R^2 so noisy fits are visible.
    """

    @property
    def name(self) -> str:
        return "singular_value_decay"

    def compute(
        self,
        representations: np.ndarray,
        labels: Optional[np.ndarray] = None,
        reference_representations: Optional[np.ndarray] = None,
    ) -> MetricResult:
        s = _singular_values(representations)
        s = s[s > 1e-12]
        if s.size < 3:
            return {"alpha": 0.0, "r_squared": 0.0}

        log_i = np.log(np.arange(1, s.size + 1))
        log_s = np.log(s)
        A = np.vstack([log_i, np.ones_like(log_i)]).T
        (alpha_neg, intercept), residuals, _, _ = np.linalg.lstsq(A, log_s, rcond=None)
        alpha = -alpha_neg

        pred = A @ np.array([alpha_neg, intercept])
        ss_res = np.sum((log_s - pred) ** 2)
        ss_tot = np.sum((log_s - log_s.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {"alpha": float(alpha), "r_squared": float(r_squared)}
