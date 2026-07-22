"""
Representational alignment metrics.

Reference:
  Kornblith, Norouzi, Lee, Hinton (2019), "Similarity of Neural Network
  Representations Revisited" -> Centered Kernel Alignment (CKA), which is
  invariant to orthogonal transforms and isotropic scaling, making it a fair
  comparison even when the two representation matrices have different
  dimensionality (exactly our case: full model vs. reduced model).
"""
from typing import Optional
import numpy as np
from .base import CollapseMetric, MetricResult


def _center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _linear_hsic(K: np.ndarray, L: np.ndarray) -> float:
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    return float(np.sum(Kc * Lc))


class LinearCKAMetric(CollapseMetric):
    """
    CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L)), using linear
    kernels K = X X^T, L = Y Y^T. Range [0, 1]; 1 = identical representational
    geometry up to rotation/isotropic scaling.

    Requires `reference_representations` — typically the full, unreduced
    model's representations on the same input samples. Sample order between
    `representations` and `reference_representations` must match.
    """

    @property
    def name(self) -> str:
        return "linear_cka_vs_reference"

    def compute(
        self,
        representations: np.ndarray,
        labels: Optional[np.ndarray] = None,
        reference_representations: Optional[np.ndarray] = None,
    ) -> MetricResult:
        if reference_representations is None:
            raise ValueError(
                f"{self.name} requires reference_representations "
                "(e.g. the full/unreduced model's representations on the "
                "same samples)."
            )

        X = representations
        Y = reference_representations
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Sample count mismatch between representations ({X.shape[0]}) "
                f"and reference ({Y.shape[0]}); they must be computed on the "
                "same ordered set of inputs."
            )

        K = X @ X.T
        L = Y @ Y.T

        hsic_xy = _linear_hsic(K, L)
        hsic_xx = _linear_hsic(K, K)
        hsic_yy = _linear_hsic(L, L)

        denom = np.sqrt(hsic_xx * hsic_yy)
        if denom < 1e-12:
            return 0.0
        return float(hsic_xy / denom)
