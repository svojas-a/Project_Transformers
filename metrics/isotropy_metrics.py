"""
Isotropy metrics — do embeddings spread evenly through the space, or crowd
into a narrow cone?

References:
  Ethayarajh (2019), "How Contextual are Contextualized Word Representations?"
    -> average pairwise cosine similarity.
  Rudman et al. (2022), "IsoScore: Measuring the Uniformity of Embedding
    Space Utilization" -> corrected isotropy score in [0, 1].
"""
from typing import Optional
import numpy as np
from .base import CollapseMetric, MetricResult


class AverageCosineSimilarityMetric(CollapseMetric):
    """
    Mean cosine similarity across all (or a sampled subset of) pairs.
    Close to 0 = isotropic/spread out. Close to 1 = collapsed into a cone.
    """

    def __init__(self, max_pairs: int = 5000, random_state: int = 0):
        self._max_pairs = max_pairs
        self._rng = np.random.default_rng(random_state)

    @property
    def name(self) -> str:
        return "avg_cosine_similarity"

    def compute(
        self,
        representations: np.ndarray,
        labels: Optional[np.ndarray] = None,
        reference_representations: Optional[np.ndarray] = None,
    ) -> MetricResult:
        n = representations.shape[0]
        if n < 2:
            return 0.0

        norms = np.linalg.norm(representations, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        normed = representations / norms

        total_pairs = n * (n - 1) // 2
        if total_pairs <= self._max_pairs:
            sims = normed @ normed.T
            iu = np.triu_indices(n, k=1)
            return float(np.mean(sims[iu]))

        # sample pairs for large n to keep this O(max_pairs) instead of O(n^2)
        i = self._rng.integers(0, n, self._max_pairs)
        j = self._rng.integers(0, n, self._max_pairs)
        mask = i != j
        i, j = i[mask], j[mask]
        sims = np.sum(normed[i] * normed[j], axis=1)
        return float(np.mean(sims))


class IsoScoreMetric(CollapseMetric):
    """
    IsoScore (Rudman et al., 2022): normalizes the covariance eigenvalue
    spectrum against the isotropic case, corrected for known scaling issues
    in the earlier partition-function isotropy measure. Range [0, 1], where
    1 = perfectly isotropic use of the space.
    """

    @property
    def name(self) -> str:
        return "iso_score"

    def compute(
        self,
        representations: np.ndarray,
        labels: Optional[np.ndarray] = None,
        reference_representations: Optional[np.ndarray] = None,
    ) -> MetricResult:
        X = representations - representations.mean(axis=0, keepdims=True)
        n, d = X.shape
        if n < 2 or d < 2:
            return 0.0

        cov = np.cov(X, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 0, None)

        # normalize eigenvalues to unit norm, compare distance from uniform
        norm_eigvals = eigvals / (np.linalg.norm(eigvals) + 1e-12)
        uniform = np.full(d, 1.0 / np.sqrt(d))

        dist = np.linalg.norm(norm_eigvals - uniform)
        max_dist = np.linalg.norm(
            np.array([1.0] + [0.0] * (d - 1)) - uniform
        )
        phi = 1 - (dist / max_dist) if max_dist > 0 else 0.0

        iso_score = (d * phi - 1) / (d - 1) if d > 1 else phi
        return float(np.clip(iso_score, 0.0, 1.0))
