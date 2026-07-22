"""
Task performance evaluation via linear probing.

Rather than fine-tuning a full classifier head per hidden-dim size (slow,
and conflates "collapse" with "how well we retrained the head"), we fit a
cheap linear probe on the SAME representations already extracted for the
collapse metrics, and report cross-validated accuracy. This is a standard
technique in representation-analysis literature (see e.g. probing-classifier
papers): it measures how linearly separable the task-relevant information
still is at a given hidden size, which is exactly the "downstream cost of
collapse" signal Phase 1 needs to answer "at which dimension does it
correctly collapse."

Kept as its own module/interface (Single Responsibility, Interface
Segregation) so a future module can swap this for a real fine-tuned
classifier without touching ExperimentRunner or the metrics code.
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class TaskPerformanceEvaluator(ABC):
    @abstractmethod
    def evaluate(self, representations: np.ndarray, labels: Optional[np.ndarray]) -> Optional[float]:
        """Returns a performance score, or None if it can't be computed
        (e.g. no labels available for this task/sample)."""
        raise NotImplementedError


class LinearProbeAccuracyEvaluator(TaskPerformanceEvaluator):
    """
    Fits sklearn LogisticRegression on (representations, labels) and returns
    mean cross-validated accuracy. Works for any integer-labeled
    classification task (sst2, mnli, the coarse conll2003 proxy labels).
    """

    def __init__(self, cv_folds: int = 3, max_iter: int = 1000, random_state: int = 0):
        self._cv_folds = cv_folds
        self._max_iter = max_iter
        self._random_state = random_state

    def evaluate(self, representations: np.ndarray, labels: Optional[np.ndarray]) -> Optional[float]:
        if labels is None:
            return None

        labels = np.asarray(labels)
        n_classes = len(np.unique(labels))
        n_samples = representations.shape[0]

        # cross_val_score requires at least cv_folds samples per class;
        # skip gracefully rather than crashing a long-running sweep.
        if n_classes < 2 or n_samples < self._cv_folds * n_classes:
            return None

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        # Standardize first: representation magnitude shrinks as hidden_dim
        # shrinks (fewer dims -> smaller norms in general), which would
        # otherwise bias the probe's regularization differently at each
        # size and confound "collapse" with "unscaled features."
        X = StandardScaler().fit_transform(representations)

        clf = LogisticRegression(max_iter=self._max_iter, random_state=self._random_state)
        try:
            scores = cross_val_score(clf, X, labels, cv=self._cv_folds)
        except ValueError:
            return None
        return float(scores.mean())