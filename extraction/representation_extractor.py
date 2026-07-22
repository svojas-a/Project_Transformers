"""
RepresentationExtractor: the only class responsible for turning raw text
into representation matrices via a ModelWrapper. Kept separate from both the
model wrapper (which owns model state) and the runner (which owns
orchestration) so each has exactly one reason to change (Single
Responsibility Principle).

BUGFIX (seed variance): previously, `_subsample` drew a fresh random subset
on EVERY call to `extract()`, consuming further from the same RNG stream
each time. Within a single seed, that meant the reference (full-size)
extraction and each per-dimension extraction all saw DIFFERENT subsamples --
which silently broke CKA (it compares representations of DIFFERENT samples,
not the same samples reduced) and, when max_samples happened to be >= the
dataset size, meant no subsampling occurred at all, making every seed
produce byte-identical results.

Fix: an extractor instance now draws its subsample ONCE (on first use) and
reuses that exact fixed subset for every subsequent extract() call. Since a
NEW extractor is constructed per seed (see ExperimentRunner), this
guarantees: (a) all dimensions within one seed are evaluated on the same
samples -- correct CKA alignment -- and (b) different seeds draw genuinely
different subsamples -- real seed-to-seed variance.
"""
from typing import List, Optional
import numpy as np

from models.model_wrapper import ModelWrapper


class RepresentationExtractor:
    def __init__(self, model_wrapper: ModelWrapper, max_samples: int = 2000, seed: int = 0):
        self._model = model_wrapper
        self._max_samples = max_samples
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._fixed_indices: Optional[np.ndarray] = None  # cached on first use

    def extract(self, texts: List[str], labels: Optional[List[int]] = None):
        sub_texts, sub_labels = self._subsample_once(texts, labels)
        reps = self._model.extract_representations(sub_texts)
        labels_arr = np.asarray(sub_labels) if sub_labels is not None else None
        return reps, labels_arr

    def _subsample_once(self, texts: List[str], labels: Optional[List[int]]):
        """
        Draws the subsample index set exactly once per extractor instance
        and caches it, so repeated extract() calls (across dimension sizes,
        within the same seed) all operate on the identical sample set.
        """
        if self._fixed_indices is None:
            if len(texts) <= self._max_samples:
                # No subsampling possible/needed -- but warn, since this
                # means this seed contributes ZERO sample-driven variance.
                # (Model-driven variance from a frozen, dropout-off model is
                # also zero, so with max_samples >= len(texts) every seed
                # WILL produce identical results -- that's expected given
                # the current frozen-inference design, not a bug, but worth
                # knowing about explicitly rather than discovering it via
                # identical plots.)
                print(
                    f"[RepresentationExtractor] seed={self._seed}: "
                    f"max_samples ({self._max_samples}) >= dataset size "
                    f"({len(texts)}); no subsampling will occur, so this "
                    f"seed will NOT differ from other seeds. Pass a smaller "
                    f"--max_samples than the dataset size to get real "
                    f"seed-to-seed variance."
                )
                self._fixed_indices = np.arange(len(texts))
            else:
                self._fixed_indices = self._rng.choice(
                    len(texts), size=self._max_samples, replace=False
                )

        idx = self._fixed_indices
        sub_texts = [texts[i] for i in idx]
        sub_labels = [labels[i] for i in idx] if labels is not None else None
        return sub_texts, sub_labels