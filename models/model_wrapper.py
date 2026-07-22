"""
Model wrappers.

ModelWrapper is a narrow interface (Interface Segregation Principle): callers
only need `load`, `get_hidden_size`, `set_hidden_size`, and
`extract_representations`. Any transformer model that can implement these
four methods is substitutable wherever a ModelWrapper is expected (Liskov
Substitution) — this is what lets Phase 1 extend to other models later
without touching the runner or metrics code.
"""
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class ModelWrapper(ABC):
    """Narrow contract every model adapter must satisfy."""

    @abstractmethod
    def load(self) -> None:
        """Instantiate/reload the underlying model at its current hidden size."""
        raise NotImplementedError

    @abstractmethod
    def get_hidden_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def set_hidden_size(self, new_dim: int) -> None:
        """
        Reconfigure the model to a reduced hidden dimension. Implementations
        decide *how* (truncation, distillation, retraining) — the wrapper
        only guarantees that after this call, get_hidden_size() == new_dim.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_representations(self, texts: List[str]) -> np.ndarray:
        """Return an (n_samples, hidden_dim) matrix of pooled representations."""
        raise NotImplementedError


class DistilBertWrapper(ModelWrapper):
    """
    DistilBERT adapter. Actual dimension reduction is delegated to an
    injected ReductionStrategy (Dependency Inversion — this class depends on
    the ReductionStrategy abstraction, not a concrete reduction algorithm).
    """

    def __init__(self, model_name: str, reduction_strategy, device: str = "cpu"):
        self._model_name = model_name
        self._reduction_strategy = reduction_strategy
        self._device = device
        self._model = None
        self._tokenizer = None
        self._current_dim = None

    def load(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._model.eval()
        # Read the real tensor shape rather than trusting config.hidden_size —
        # DistilBertConfig stores this as `dim`, not `hidden_size`, and other
        # architectures vary too. This keeps DistilBertWrapper correct
        # without hardcoding a config attribute name.
        self._current_dim = self._model.get_input_embeddings().weight.shape[1]

    def get_hidden_size(self) -> int:
        return self._current_dim

    def set_hidden_size(self, new_dim: int) -> None:
        if self._model is None:
            raise RuntimeError("Call load() before set_hidden_size().")
        self._model = self._reduction_strategy.reduce(self._model, new_dim)
        self._current_dim = new_dim

    def extract_representations(self, texts: List[str]) -> np.ndarray:
        import torch
        if self._model is None:
            raise RuntimeError("Call load() before extract_representations().")

        all_reps = []
        with torch.no_grad():
            for text in texts:
                enc = self._tokenizer(
                    text, return_tensors="pt", truncation=True,
                    padding=True, max_length=128
                ).to(self._device)
                out = self._model(**enc)
                # mean-pool over tokens (excluding padding) -> one vector per sample
                hidden = out.last_hidden_state  # (1, seq_len, dim)
                mask = enc["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                all_reps.append(pooled.squeeze(0).cpu().numpy())
        return np.stack(all_reps, axis=0)
