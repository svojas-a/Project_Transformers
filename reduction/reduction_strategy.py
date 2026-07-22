"""
Dimension-reduction strategies (Strategy Pattern).

New reduction methods (distillation, low-rank factorization, structured
pruning, etc.) get added by writing a new class here and registering it in
the factory — the runner, model wrapper, and metrics never need to change.
This is the Open/Closed Principle in action: open for extension, closed for
modification.
"""
"""
Dimension-reduction strategies (Strategy Pattern).

New reduction methods (distillation, low-rank factorization, structured
pruning, etc.) get added by writing a new class here and registering it in
the factory — the runner, model wrapper, and metrics never need to change.
This is the Open/Closed Principle in action: open for extension, closed for
modification.
"""
from abc import ABC, abstractmethod
import copy


class ReductionStrategy(ABC):
    @abstractmethod
    def reduce(self, model, target_dim: int):
        """Return a model whose hidden representation size is target_dim."""
        raise NotImplementedError


class TruncationStrategy(ReductionStrategy):
    """
    Slices weight matrices down to target_dim along the hidden dimension.
    Fastest option — good for the Phase 1 pipeline build-out — but note it
    conflates "collapse from truncation" with "collapse from capacity";
    document this caveat when interpreting Phase 1 results.

    IMPORTANT: swapping a layer's `weight`/`bias` Parameter to a smaller
    tensor is NOT enough — nn.Linear, nn.LayerNorm, and nn.Embedding all
    cache their shape as separate Python attributes (in_features/
    out_features, normalized_shape, embedding_dim) that PyTorch's forward
    pass actually checks against. This strategy rebuilds each affected
    module with the correct attributes rather than only slicing tensors.

    It also keeps attention head geometry consistent: DistilBERT's
    self-attention divides the hidden dim by n_heads at forward time, so
    target_dim must be divisible by n_heads, and any module attribute that
    mirrors the hidden size (e.g. MultiHeadSelfAttention.dim) is synced too
    — including the separately-cached PER-HEAD dimension (e.g.
    attention_head_size = dim // n_heads), which is what caused the
    "shape '[1, 12, -1, 64]' is invalid" crash.
    """

    def reduce(self, model, target_dim: int):
        import torch.nn as nn

        model = copy.deepcopy(model)
        current_dim = self._infer_current_dim(model)
        if target_dim >= current_dim:
            return model  # nothing to do; guards against upward "reduction"

        n_heads = getattr(model.config, "n_heads", None) or getattr(
            model.config, "num_attention_heads", 1
        )
        if target_dim % n_heads != 0:
            raise ValueError(
                f"target_dim={target_dim} is not divisible by n_heads={n_heads}. "
                "Round your DimensionSchedule to multiples of n_heads (see "
                "DimensionSchedule(..., round_to=n_heads)) or attention head "
                "reshaping will produce silently wrong results."
            )

        self._resize_modules(model, current_dim, target_dim)
        self._sync_scalar_dim_attrs(model, current_dim, target_dim, n_heads)
        return model

    @staticmethod
    def _infer_current_dim(model) -> int:
        """
        Reads the *actual* current hidden size off the embedding weight
        rather than trusting model.config, since config can drift out of
        sync with real tensor shapes across repeated reductions.
        """
        return model.get_input_embeddings().weight.shape[1]

    def _resize_modules(self, model, current_dim: int, target_dim: int) -> None:
        import torch
        import torch.nn as nn

        with torch.no_grad():
            for name, module in list(model.named_modules()):
                if isinstance(module, nn.Linear):
                    new_in = target_dim if module.in_features == current_dim else module.in_features
                    new_out = target_dim if module.out_features == current_dim else module.out_features
                    if new_in == module.in_features and new_out == module.out_features:
                        continue
                    new_module = nn.Linear(new_in, new_out, bias=module.bias is not None)
                    new_module.weight.copy_(module.weight[:new_out, :new_in])
                    if module.bias is not None:
                        new_module.bias.copy_(module.bias[:new_out])
                    self._set_submodule(model, name, new_module)

                elif isinstance(module, nn.LayerNorm):
                    if current_dim not in tuple(module.normalized_shape):
                        continue
                    new_module = nn.LayerNorm(
                        target_dim, eps=module.eps, elementwise_affine=module.elementwise_affine
                    )
                    if module.elementwise_affine:
                        new_module.weight.copy_(module.weight[:target_dim])
                        new_module.bias.copy_(module.bias[:target_dim])
                    self._set_submodule(model, name, new_module)

                elif isinstance(module, nn.Embedding):
                    if module.embedding_dim != current_dim:
                        continue
                    new_module = nn.Embedding(
                        module.num_embeddings, target_dim,
                        padding_idx=module.padding_idx,
                    )
                    new_module.weight.copy_(module.weight[:, :target_dim])
                    self._set_submodule(model, name, new_module)

    @staticmethod
    def _sync_scalar_dim_attrs(model, current_dim: int, target_dim: int, n_heads: int) -> None:
        """
        Some modules (e.g. DistilBERT's MultiHeadSelfAttention, Embeddings)
        cache the hidden size as a plain int attribute (self.dim) used at
        forward time for head-splitting math. These aren't nn.Parameters or
        buffers so _resize_modules never touches them — sweep every
        submodule's __dict__ and update any int attribute that matches the
        old hidden size. Also updates model.config for the same reason.

        Separately, many attention implementations ALSO cache the PER-HEAD
        dimension (e.g. `attention_head_size = dim // n_heads`, commonly 64
        for a 768-dim/12-head model) as its own int attribute. Because 64 !=
        current_dim, the sweep above never touches it, which is exactly what
        caused `query_layer.view(...)` to fail: heads still assumed 64-wide
        after the hidden dim shrank. We compute and correct that attribute
        explicitly here.
        """
        for module in model.modules():
            for attr_name, attr_val in list(vars(module).items()):
                if isinstance(attr_val, int) and attr_val == current_dim:
                    setattr(module, attr_name, target_dim)

        for attr_name, attr_val in list(vars(model.config).items()):
            if isinstance(attr_val, int) and attr_val == current_dim:
                setattr(model.config, attr_name, target_dim)

        # --- per-head dimension fix ---
        if n_heads and current_dim % n_heads == 0:
            old_head_dim = current_dim // n_heads
            new_head_dim = target_dim // n_heads
            if old_head_dim != new_head_dim:
                for module in model.modules():
                    for attr_name, attr_val in list(vars(module).items()):
                        if isinstance(attr_val, int) and attr_val == old_head_dim:
                            setattr(module, attr_name, new_head_dim)
                for attr_name, attr_val in list(vars(model.config).items()):
                    if isinstance(attr_val, int) and attr_val == old_head_dim:
                        setattr(model.config, attr_name, new_head_dim)

    @staticmethod
    def _set_submodule(model, name, new_module) -> None:
        """Replaces the child module at dotted path `name` with new_module."""
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_module)


class PCAProjectionStrategy(ReductionStrategy):
    """
    Post-hoc projection: wraps the full-size model's output with a PCA
    projection down to target_dim, fit on a calibration batch. Cheapest
    exploratory option; does not touch model weights.

    Usage note: this strategy needs calibration data, so it's initialized
    with a callable that supplies representative texts, keeping the strategy
    self-contained (High Cohesion) rather than reaching into the runner.
    """

    def __init__(self, calibration_texts_provider, tokenizer_provider):
        self._get_texts = calibration_texts_provider
        self._get_tokenizer = tokenizer_provider

    def reduce(self, model, target_dim: int):
        import torch
        from sklearn.decomposition import PCA

        tokenizer = self._get_tokenizer()
        texts = self._get_texts()
        reps = []
        model.eval()
        with torch.no_grad():
            for t in texts:
                enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=128)
                out = model(**enc)
                reps.append(out.last_hidden_state.mean(1).squeeze(0).numpy())

        import numpy as np
        reps = np.stack(reps)
        pca = PCA(n_components=target_dim)
        pca.fit(reps)

        return _PCAWrappedModel(model, pca)


class _PCAWrappedModel:
    """Thin decorator around a model that projects outputs through PCA."""

    def __init__(self, base_model, pca):
        self._base_model = base_model
        self._pca = pca
        self.config = base_model.config
        self.config.hidden_size = pca.n_components_

    def __call__(self, **kwargs):
        import torch
        out = self._base_model(**kwargs)
        pooled = out.last_hidden_state.mean(1).detach().numpy()
        projected = self._pca.transform(pooled)
        out.last_hidden_state = torch.tensor(projected).unsqueeze(1)
        return out

    def to(self, device):
        self._base_model.to(device)
        return self

    def eval(self):
        self._base_model.eval()
        return self


class ReductionStrategyFactory:
    """
    GRASP Creator: centralizes knowledge of which concrete strategy to
    instantiate for a given config name, so client code never imports
    concrete strategy classes directly.
    """

    _registry = {
        "truncation": TruncationStrategy,
    }

    @classmethod
    def register(cls, name: str, strategy_cls) -> None:
        cls._registry[name] = strategy_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> ReductionStrategy:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown reduction strategy '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)