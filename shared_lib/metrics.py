"""
shared_lib/metrics.py  —  M2: Metrics Library

Implements the four measurements that define "collapse" in this study:
    1. compute_effective_rank        — entropy of normalized singular values
    2. compute_stable_rank           — (sum(sigma))^2 / sum(sigma^2)
    3. compute_token_cosine_similarity — mean pairwise cosine similarity of tokens
    4. compute_attention_entropy     — entropy of attention distributions

Design contract (per Team Execution Plan, Section 5 / M2):
    - Pure functions only. No file I/O, no logging, no experiment-specific
      logic. This file has zero knowledge of any phaseN_*/ folder or of
      shared_lib/models.py (SOLID: Single Responsibility).
    - Fully vectorized over the batch dimension (torch.linalg ops broadcast
      over leading dims) — no Python-level loops over batch or layer.
    - SVD-based metrics get an epsilon guard + NaN/Inf check, since they are
      numerically unstable at very small d (16, 32).
    - Called every N steps inside M4's training loop on tensors handed to it
      by M3's hooks, and reused UNCHANGED by M5/M6 when re-analyzing saved
      .npz arrays — so no torch-only assumptions that would break on numpy
      arrays loaded back from disk. Every function accepts either.

Naming (Section 7.1 of the plan):
    - funcs: snake_case, verb_noun            -> compute_effective_rank
    - constants: UPPER_SNAKE_CASE             -> DEFAULT_EPSILON
    - raw_ prefix for raw/unreduced tensors, d_ for dimension-related values,
      n_ for counts, is_/has_ for booleans.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import torch

Reduction = Literal["mean", "none"]
ArrayLike = Union[torch.Tensor, np.ndarray]

DEFAULT_EPSILON = 1e-8


# --------------------------------------------------------------------------- #
# Internal helpers (private — not part of M2's public contract)
# --------------------------------------------------------------------------- #


def _as_tensor(raw_array: ArrayLike) -> torch.Tensor:
    """Accept torch.Tensor or np.ndarray (M5/M6 load .npz -> numpy) and
    return a float64 torch.Tensor for numerically stable SVD/log ops."""
    if isinstance(raw_array, np.ndarray):
        tensor = torch.from_numpy(raw_array)
    else:
        tensor = raw_array
    return tensor.detach().to(dtype=torch.float64)


def _check_finite(result: torch.Tensor, metric_name: str, d_dim: int) -> None:
    """NaN/Inf check required for SVD-based metrics (see module docstring).
    Raises rather than silently swallowing, so M4's sweep loop can catch,
    log, and re-queue the run instead of writing a corrupted .npz."""
    if not torch.isfinite(result).all():
        raise ValueError(
            f"{metric_name} produced NaN/Inf at d={d_dim}. This is the known "
            f"instability at small hidden dims (16, 32) — check the epsilon "
            f"guard and inspect the input tensor for degenerate (all-zero or "
            f"duplicate) rows before treating this as a model bug."
        )


def _reduce(result: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    return result.mean() if reduction == "mean" else result


# --------------------------------------------------------------------------- #
# 1. Effective rank — entropy of normalized singular values
# --------------------------------------------------------------------------- #


def compute_effective_rank(
    raw_hidden_states: ArrayLike,
    epsilon: float = DEFAULT_EPSILON,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Effective rank (Roy & Vetterli): exp(entropy of normalized singular values).

    Args:
        raw_hidden_states: [seq_len, d] or [batch, seq_len, d] hidden states.
        epsilon: numerical guard for the near-zero-singular-value case at
            small d — added inside the normalization and the log.
        reduction: "mean" collapses the batch dim to a scalar; "none" returns
            one effective-rank value per batch element.

    Returns:
        0-d tensor (reduction="mean") or [batch] tensor (reduction="none").
    """
    hidden = _as_tensor(raw_hidden_states)
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)  # -> [1, seq_len, d]

    d_dim = hidden.shape[-1]
    singular_values = torch.linalg.svdvals(hidden)  # [batch, min(seq_len, d)]

    p = singular_values / (singular_values.sum(dim=-1, keepdim=True) + epsilon)
    entropy = -(p * torch.log(p + epsilon)).sum(dim=-1)  # [batch]
    effective_rank = torch.exp(entropy)

    _check_finite(effective_rank, "compute_effective_rank", d_dim)
    return _reduce(effective_rank, reduction)


# --------------------------------------------------------------------------- #
# 2. Stable rank — (sum(sigma))^2 / sum(sigma^2)
# --------------------------------------------------------------------------- #


def compute_stable_rank(
    raw_hidden_states: ArrayLike,
    epsilon: float = DEFAULT_EPSILON,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Stable rank: (sum_i(sigma_i))^2 / sum_i(sigma_i^2).

    Args mirror compute_effective_rank. Same epsilon guard + NaN check,
    since this is also SVD-based and unstable at small d.
    """
    hidden = _as_tensor(raw_hidden_states)
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)

    d_dim = hidden.shape[-1]
    singular_values = torch.linalg.svdvals(hidden)  # [batch, min(seq_len, d)]

    numerator = singular_values.sum(dim=-1).pow(2)
    denominator = singular_values.pow(2).sum(dim=-1) + epsilon
    stable_rank = numerator / denominator

    _check_finite(stable_rank, "compute_stable_rank", d_dim)
    return _reduce(stable_rank, reduction)


# --------------------------------------------------------------------------- #
# 3. Pairwise token cosine similarity
# --------------------------------------------------------------------------- #


def compute_token_cosine_similarity(
    raw_hidden_states: ArrayLike,
    epsilon: float = DEFAULT_EPSILON,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Mean pairwise cosine similarity between token vectors in a sequence.

    Collapse shows up here as similarity -> 1: every token vector pointing
    the same direction. Excludes the diagonal (self-similarity is always 1
    and would trivially inflate the mean, masking real collapse).

    Args:
        raw_hidden_states: [seq_len, d] or [batch, seq_len, d].
        epsilon: guard against dividing by a near-zero-norm token vector
            (e.g. a padding token that collapsed to all-zero).
        reduction: "mean" -> scalar; "none" -> [batch].
    """
    hidden = _as_tensor(raw_hidden_states)
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)  # -> [1, seq_len, d]

    n_tokens = hidden.shape[-2]
    norms = hidden.norm(dim=-1, keepdim=True)  # [batch, seq_len, 1]
    normalized = hidden / (norms + epsilon)

    similarity_matrix = torch.bmm(
        normalized, normalized.transpose(-2, -1)
    )  # [batch, seq_len, seq_len]

    off_diagonal_mask = ~torch.eye(n_tokens, dtype=torch.bool, device=hidden.device)
    off_diagonal_values = similarity_matrix[:, off_diagonal_mask].reshape(
        hidden.shape[0], -1
    )

    mean_cosine_similarity = off_diagonal_values.mean(dim=-1)  # [batch]
    return _reduce(mean_cosine_similarity, reduction)


# --------------------------------------------------------------------------- #
# 4. Attention entropy
# --------------------------------------------------------------------------- #


def compute_attention_entropy(
    raw_attention_weights: ArrayLike,
    epsilon: float = DEFAULT_EPSILON,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Entropy of each attention row-distribution, averaged over heads/tokens.

    Args:
        raw_attention_weights: post-softmax attention weights of shape
            [..., seq_len_q, seq_len_k] — any number of leading dims
            (batch, heads, layer, etc.) is supported; each row along the
            last axis is treated as a probability distribution and must
            already sum to ~1.
        epsilon: guard against log(0) for a fully peaked (one-hot) row,
            which is itself a valid, extreme collapse signal (entropy -> 0)
            and must not produce NaN.
        reduction: "mean" -> scalar; "none" -> entropy per row, same shape
            as raw_attention_weights minus the last dim.
    """
    attn = _as_tensor(raw_attention_weights)
    row_entropy = -(attn * torch.log(attn + epsilon)).sum(dim=-1)

    _check_finite(row_entropy, "compute_attention_entropy", attn.shape[-1])
    return _reduce(row_entropy, reduction)
