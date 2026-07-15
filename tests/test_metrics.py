"""
tests/test_metrics.py  —  Unit tests for shared_lib/metrics.py (M2).

Per the Team Execution Plan (M2 implementation notes):
    "Write and pass unit tests BEFORE wiring these into the Phase-1 training
    loop — a metric bug found after 63 runs means re-running all of them."

Every metric is checked against a hand-computed 3-vector toy example
(orthonormal tokens = maximally diverse, and identical/parallel tokens =
fully collapsed), plus the small-d NaN/epsilon-guard behavior called out
as the specific numerical risk for M2.
"""

import numpy as np
import pytest
import torch

from shared_lib.metrics import (
    compute_attention_entropy,
    compute_effective_rank,
    compute_stable_rank,
    compute_token_cosine_similarity,
)

ATOL = 1e-4

# --------------------------------------------------------------------------- #
# Hand-computed 3-vector toy examples (see /tmp/check_math.py derivation)
# --------------------------------------------------------------------------- #

# Orthonormal rows -> singular values [1, 1, 1] -> "maximally spread" case.
ORTHONORMAL_TOKENS = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

# Three identical rows -> singular values [sqrt(3), 0, 0] -> fully collapsed.
COLLAPSED_TOKENS = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
)


# --------------------------------------------------------------------------- #
# compute_effective_rank
# --------------------------------------------------------------------------- #


class TestComputeEffectiveRank:
    def test_orthonormal_tokens_give_full_rank(self):
        result = compute_effective_rank(ORTHONORMAL_TOKENS)
        assert result == pytest.approx(3.0, abs=ATOL)

    def test_collapsed_tokens_give_rank_one(self):
        result = compute_effective_rank(COLLAPSED_TOKENS)
        assert result == pytest.approx(1.0, abs=ATOL)

    def test_batched_input_no_reduction_returns_per_example(self):
        batch = torch.stack([ORTHONORMAL_TOKENS, COLLAPSED_TOKENS])
        result = compute_effective_rank(batch, reduction="none")
        assert result.shape == (2,)
        assert result[0].item() == pytest.approx(3.0, abs=ATOL)
        assert result[1].item() == pytest.approx(1.0, abs=ATOL)

    def test_accepts_numpy_array(self):
        # M5/M6 load .npz back as numpy — the function must work unchanged.
        result = compute_effective_rank(ORTHONORMAL_TOKENS.numpy())
        assert result == pytest.approx(3.0, abs=ATOL)

    def test_small_d_does_not_produce_nan(self):
        # d=16 is one of the two explicitly called-out unstable sizes.
        torch.manual_seed(0)
        degenerate = torch.zeros(4, 16)  # all-zero rows: worst case for SVD
        result = compute_effective_rank(degenerate)
        assert torch.isfinite(result)


# --------------------------------------------------------------------------- #
# compute_stable_rank
# --------------------------------------------------------------------------- #


class TestComputeStableRank:
    def test_orthonormal_tokens_give_full_rank(self):
        result = compute_stable_rank(ORTHONORMAL_TOKENS)
        assert result == pytest.approx(3.0, abs=ATOL)

    def test_collapsed_tokens_give_rank_one(self):
        result = compute_stable_rank(COLLAPSED_TOKENS)
        assert result == pytest.approx(1.0, abs=ATOL)

    def test_small_d_does_not_produce_nan(self):
        degenerate = torch.zeros(4, 16)
        result = compute_stable_rank(degenerate)
        assert torch.isfinite(result)


# --------------------------------------------------------------------------- #
# compute_token_cosine_similarity
# --------------------------------------------------------------------------- #


class TestComputeTokenCosineSimilarity:
    def test_orthonormal_tokens_give_zero_similarity(self):
        result = compute_token_cosine_similarity(ORTHONORMAL_TOKENS)
        assert result == pytest.approx(0.0, abs=ATOL)

    def test_collapsed_tokens_give_similarity_one(self):
        result = compute_token_cosine_similarity(COLLAPSED_TOKENS)
        assert result == pytest.approx(1.0, abs=ATOL)

    def test_excludes_self_similarity_diagonal(self):
        # A single repeated pair plus one orthogonal outlier: if the
        # diagonal (self-similarity=1) leaked in, this would read higher
        # than the true off-diagonal mean.
        tokens = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        # off-diagonal pairs: (0,1)=1.0, (0,2)=0.0, (1,2)=0.0 -> mean = 1/3
        result = compute_token_cosine_similarity(tokens)
        assert result == pytest.approx(1.0 / 3.0, abs=ATOL)

    def test_zero_norm_token_does_not_produce_nan(self):
        tokens = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],  # e.g. a padding token
                [0.0, 1.0, 0.0],
            ]
        )
        result = compute_token_cosine_similarity(tokens)
        assert torch.isfinite(result)


# --------------------------------------------------------------------------- #
# compute_attention_entropy
# --------------------------------------------------------------------------- #


class TestComputeAttentionEntropy:
    def test_uniform_attention_gives_max_entropy(self):
        uniform = torch.full((4,), 0.25)
        result = compute_attention_entropy(uniform)
        assert result == pytest.approx(np.log(4), abs=ATOL)

    def test_one_hot_attention_gives_zero_entropy(self):
        peaked = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = compute_attention_entropy(peaked)
        assert result == pytest.approx(0.0, abs=1e-3)

    def test_handles_arbitrary_leading_dims(self):
        # [batch, heads, seq_q, seq_k] shaped attention, uniform over seq_k.
        attn = torch.full((2, 3, 5, 4), 0.25)
        result = compute_attention_entropy(attn, reduction="none")
        assert result.shape == (2, 3, 5)
        expected = torch.full((2, 3, 5), np.log(4), dtype=result.dtype)
        assert torch.allclose(result, expected, atol=ATOL)


# --------------------------------------------------------------------------- #
# Cross-cutting: pure-function contract (SOLID Single Responsibility)
# --------------------------------------------------------------------------- #


def test_functions_do_not_mutate_input():
    original = ORTHONORMAL_TOKENS.clone()
    compute_effective_rank(ORTHONORMAL_TOKENS)
    compute_stable_rank(ORTHONORMAL_TOKENS)
    compute_token_cosine_similarity(ORTHONORMAL_TOKENS)
    assert torch.equal(original, ORTHONORMAL_TOKENS)
