"""
Tests for src/analysis/causal_propagation.py

These tests avoid downloading the real pretrained DistilBERT (slow, and
requires network access in CI) by building a tiny randomly-initialized
DistilBERT from a DistilBertConfig instead. This is enough to verify the
actual mechanism under test -- the low-rank collapse hook and the metric
functions -- without depending on model weights or external data.
"""

import numpy as np
import pytest
import torch
from transformers import DistilBertConfig, DistilBertModel

from src.analysis.causal_propagation import (
    LowRankCollapseHook,
    attention_entropy,
    compute_all_metrics,
    effective_rank,
    mean_pairwise_cosine,
    stable_rank,
)


@pytest.fixture
def tiny_model():
    """A small, randomly-initialized 6-layer DistilBERT, fast enough to run
    forward passes on in CI without a GPU or network access."""
    config = DistilBertConfig(
        dim=32,
        hidden_dim=128,
        n_heads=4,
        n_layers=6,
        vocab_size=1000,
        attn_implementation="eager",
    )
    model = DistilBertModel(config)
    model.eval()
    return model


@pytest.fixture
def tiny_input():
    torch.manual_seed(0)
    return torch.randint(0, 1000, (1, 10))


# --------------------------------------------------------------------------
# Metric function tests -- known-input, known-output sanity checks
# --------------------------------------------------------------------------


class TestMetricFunctions:
    def test_effective_rank_full_rank_matrix(self):
        # A random full-rank matrix should have effective rank close to
        # its true dimensionality, not collapsed to 1.
        torch.manual_seed(1)
        h = torch.randn(20, 16)
        result = effective_rank(h)
        assert result > 5  # nowhere near collapsed
        assert result <= 16  # can't exceed the dimensionality

    def test_effective_rank_rank_one_matrix(self):
        # A rank-1 matrix (all rows are scalar multiples of one vector)
        # should have effective rank close to 1.
        base = torch.randn(1, 16)
        h = base.repeat(20, 1) * torch.arange(1, 21).unsqueeze(1).float()
        result = effective_rank(h)
        assert result == pytest.approx(1.0, abs=1e-3)

    def test_stable_rank_identity_like(self):
        # Orthogonal-ish rows -> stable rank should be well above 1
        torch.manual_seed(2)
        h = torch.eye(10, 16)
        result = stable_rank(h)
        assert result > 5

    def test_mean_pairwise_cosine_identical_rows(self):
        # All rows identical -> cosine similarity between every pair is 1
        h = torch.ones(10, 16)
        result = mean_pairwise_cosine(h)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_mean_pairwise_cosine_orthogonal_rows(self):
        # Orthogonal rows -> mean pairwise cosine should be close to 0
        h = torch.eye(8, 8)
        result = mean_pairwise_cosine(h)
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_attention_entropy_uniform_is_maximal(self):
        # Uniform attention distribution has maximal entropy = log(seq_len)
        seq_len = 10
        uniform_attn = torch.full((4, seq_len, seq_len), 1.0 / seq_len)
        result = attention_entropy(uniform_attn)
        assert result == pytest.approx(np.log(seq_len), abs=1e-3)

    def test_attention_entropy_one_hot_is_zero(self):
        # Fully peaked (one-hot) attention has zero entropy
        seq_len = 10
        one_hot = torch.zeros(4, seq_len, seq_len)
        one_hot[:, :, 0] = 1.0
        result = attention_entropy(one_hot)
        assert result == pytest.approx(0.0, abs=1e-3)

    def test_compute_all_metrics_returns_expected_keys(self):
        hidden = torch.randn(10, 16)
        attn = torch.softmax(torch.randn(4, 10, 10), dim=-1)
        result = compute_all_metrics(hidden, attn)
        assert set(result.keys()) == {
            "effective_rank",
            "stable_rank",
            "mean_pairwise_cosine",
            "attention_entropy",
        }
        assert all(isinstance(v, float) for v in result.values())


# --------------------------------------------------------------------------
# Collapse hook tests -- the core causal mechanism
# --------------------------------------------------------------------------


class TestLowRankCollapseHook:
    def test_hook_inactive_by_default_leaves_output_unchanged(
        self, tiny_model, tiny_input
    ):
        hook_fn = LowRankCollapseHook(target_rank=1)
        handle = tiny_model.transformer.layer[2].register_forward_hook(hook_fn)

        hook_fn.active = False
        with torch.no_grad():
            out_no_hook = tiny_model(input_ids=tiny_input, output_hidden_states=True)
        handle.remove()

        with torch.no_grad():
            out_baseline = tiny_model(input_ids=tiny_input, output_hidden_states=True)

        # With the hook inactive, output should match a run with no hook at all
        assert torch.allclose(
            out_no_hook.hidden_states[-1], out_baseline.hidden_states[-1], atol=1e-6
        )

    def test_hook_reduces_rank_at_source_layer(self, tiny_model, tiny_input):
        target_rank = 1
        hook_fn = LowRankCollapseHook(target_rank=target_rank)
        handle = tiny_model.transformer.layer[2].register_forward_hook(hook_fn)
        hook_fn.active = True

        with torch.no_grad():
            out = tiny_model(input_ids=tiny_input, output_hidden_states=True)
        handle.remove()

        # hidden_states[0] = embeddings, hidden_states[l+1] = output of layer l
        source_layer_output = out.hidden_states[3][
            0
        ]  # layer index 2 -> hidden_states[3]
        rank = np.linalg.matrix_rank(source_layer_output.numpy())
        assert rank <= target_rank

    def test_hook_effect_propagates_downstream(self, tiny_model, tiny_input):
        """The central causal claim of this module: collapsing a source
        layer should change the representations of layers *after* it,
        even though the hook only touches the source layer directly."""
        source_layer = 2
        hook_fn = LowRankCollapseHook(target_rank=1)
        handle = tiny_model.transformer.layer[source_layer].register_forward_hook(
            hook_fn
        )

        hook_fn.active = False
        with torch.no_grad():
            out_clean = tiny_model(input_ids=tiny_input, output_hidden_states=True)

        hook_fn.active = True
        with torch.no_grad():
            out_collapsed = tiny_model(input_ids=tiny_input, output_hidden_states=True)
        handle.remove()

        for downstream_layer in range(source_layer + 1, 6):
            clean = out_clean.hidden_states[downstream_layer + 1][0]
            collapsed = out_collapsed.hidden_states[downstream_layer + 1][0]
            # Representations at every downstream layer must differ from
            # the clean run -- this is the propagation signal itself.
            assert not torch.allclose(clean, collapsed, atol=1e-4), (
                f"Expected layer {downstream_layer} to be affected by the "
                f"upstream collapse at layer {source_layer}, but it was unchanged."
            )

    def test_hook_does_not_touch_upstream_layers(self, tiny_model, tiny_input):
        """Layers *before* the source layer should be completely identical
        between clean and collapsed runs, since the hook is registered only
        on the source layer and forward passes are strictly feed-forward."""
        source_layer = 3
        hook_fn = LowRankCollapseHook(target_rank=1)
        handle = tiny_model.transformer.layer[source_layer].register_forward_hook(
            hook_fn
        )

        hook_fn.active = False
        with torch.no_grad():
            out_clean = tiny_model(input_ids=tiny_input, output_hidden_states=True)

        hook_fn.active = True
        with torch.no_grad():
            out_collapsed = tiny_model(input_ids=tiny_input, output_hidden_states=True)
        handle.remove()

        for upstream_layer in range(0, source_layer):
            clean = out_clean.hidden_states[upstream_layer][0]
            collapsed = out_collapsed.hidden_states[upstream_layer][0]
            assert torch.allclose(clean, collapsed, atol=1e-6)

    def test_hook_respects_requested_rank_upper_bound(self, tiny_model, tiny_input):
        # Requesting a rank larger than the hidden dimension should not error;
        # it should just be capped at the max possible rank.
        hook_fn = LowRankCollapseHook(target_rank=999)
        handle = tiny_model.transformer.layer[0].register_forward_hook(hook_fn)
        hook_fn.active = True
        with torch.no_grad():
            out = tiny_model(input_ids=tiny_input, output_hidden_states=True)
        handle.remove()
        assert out is not None  # no crash
