"""
Validates the fixed TruncationStrategy against a hand-built module that
mirrors DistilBERT's structural quirks (nn.Embedding, nn.LayerNorm,
nn.Linear, and a self-attention-like submodule that caches `dim` as a plain
int attribute used at forward time) -- without needing network access to
download real weights. If this passes, the same logic applies to the real
DistilBertWrapper.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from reduction.reduction_strategy import TruncationStrategy


class FakeAttention(nn.Module):
    """Mirrors DistilBERT's MultiHeadSelfAttention: caches `dim` as a plain
    int attribute (not a Parameter/buffer) and uses it for head-splitting
    math at forward time -- exactly the pattern that broke in the wild."""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)

    def forward(self, x):
        bs, seq_len, _ = x.shape
        head_dim = self.dim // self.n_heads  # would break if self.dim is stale
        q = self.q_lin(x).view(bs, seq_len, self.n_heads, head_dim)
        k = self.k_lin(x).view(bs, seq_len, self.n_heads, head_dim)
        v = self.v_lin(x).view(bs, seq_len, self.n_heads, head_dim)
        attn = torch.einsum("bshd,bthd->bhst", q, k) / (head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhst,bthd->bshd", attn, v).reshape(bs, seq_len, self.dim)
        return self.out_lin(out)


class FakeDistilBertBlock(nn.Module):
    def __init__(self, dim, ffn_dim, n_heads):
        super().__init__()
        self.attention = FakeAttention(dim, n_heads)
        self.sa_layer_norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, ffn_dim)
        self.lin2 = nn.Linear(ffn_dim, dim)
        self.output_layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.sa_layer_norm(x + self.attention(x))
        ffn_out = self.lin2(torch.relu(self.lin1(x)))
        return self.output_layer_norm(x + ffn_out)


class FakeConfig:
    def __init__(self, dim, n_heads):
        self.dim = dim
        self.n_heads = n_heads


class FakeDistilBert(nn.Module):
    def __init__(self, vocab_size=1000, dim=768, ffn_dim=3072, n_heads=12):
        super().__init__()
        self.config = FakeConfig(dim, n_heads)
        self.word_embeddings = nn.Embedding(vocab_size, dim)
        self.embed_layer_norm = nn.LayerNorm(dim)
        self.block = FakeDistilBertBlock(dim, ffn_dim, n_heads)

    def get_input_embeddings(self):
        return self.word_embeddings

    def forward(self, input_ids):
        x = self.embed_layer_norm(self.word_embeddings(input_ids))
        return self.block(x)


def test_truncation_produces_working_forward_pass():
    model = FakeDistilBert(vocab_size=1000, dim=768, ffn_dim=3072, n_heads=12)
    strategy = TruncationStrategy()

    input_ids = torch.randint(0, 1000, (2, 10))

    # sanity: original model runs fine
    out_full = model(input_ids)
    assert out_full.shape == (2, 10, 768)

    # first reduction step: 768 -> 612 (rounded to multiple of 12, matches
    # DimensionSchedule(round_to=12) behavior)
    reduced_1 = strategy.reduce(model, target_dim=612)
    out_1 = reduced_1(input_ids)
    assert out_1.shape == (2, 10, 612), f"got {out_1.shape}"
    assert reduced_1.block.attention.dim == 612, "attention.dim wasn't synced"
    assert reduced_1.block.sa_layer_norm.normalized_shape == (612,)
    assert reduced_1.word_embeddings.embedding_dim == 612

    # second reduction step, chained: 612 -> 480
    reduced_2 = strategy.reduce(reduced_1, target_dim=480)
    out_2 = reduced_2(input_ids)
    assert out_2.shape == (2, 10, 480), f"got {out_2.shape}"
    assert reduced_2.block.attention.dim == 480

    # non-divisible target should raise a clear error, not a cryptic crash
    try:
        strategy.reduce(reduced_2, target_dim=100)  # 100 % 12 != 0
        raise AssertionError("expected ValueError for non-divisible target_dim")
    except ValueError as e:
        assert "divisible" in str(e)

    print("All TruncationStrategy structural tests passed.")
    print(f"  full -> 768: {out_full.shape}")
    print(f"  768 -> 612:  {out_1.shape}")
    print(f"  612 -> 480:  {out_2.shape}")


if __name__ == "__main__":
    test_truncation_produces_working_forward_pass()
