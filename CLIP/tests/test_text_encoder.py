"""
Tests for CLIP/src/causal_attention.py + text_encoder.py.

核心要验证的事
--------------
1. shape: (B, L) token_ids → (B, d_shared) text_embed
2. causal mask 真的阻断未来 — 改一个 token 后, 它之前位置的输出必须 bit-exact 不变
3. EOS indexing 取的是 eos_pos 那一行, 不是末位置或固定位置
4. 所有 trainable 参数都注册在 .parameters() 里 — 优化器看得到
5. build_causal_mask 形状和数值 (上三角 -inf, 主对角线 + 下三角 0)

Run
---
    cd D:/Dev/repos/paper-reforge/CLIP
    python -m pytest tests/test_text_encoder.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from causal_attention import (
    CausalMultiHeadSelfAttention,
    build_causal_mask,
)
from text_encoder import CausalEncoderBlock, TextTransformer


# ---------------------------------------------------------------------------- #
# build_causal_mask                                                            #
# ---------------------------------------------------------------------------- #


def test_causal_mask_shape():
    mask = build_causal_mask(8)
    assert mask.shape == (8, 8)


def test_causal_mask_values():
    """对角线 + 下三角 = 0, 严格上三角 = -inf."""
    L = 5
    mask = build_causal_mask(L)
    for i in range(L):
        for j in range(L):
            if j <= i:
                assert mask[i, j].item() == 0.0, f"位置 ({i},{j}) 应该是 0"
            else:
                assert mask[i, j].item() == float("-inf"), f"位置 ({i},{j}) 应该是 -inf"


def test_causal_mask_diagonal_not_masked():
    """主对角线必须是 0 — query 必须能看自己, 否则 softmax 出 NaN 行."""
    L = 6
    mask = build_causal_mask(L)
    diag = torch.diagonal(mask)
    assert torch.all(diag == 0.0)


# ---------------------------------------------------------------------------- #
# CausalMultiHeadSelfAttention                                                 #
# ---------------------------------------------------------------------------- #


def test_causal_attention_shape():
    B, L, d_model, h = 2, 8, 32, 4
    attn = CausalMultiHeadSelfAttention(d_model, h, dropout=0.0)
    attn.eval()

    x = torch.randn(B, L, d_model)
    mask = build_causal_mask(L)
    out, weights = attn(x, attn_mask=mask)

    assert out.shape == (B, L, d_model)
    assert weights.shape == (B, h, L, L)


def test_causal_attention_zero_to_future():
    """加上 causal mask 后, attention weights 严格上三角必须全 0."""
    torch.manual_seed(0)
    B, L, d_model, h = 1, 6, 16, 2
    attn = CausalMultiHeadSelfAttention(d_model, h, dropout=0.0)
    attn.eval()

    x = torch.randn(B, L, d_model)
    mask = build_causal_mask(L)
    _, weights = attn(x, attn_mask=mask)

    # weights[..., i, j] for j > i 必须 = 0
    upper = torch.triu(weights[0, 0], diagonal=1)
    assert upper.abs().max().item() < 1e-7, "未来位置的 attention weight 没归零"


def test_causal_attention_rows_sum_to_one():
    """每行 softmax 后必须和为 1 (主对角线 + 下三角的概率分布)."""
    torch.manual_seed(0)
    B, L, d_model, h = 2, 5, 16, 2
    attn = CausalMultiHeadSelfAttention(d_model, h, dropout=0.0)
    attn.eval()

    x = torch.randn(B, L, d_model)
    mask = build_causal_mask(L)
    _, weights = attn(x, attn_mask=mask)

    row_sums = weights.sum(dim=-1)  # (B, h, L)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_causal_attention_no_mask_is_full_attention():
    """不传 mask 时退化成无约束 self-attention — 每行的所有列都可以非零."""
    torch.manual_seed(0)
    B, L, d_model, h = 1, 4, 16, 2
    attn = CausalMultiHeadSelfAttention(d_model, h, dropout=0.0)
    attn.eval()

    x = torch.randn(B, L, d_model)
    _, weights = attn(x, attn_mask=None)

    # 至少有一些上三角的权重应该非零 (没 mask 应该可以看未来)
    upper = torch.triu(weights[0, 0], diagonal=1)
    assert upper.abs().max().item() > 1e-3


# ---------------------------------------------------------------------------- #
# Causal property at the FULL TextTransformer level                            #
# ---------------------------------------------------------------------------- #


def test_text_encoder_shape():
    vocab_size, max_len = 50, 16
    d_model, d_shared = 64, 32

    model = TextTransformer(
        vocab_size=vocab_size, max_len=max_len,
        d_model=d_model, d_shared=d_shared,
        depth=2, num_heads=4, dropout=0.0,
    )
    model.eval()

    B = 3
    token_ids = torch.randint(0, vocab_size, (B, max_len))
    eos_pos = torch.tensor([3, 8, max_len - 1])
    out = model(token_ids, eos_pos)

    assert out.shape == (B, d_shared)


def test_causal_property_changing_future_does_not_affect_past():
    """
    *** 这是 M1 最重要的一个测试 ***

    Causal mask 的物理含义是: 改变位置 k 之后的 token, 不应该影响
    位置 < k 的输出向量. 我们直接构造两个只在末尾不同的输入,
    抽出中间层每个位置的 hidden state, 比较 bit-exact.
    """
    torch.manual_seed(0)
    vocab_size, max_len = 30, 8
    d_model = 32

    model = TextTransformer(
        vocab_size=vocab_size, max_len=max_len,
        d_model=d_model, d_shared=16,
        depth=2, num_heads=4, dropout=0.0,
    )
    model.eval()

    # 两个序列: 前 5 个 token 完全相同, 后 3 个不同
    ids_a = torch.tensor([[1, 5, 7, 9, 11, 13, 15, 2]])  # last is EOS_ID=2
    ids_b = torch.tensor([[1, 5, 7, 9, 11, 20, 25, 2]])  # 改了位置 5, 6

    # 手动跑到 final_norm 之前 (能看到每个位置的隐藏状态)
    with torch.no_grad():
        x_a = model.token_emb(ids_a) + model.pos_emb
        x_b = model.token_emb(ids_b) + model.pos_emb
        for blk in model.blocks:
            x_a = blk(x_a, attn_mask=model.causal_mask)
            x_b = blk(x_b, attn_mask=model.causal_mask)

    # 前 5 个位置 (索引 0..4) 不应该受位置 5/6 改动的影响
    diff_past = (x_a[0, :5] - x_b[0, :5]).abs().max().item()
    assert diff_past < 1e-6, (
        f"Causal mask 被破坏 — 改未来 token 影响了过去位置: max diff = {diff_past:.2e}"
    )

    # Sanity check: 位置 5 之后必须确实不同 (否则我们的扰动没生效)
    diff_future = (x_a[0, 5:] - x_b[0, 5:]).abs().max().item()
    assert diff_future > 1e-3, "扰动没生效, 测试本身有问题"


# ---------------------------------------------------------------------------- #
# EOS indexing correctness                                                     #
# ---------------------------------------------------------------------------- #


def test_eos_indexing_picks_correct_position():
    """
    构造一个特殊情形: 不同样本的 eos_pos 不同, 验证模型确实是从
    eos_pos 那一行取 feature, 而不是固定取最后一位或第一位.

    手段: 模型 eval 模式跑一遍, 拿到 (B, max_len, d_model) 的中间状态,
    再分别从 eos_pos 位置取 feature, 应该和 model.forward 出来的
    sentence_feat 一致 (在 projection 之前).
    """
    torch.manual_seed(0)
    vocab_size, max_len = 30, 10
    d_model = 32

    model = TextTransformer(
        vocab_size=vocab_size, max_len=max_len,
        d_model=d_model, d_shared=16,
        depth=2, num_heads=4, dropout=0.0,
    )
    model.eval()

    B = 3
    token_ids = torch.randint(0, vocab_size, (B, max_len))
    eos_pos = torch.tensor([2, 5, 9])

    # 手动跑到 final_norm 之后, 在 projection 之前
    with torch.no_grad():
        x = model.token_emb(token_ids) + model.pos_emb
        for blk in model.blocks:
            x = blk(x, attn_mask=model.causal_mask)
        x = model.final_norm(x)  # (B, max_len, d_model)

        # 手动取 EOS feature
        expected_sentence = x[torch.arange(B), eos_pos]  # (B, d_model)
        expected_embed = model.text_projection(expected_sentence)

        # 模型 forward 出来的
        out = model(token_ids, eos_pos)

    assert torch.allclose(out, expected_embed, atol=1e-6)


def test_different_eos_positions_give_different_outputs():
    """同一个 batch 同样的 token_ids, eos_pos 不同 → output 应该不同 (取了不同行)."""
    torch.manual_seed(0)
    vocab_size, max_len = 30, 10
    model = TextTransformer(
        vocab_size=vocab_size, max_len=max_len,
        d_model=32, d_shared=16, depth=2, num_heads=4, dropout=0.0,
    )
    model.eval()

    token_ids = torch.randint(0, vocab_size, (1, max_len)).expand(2, -1).contiguous()
    eos_pos_a = torch.tensor([2, 2])
    eos_pos_b = torch.tensor([8, 8])

    with torch.no_grad():
        out_a = model(token_ids, eos_pos_a)
        out_b = model(token_ids, eos_pos_b)

    diff = (out_a - out_b).abs().max().item()
    assert diff > 1e-3, "不同 eos_pos 出来的 embedding 居然一样, EOS indexing 没生效"


# ---------------------------------------------------------------------------- #
# Parameter registration                                                       #
# ---------------------------------------------------------------------------- #


def test_all_params_registered():
    """所有 trainable 组件 — token_emb / pos_emb / blocks / final_norm /
    text_projection — 都必须出现在 .parameters() 里."""
    model = TextTransformer(
        vocab_size=50, max_len=16, d_model=32, d_shared=16,
        depth=2, num_heads=4,
    )
    names = {n for n, _ in model.named_parameters()}

    # token embedding
    assert "token_emb.weight" in names
    # positional embedding (nn.Parameter 直接放在 self 上)
    assert "pos_emb" in names
    # at least one block 的 attention W_Q / FFN fc1 / LayerNorm
    assert any("blocks.0.attn.W_Q.weight" in n for n in names)
    assert any("blocks.0.ffn.fc1.weight" in n for n in names)
    assert any("blocks.0.norm1.weight" in n for n in names)
    # final norm
    assert "final_norm.weight" in names
    # text projection (no bias, only weight)
    assert "text_projection.weight" in names
    assert "text_projection.bias" not in names


def test_causal_mask_not_in_state_dict():
    """causal_mask 用 persistent=False 注册 — 不应该出现在 state_dict 里."""
    model = TextTransformer(
        vocab_size=50, max_len=16, d_model=32, d_shared=16,
        depth=2, num_heads=4,
    )
    keys = model.state_dict().keys()
    assert "causal_mask" not in keys


# ---------------------------------------------------------------------------- #
# Block-level smoke test                                                       #
# ---------------------------------------------------------------------------- #


def test_causal_encoder_block_shape():
    B, L, d_model = 2, 6, 32
    blk = CausalEncoderBlock(d_model=d_model, num_heads=4, dropout=0.0)
    blk.eval()

    x = torch.randn(B, L, d_model)
    mask = build_causal_mask(L)
    out = blk(x, attn_mask=mask)
    assert out.shape == x.shape


def test_causal_encoder_block_identity_highway():
    """
    Pre-norm 性质: zero 掉所有 sublayer (attention + FFN) 权重,
    block 应该退化成 identity. 这是 Transformer 阶段验过的同一个 invariant.
    """
    torch.manual_seed(0)
    B, L, d_model = 2, 5, 16
    blk = CausalEncoderBlock(d_model=d_model, num_heads=4, dropout=0.0)
    blk.eval()

    with torch.no_grad():
        for name, p in blk.named_parameters():
            if name.startswith("attn.") or name.startswith("ffn."):
                p.zero_()

    x = torch.randn(B, L, d_model)
    mask = build_causal_mask(L)
    out = blk(x, attn_mask=mask)
    max_diff = (out - x).abs().max().item()
    assert max_diff < 1e-6, f"identity-highway 失败: max diff = {max_diff:.2e}"
