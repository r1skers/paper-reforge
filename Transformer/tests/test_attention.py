"""
T2.4 — Unit tests for MultiHeadSelfAttention.

Run
---
    pytest tests/                     # standard
    python tests/test_attention.py    # works without pytest installed

What we cover
-------------
1.  Output shape    — sanity that forward returns (B, n, d_model)
2.  Attn shape + row sums — softmax normalization invariant
3.  Numerical equivalence vs numpy reference — the strict correctness proof
4.  Padding mask zeros masked key columns
5.  Padding mask preserves row sums (no NaN, no broken normalization)
6.  Gradient flow — every projection receives a non-zero gradient
"""

import sys
from pathlib import Path

# Make `src/` importable whether we run via pytest or directly.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch

from attention import MultiHeadSelfAttention
from attention_numpy import multi_head_attention


# --------------------------------------------------------------------------- #
# 1.  Shape sanity                                                            #
# --------------------------------------------------------------------------- #

def test_output_shape():
    """Forward output must have shape (B, n, d_model), same as input."""
    B, n, d_model, h = 3, 7, 16, 4
    mha = MultiHeadSelfAttention(d_model, h)
    x = torch.randn(B, n, d_model)
    out, _ = mha(x)
    assert out.shape == (B, n, d_model), f"got {tuple(out.shape)}"


# --------------------------------------------------------------------------- #
# 2.  Attention shape and softmax row-sum invariant                           #
# --------------------------------------------------------------------------- #

def test_attn_shape_and_row_sums():
    """attn must be (B, h, n, n) and every row must sum to 1 (softmax)."""
    B, n, d_model, h = 3, 7, 16, 4
    mha = MultiHeadSelfAttention(d_model, h)
    mha.eval()  # disable dropout so weights stay normalized
    x = torch.randn(B, n, d_model)
    _, attn = mha(x)

    assert attn.shape == (B, h, n, n), f"got {tuple(attn.shape)}"
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
        f"row sums deviated from 1: max |sum-1| = {(row_sums - 1).abs().max().item()}"


# --------------------------------------------------------------------------- #
# 3.  Strict numerical equivalence: PyTorch (float64) ≈ numpy reference       #
# --------------------------------------------------------------------------- #

def test_pytorch_matches_numpy():
    """
    With matching weights, the PyTorch implementation must reproduce the
    numpy reference to ~float64 machine precision.

    This is THE correctness test — if the reshape/transpose choreography
    or the mask broadcast is wrong, this test catches it instantly.
    """
    rng = np.random.default_rng(seed=42)
    n, d_model, h = 5, 8, 2

    X_np  = rng.standard_normal((n, d_model))
    WQ_np = rng.standard_normal((d_model, d_model))
    WK_np = rng.standard_normal((d_model, d_model))
    WV_np = rng.standard_normal((d_model, d_model))
    WO_np = rng.standard_normal((d_model, d_model))

    # numpy reference (no batch dim)
    out_np, attn_np = multi_head_attention(X_np, WQ_np, WK_np, WV_np, WO_np, h)

    # PyTorch: use float64, copy weights (mind the transpose)
    mha = MultiHeadSelfAttention(d_model, h, bias=False).double()
    mha.eval()
    with torch.no_grad():
        mha.W_Q.weight.copy_(torch.from_numpy(WQ_np.T))
        mha.W_K.weight.copy_(torch.from_numpy(WK_np.T))
        mha.W_V.weight.copy_(torch.from_numpy(WV_np.T))
        mha.W_O.weight.copy_(torch.from_numpy(WO_np.T))

    x_t = torch.from_numpy(X_np).unsqueeze(0)  # (1, n, d_model)
    with torch.no_grad():
        out_t, attn_t = mha(x_t)

    np.testing.assert_allclose(out_t.squeeze(0).numpy(),  out_np,  atol=1e-12)
    np.testing.assert_allclose(attn_t.squeeze(0).numpy(), attn_np, atol=1e-12)


# --------------------------------------------------------------------------- #
# 4.  Padding mask zeros out masked key columns                               #
# --------------------------------------------------------------------------- #

def test_padding_mask_zeros_keys():
    """
    Masked key columns must receive EXACTLY zero attention.
    (softmax(-inf) = 0 strictly, no float wiggle.)
    """
    B, n, d_model, h = 2, 6, 8, 2
    mha = MultiHeadSelfAttention(d_model, h)
    mha.eval()
    x = torch.randn(B, n, d_model)

    # Different pad lengths per sequence to make sure broadcasting is right.
    mask = torch.tensor([
        [True, True, True, True,  False, False],   # batch 0: pad last 2
        [True, True, True, False, False, False],   # batch 1: pad last 3
    ])
    _, attn = mha(x, mask=mask)

    assert (attn[0, :, :, 4:6] == 0).all(), "batch 0: cols 4,5 should be 0"
    assert (attn[1, :, :, 3:6] == 0).all(), "batch 1: cols 3..5 should be 0"


# --------------------------------------------------------------------------- #
# 5.  Padding mask preserves row-sum-1 (and produces no NaN)                  #
# --------------------------------------------------------------------------- #

def test_padding_mask_preserves_row_sum():
    """
    After masking, softmax must still normalize over the remaining (real)
    keys, so every (B, h, n) row should still sum to 1.

    Also: no NaN anywhere.  NaN would mean a query row was fully masked
    (all keys = -inf), which our test setup avoids (we always leave at
    least one real key).
    """
    B, n, d_model, h = 2, 6, 8, 2
    mha = MultiHeadSelfAttention(d_model, h)
    mha.eval()
    x = torch.randn(B, n, d_model)
    mask = torch.tensor([
        [True, True, True, True,  False, False],
        [True, True, True, False, False, False],
    ])
    _, attn = mha(x, mask=mask)

    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    assert not torch.isnan(attn).any(), "NaN appeared in attention"


# --------------------------------------------------------------------------- #
# 6.  Gradient flow: every projection sees a non-zero gradient                #
# --------------------------------------------------------------------------- #

def test_gradient_flow():
    """
    A backward pass on a scalar loss must populate non-zero gradients on
    W_Q, W_K, W_V, W_O.  If any are missing or all-zero, something in
    forward is detaching from the graph or accidentally constant.
    """
    B, n, d_model, h = 2, 4, 8, 2
    mha = MultiHeadSelfAttention(d_model, h)
    x = torch.randn(B, n, d_model)
    out, _ = mha(x)
    loss = out.sum()
    loss.backward()

    for name in ("W_Q", "W_K", "W_V", "W_O"):
        weight = getattr(mha, name).weight
        assert weight.grad is not None, f"{name}.weight.grad is None"
        assert weight.grad.abs().sum().item() > 0.0, f"{name}.weight.grad is all zero"


# --------------------------------------------------------------------------- #
# Standalone runner (so the file works without pytest installed)              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    tests = [
        test_output_shape,
        test_attn_shape_and_row_sums,
        test_pytorch_matches_numpy,
        test_padding_mask_zeros_keys,
        test_padding_mask_preserves_row_sum,
        test_gradient_flow,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print()
    print(f"{len(tests) - failed} / {len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
