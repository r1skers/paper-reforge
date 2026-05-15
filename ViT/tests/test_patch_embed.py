"""
Tests for V1 — patch embedding.

Five tests:
    1. Output shape (Conv impl)
    2. Output shape (Unfold impl)
    3. Parameter count matches d_model * (C * P * P) + d_model
    4. PatchEmbedConv uses stride == kernel (non-overlapping invariant)
    5. BIT-EXACT EQUIVALENCE between Conv and Unfold under weight transfer
       — this is the central test, it proves the math we discussed in
       lecture (Conv2d-stride=kernel === "slice + flatten + Linear").

The equivalence test runs in float64 to remove fp32 rounding noise and
asserts max-abs diff < 1e-12.
"""

import sys
from pathlib import Path

# Make `src/` importable whether we run via pytest or directly.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from patch_embed import PatchEmbedConv, PatchEmbedUnfold


# ---------------------------------------------------------------------------- #
# Test 1 — Conv impl output shape                                              #
# ---------------------------------------------------------------------------- #
def test_conv_output_shape():
    """PatchEmbedConv: (B, C, H, W) -> (B, N, d_model)."""
    # ------------------------------------------------------------------ #
    # TODO 1 — Fill in a forward pass and assert shape.                   #
    #                                                                     #
    #   B, C, H, P, d = 2, 3, 32, 4, 128                                  #
    #   pe = PatchEmbedConv(img_size=H, patch_size=P,                     #
    #                      in_chans=C, d_model=d)                         #
    #   pe.eval()                                                          #
    #   x = torch.randn(B, C, H, H)                                       #
    #   y = pe(x)                                                          #
    #   N = (H // P) ** 2     # = 64 for 32/4                              #
    #   assert y.shape == (B, N, d), f"got {tuple(y.shape)}"               #
    # ------------------------------------------------------------------ #
    B, C, H, P, d = 2, 3, 32, 4, 128
    pe = PatchEmbedConv(img_size=H, patch_size=P, in_chans=C, d_model=d)
    pe.eval()
    x = torch.randn(B, C, H, H)
    y = pe(x)
    N = (H // P) ** 2
    assert y.shape == (B, N, d), f"got {tuple(y.shape)}"


# ---------------------------------------------------------------------------- #
# Test 2 — Unfold impl output shape                                            #
# ---------------------------------------------------------------------------- #
def test_unfold_output_shape():
    """PatchEmbedUnfold: (B, C, H, W) -> (B, N, d_model)."""
    # ------------------------------------------------------------------ #
    # TODO 2 — Same pattern as test 1, but with PatchEmbedUnfold.         #
    # ------------------------------------------------------------------ #
    raise NotImplementedError("Fill in TODO 2")


# ---------------------------------------------------------------------------- #
# Test 3 — Parameter count                                                     #
# ---------------------------------------------------------------------------- #
def test_param_count_matches_formula():
    """Both impls must have exactly  d_model * (C * P * P) + d_model  params."""
    # ------------------------------------------------------------------ #
    # TODO 3 — Compute the expected count and compare.                    #
    #                                                                     #
    #   C, P, d = 3, 4, 128                                                #
    #   expected = d * (C * P * P) + d        # weights + bias             #
    #                                                                     #
    #   conv   = PatchEmbedConv  (img_size=32, patch_size=P,               #
    #                              in_chans=C, d_model=d)                  #
    #   unfold = PatchEmbedUnfold(img_size=32, patch_size=P,               #
    #                              in_chans=C, d_model=d)                  #
    #                                                                     #
    #   n_conv   = sum(p.numel() for p in conv.parameters())               #
    #   n_unfold = sum(p.numel() for p in unfold.parameters())             #
    #                                                                     #
    #   assert n_conv   == expected, f"conv:   {n_conv}   vs {expected}"   #
    #   assert n_unfold == expected, f"unfold: {n_unfold} vs {expected}"   #
    # ------------------------------------------------------------------ #
    raise NotImplementedError("Fill in TODO 3")


# ---------------------------------------------------------------------------- #
# Test 4 — Stride == kernel invariant on the conv impl                         #
# ---------------------------------------------------------------------------- #
def test_conv_stride_equals_kernel():
    """The non-overlapping invariant: PatchEmbedConv.proj.stride must equal
    its kernel_size. If it doesn't, the 'tokenization' breaks down into
    overlapping CNN-style feature extraction and the bit-exact equivalence
    with Unfold also fails."""
    # ------------------------------------------------------------------ #
    # TODO 4 — Inspect conv.proj and assert.                              #
    #                                                                     #
    #   pe = PatchEmbedConv(img_size=32, patch_size=4,                    #
    #                       in_chans=3, d_model=128)                      #
    #   assert pe.proj.kernel_size == pe.proj.stride, (                   #
    #       f"kernel {pe.proj.kernel_size} vs stride {pe.proj.stride}"    #
    #   )                                                                  #
    # ------------------------------------------------------------------ #
    raise NotImplementedError("Fill in TODO 4")


# ---------------------------------------------------------------------------- #
# Test 5 — THE CENTRAL TEST: bit-exact equivalence                             #
# ---------------------------------------------------------------------------- #
def test_conv_unfold_bit_exact_under_weight_copy():
    """The two implementations must produce numerically identical outputs
    when their weights are properly aligned.

    The alignment rule (which is the whole point of this exercise):

        W_lin [j, :]  =  W_conv [j, :, :, :].flatten()
        b_lin [j]     =  b_conv [j]

    PyTorch's Conv2d weight has shape (out, in, kH, kW). Flattening dim 1
    onwards gives (out, in * kH * kW), which matches Linear's weight
    shape (out, in_features) exactly — IF the flattening order matches
    how F.unfold lays out its output. PyTorch uses C-major for both,
    so it does match.

    Running in float64 to suppress fp32 noise.
    """
    # ------------------------------------------------------------------ #
    # TODO 5 — Implement the bit-exact check.                             #
    #                                                                     #
    #  Step A — build both modules with same config and cast to double:   #
    #                                                                     #
    #   torch.manual_seed(0)                                              #
    #   B, C, H, P, d = 2, 3, 32, 4, 128                                  #
    #   conv   = PatchEmbedConv  (img_size=H, patch_size=P,                #
    #                              in_chans=C, d_model=d).double()        #
    #   unfold = PatchEmbedUnfold(img_size=H, patch_size=P,                #
    #                              in_chans=C, d_model=d).double()        #
    #   conv.eval()                                                        #
    #   unfold.eval()                                                      #
    #                                                                     #
    #  Step B — copy conv weights into the linear layer:                  #
    #                                                                     #
    #   with torch.no_grad():                                              #
    #       # conv.proj.weight: (d, C, P, P)                               #
    #       # flatten(1) -> (d, C*P*P) — exactly Linear's weight shape    #
    #       unfold.proj.weight.copy_(conv.proj.weight.flatten(1))         #
    #       unfold.proj.bias  .copy_(conv.proj.bias)                       #
    #                                                                     #
    #  Step C — forward both with the same input:                         #
    #                                                                     #
    #   x = torch.randn(B, C, H, H, dtype=torch.float64)                   #
    #   y_conv   = conv(x)                                                 #
    #   y_unfold = unfold(x)                                               #
    #                                                                     #
    #  Step D — assert max abs diff is below the float64 tolerance:       #
    #                                                                     #
    #   max_diff = (y_conv - y_unfold).abs().max().item()                  #
    #   assert max_diff < 1e-12, f"not bit-exact: max diff = {max_diff}"   #
    #                                                                     #
    # If this test fails, the most common culprits are:                    #
    #   - forgot to copy bias                                               #
    #   - wrong flatten order (use .flatten(1) on the conv weight,         #
    #     NOT some manual reshape that scrambles axes)                     #
    #   - one impl is in fp32 and the other in fp64                        #
    # ------------------------------------------------------------------ #
    raise NotImplementedError("Fill in TODO 5")


if __name__ == "__main__":
    test_conv_output_shape()
    # test_unfold_output_shape()
    # test_param_count_matches_formula()
    # test_conv_stride_equals_kernel()
    # test_conv_unfold_bit_exact_under_weight_copy()
    print("All patch_embed tests passed!")
