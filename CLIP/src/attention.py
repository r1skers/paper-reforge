"""
Multi-head self-attention (bidirectional, with optional key-padding mask).

VENDORED
--------
This is a verbatim copy of paper-reforge/Transformer/src/attention.py
brought into CLIP/src/ to keep CLIP self-contained — no `sys.path.append`
chains across paper-reforge subdirectories.

If you fix a bug here, also fix the original Transformer copy (or vice versa).
A future refactor (paper-reforge as an editable package) will retire this
duplication.

Note: CLIP's text encoder does NOT use this class — it uses a CAUSAL variant
in causal_attention.py, which takes an additive mask of shape (L, L).
This bidirectional version is used by CLIP's IMAGE encoder via EncoderBlock
(see transformer_block.py).

Run
---
    python attention.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention.

    Parameters
    ----------
    d_model : int
        Total embedding dim.
    num_heads : int
        Number of attention heads h. Must divide d_model.
        Per-head dim is d_k = d_v = d_model // num_heads.
    dropout : float, default 0.0
        Dropout prob applied to the attention weights.
    bias : bool, default False
        Whether the Q/K/V/O linear projections include a bias.
        Original Transformer uses bias=False for these; we match that.

    Forward
    -------
    x : torch.Tensor
        Shape (B, n, d_model).
    mask : torch.BoolTensor or None
        Shape (B, n).  True = real token (allowed as a key), False = padding.

    Returns
    -------
    output : torch.Tensor
        Shape (B, n, d_model).
    attn : torch.Tensor
        Shape (B, h, n, n) — attention weights, kept for inspection.
    """

    def __init__(self, d_model, num_heads, dropout=0.0, bias=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads

        # ONE big projection per role; reshape recovers per-head slices.
        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)
        self.W_O = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, n, d_model = x.shape
        h, d_k = self.h, self.d_k

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.reshape(B, n, h, d_k).transpose(1, 2)
        K = K.reshape(B, n, h, d_k).transpose(1, 2)
        V = V.reshape(B, n, h, d_k).transpose(1, 2)

        S = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            # mask shape (B, n) — broadcast to (B, 1, 1, n), then key positions
            S = S.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn = F.softmax(S, dim=-1)
        attn = self.attn_dropout(attn)
        head_out = attn @ V

        output = head_out.transpose(1, 2).reshape(B, n, d_model)
        output = self.W_O(output)
        return output, attn


def main():
    torch.manual_seed(0)
    B, n, d_model, h = 2, 5, 8, 2
    x = torch.randn(B, n, d_model)
    mha = MultiHeadSelfAttention(d_model, h, bias=False)
    out, attn = mha(x)
    print(f"x.shape       = {tuple(x.shape)}")
    print(f"output.shape  = {tuple(out.shape)}    (expected ({B}, {n}, {d_model}))")
    print(f"attn.shape    = {tuple(attn.shape)}   (expected ({B}, {h}, {n}, {n}))")
    print(f"row sums == 1?  {torch.allclose(attn.sum(dim=-1), torch.ones(B, h, n))}")


if __name__ == "__main__":
    main()
