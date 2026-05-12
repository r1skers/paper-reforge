"""
T2.3 — Multi-head self-attention as a PyTorch nn.Module.

Differences from attention_numpy.py
-----------------------------------
- Adds a batch dimension B (real-world tensors are always batched).
- Uses nn.Linear modules — weights/grads managed by PyTorch.
- Uses F.softmax and torch.matmul (auto-batched).
- Optional dropout on the attention weights (as in the original paper).

Run
---
    python attention.py
"""

import math

import numpy as np
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
        Number of attention heads h.  Must divide d_model.
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

        # ------------------------------------------------------------------ #
        # TODO 1 — Project to Q, K, V using the nn.Linear modules.            #
        # Just call them like functions (do NOT touch .weight manually):      #
        #                                                                     #
        #   Q = self.W_Q(x)     ->  (B, n, d_model)                           #
        #   K = self.W_K(x)     ->  (B, n, d_model)                           #
        #   V = self.W_V(x)     ->  (B, n, d_model)                           #
        # ------------------------------------------------------------------ #
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # ------------------------------------------------------------------ #
        # TODO 2 — Reshape from (B, n, d_model) to (B, h, n, d_k).            #
        # Same pattern as numpy, with B as an extra leading dim:              #
        #                                                                     #
        #   .reshape(B, n, h, d_k)        ->  (B, n, h, d_k)                  #
        #   .transpose(1, 2)              ->  (B, h, n, d_k)                  #
        #                                                                     #
        # Do this for Q, K, V.                                                #
        # ------------------------------------------------------------------ #
        Q = Q.reshape(B, n, h, d_k).transpose(1, 2)
        K = K.reshape(B, n, h, d_k).transpose(1, 2)
        V = V.reshape(B, n, h, d_k).transpose(1, 2)
        # ------------------------------------------------------------------ #
        # TODO 3 — Batched scaled dot-product attention.                      #
        #                                                                     #
        # Scores (note .transpose(-2, -1), NOT .T — same lesson as numpy):    #
        #   S = Q @ K.transpose(-2, -1) / math.sqrt(d_k)                      #
        #       ->  (B, h, n, n)                                              #
        #                                                                     #
        # Mask (only if mask is not None):                                    #
        #   `mask` has shape (B, n).  Broadcast to (B, 1, 1, n) so it covers  #
        #   (h, n_query, n_key) trailing dims, then mask key positions:       #
        #                                                                     #
        #       m = mask[:, None, None, :]              # (B, 1, 1, n)        #
        #       S = S.masked_fill(~m, float('-inf'))                          #
        #                                                                     #
        # Softmax along the last dim, then dropout the weights:               #
        #   attn = F.softmax(S, dim=-1)                  -> (B, h, n, n)      #
        #   attn = self.attn_dropout(attn)                                    #
        #                                                                     #
        # Apply to V:                                                         #
        #   head_out = attn @ V                          -> (B, h, n, d_k)    #
        # ------------------------------------------------------------------ #
        S = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            S = S.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn = F.softmax(S, dim=-1)
        attn = self.attn_dropout(attn)
        head_out = attn @ V
        # ------------------------------------------------------------------ #
        # TODO 4 — Recombine heads and apply output projection.               #
        #                                                                     #
        #   head_out.transpose(1, 2)                     -> (B, n, h, d_k)    #
        #   .reshape(B, n, d_model)                      -> (B, n, d_model)   #
        #   self.W_O(...)                                -> (B, n, d_model)   #
        #                                                                     #
        # NOTE: after transpose the tensor may not be contiguous.             #
        # `.reshape()` handles non-contiguous tensors automatically;          #
        # `.view()` would error out.  Prefer .reshape() unless you know       #
        # the tensor is contiguous.                                           #
        #                                                                     #
        # Return (output, attn).                                              #
        # ------------------------------------------------------------------ #
        output = head_out.transpose(1, 2).reshape(B, n, d_model)
        output = self.W_O(output)
        return output, attn


# ---------------------------------------------------------------------------- #
# Driver: shape + numerical equivalence vs numpy + mask smoke test             #
# ---------------------------------------------------------------------------- #


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(seed=0)

    # ---- Shape check ------------------------------------------------------- #
    print("=" * 60)
    print("T2.3 shape check")
    print("=" * 60)

    B, n, d_model, h = 2, 5, 8, 2
    x = torch.randn(B, n, d_model)
    mha = MultiHeadSelfAttention(d_model, h, bias=False)
    out, attn = mha(x)
    print(f"x.shape       = {tuple(x.shape)}")
    print(f"output.shape  = {tuple(out.shape)}    (expected ({B}, {n}, {d_model}))")
    print(f"attn.shape    = {tuple(attn.shape)}   (expected ({B}, {h}, {n}, {n}))")
    print(f"row sums (every (B, h, n) row should be 1.0):")
    print(attn.sum(dim=-1))

    # ---- Numerical equivalence: PyTorch (float64) == numpy ---------------- #
    # Goal: prove the PyTorch implementation is bit-equivalent to the numpy
    # reference, given matching weights.  We use float64 in torch to avoid
    # float32 rounding masking real bugs.
    print()
    print("=" * 60)
    print("T2.3 numerical equivalence: PyTorch (float64) == numpy")
    print("=" * 60)

    from attention_numpy import multi_head_attention  # local import

    n, d_model, h = 5, 8, 2
    X_np  = rng.standard_normal((n, d_model))
    WQ_np = rng.standard_normal((d_model, d_model))
    WK_np = rng.standard_normal((d_model, d_model))
    WV_np = rng.standard_normal((d_model, d_model))
    WO_np = rng.standard_normal((d_model, d_model))

    # numpy reference run (no batch dim)
    out_np, attn_np = multi_head_attention(X_np, WQ_np, WK_np, WV_np, WO_np, h)

    # PyTorch run with copied weights — MIND THE TRANSPOSE.
    #   numpy:  Q = X @ W_Q     W_Q shape (in, out)
    #   torch:  Q = x @ W^T     nn.Linear weight has shape (out, in)
    #           => torch weight = W_Q_np.T
    mha64 = MultiHeadSelfAttention(d_model, h, bias=False).double()
    mha64.eval()  # disable dropout
    with torch.no_grad():
        mha64.W_Q.weight.copy_(torch.from_numpy(WQ_np.T))
        mha64.W_K.weight.copy_(torch.from_numpy(WK_np.T))
        mha64.W_V.weight.copy_(torch.from_numpy(WV_np.T))
        mha64.W_O.weight.copy_(torch.from_numpy(WO_np.T))

    x_t = torch.from_numpy(X_np).unsqueeze(0)  # (1, n, d_model)
    out_t, attn_t = mha64(x_t)

    diff_out  = np.abs(out_t.squeeze(0).detach().numpy()  - out_np ).max()
    diff_attn = np.abs(attn_t.squeeze(0).detach().numpy() - attn_np).max()
    print(f"max |out_torch  - out_numpy |  = {diff_out:.2e}    (expect < 1e-12)")
    print(f"max |attn_torch - attn_numpy|  = {diff_attn:.2e}    (expect < 1e-12)")

    # ---- Mask smoke test (formal tests come in T2.4) ---------------------- #
    print()
    print("=" * 60)
    print("T2.3 mask smoke test: pad last 2 key positions")
    print("=" * 60)

    B, n, d_model, h = 1, 5, 8, 2
    x = torch.randn(B, n, d_model)
    mha = MultiHeadSelfAttention(d_model, h, bias=False)
    mha.eval()

    # True = real token, False = pad
    mask = torch.tensor([[True, True, True, False, False]])  # (B, n)
    out, attn = mha(x, mask=mask)

    last_two_col_sum = attn[..., -2:].sum().item()
    row_sums = attn.sum(dim=-1)
    print(f"attn.shape                  = {tuple(attn.shape)}")
    print(f"sum over last 2 key cols    = {last_two_col_sum:.2e}    (expect 0)")
    print(f"all row sums == 1?          = {torch.allclose(row_sums, torch.ones_like(row_sums))}")


if __name__ == "__main__":
    main()
