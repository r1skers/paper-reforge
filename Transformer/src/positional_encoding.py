"""
T3.1 — Sinusoidal positional encoding.

From 'Attention Is All You Need' (Vaswani et al., 2017):

    PE[pos, 2i  ] = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

Even dims use sin, odd dims use cos, paired so that each (2i, 2i+1) couple
forms one 2D rotation frequency.  Higher dim index => lower frequency.

Run
---
    python positional_encoding.py
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-trainable) sinusoidal positional encoding.

    Parameters
    ----------
    d_model : int
        Embedding dimension.  Must be even (sin/cos paired).
    max_len : int, default 5000
        Maximum sequence length we precompute PE for.  Forward will assert
        the input length <= max_len.

    Forward
    -------
    x : (B, n, d_model)
        Token embeddings.
    returns : (B, n, d_model)
        x with the first n rows of the PE table added position-wise.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even (sin/cos paired)"
        self.d_model = d_model
        self.max_len = max_len

        # ------------------------------------------------------------------ #
        # TODO 1 — Precompute the PE table of shape (max_len, d_model).       #
        #                                                                     #
        # Step A.  position vector:                                           #
        #   pos = torch.arange(max_len).unsqueeze(1).float()   # (max_len, 1) #
        #                                                                     #
        # Step B.  frequencies — one per (2i, 2i+1) pair.                     #
        #   div_term = torch.exp(                                             #
        #       -math.log(10000.0)                                            #
        #       * torch.arange(0, d_model, 2).float()                         #
        #       / d_model                                                     #
        #   )                            # shape: (d_model / 2,)              #
        #                                                                     #
        #   This computes  1 / 10000^(2i / d_model)  in log-space, which is   #
        #   numerically stable and is the form every reference impl uses.    #
        #                                                                     #
        # Step C.  fill the table:                                            #
        #   pe = torch.zeros(max_len, d_model)                                #
        #   pe[:, 0::2] = torch.sin(pos * div_term)   # even cols: sin        #
        #   pe[:, 1::2] = torch.cos(pos * div_term)   # odd  cols: cos        #
        #                                                                     #
        # Broadcasting: pos is (max_len, 1), div_term is (d_model/2,),        #
        # product is (max_len, d_model/2).  This is the heart of the          #
        # "vectorized vs nested-loop" win.                                   #
        # ------------------------------------------------------------------ #
        pos = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, d_model, 2).float()
            / d_model
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        # ------------------------------------------------------------------ #
        # TODO 2 — Register pe as a BUFFER, not a Parameter.                  #
        #                                                                     #
        # A buffer is part of the module state — saved/loaded with the model, #
        # moved to .cuda() with the model — but receives no gradients.        #
        # Sinusoidal PE is fixed by the paper, so this is exactly right.     #
        #                                                                     #
        #   self.register_buffer('pe', pe)                                     #
        # ------------------------------------------------------------------ #
        self.register_buffer('pe', pe)

    def forward(self, x):
        # ------------------------------------------------------------------ #
        # TODO 3 — Add the first n rows of self.pe onto x.                    #
        #                                                                     #
        # n = x.size(1)                                                       #
        # assert n <= self.max_len, "input longer than precomputed PE table"  #
        # Broadcasting: pe[:n] is (n, d_model); x is (B, n, d_model).         #
        # PyTorch broadcasts the missing batch dim automatically.             #
        #                                                                     #
        #   return x + self.pe[:n]                                            #
        # ------------------------------------------------------------------ #

        n = x.size(1)
        assert n <= self.max_len, "input longer than precomputed PE table"
        return x + self.pe[:n]


# ---------------------------------------------------------------------------- #
# Driver: quick eyeball sanity                                                 #
# ---------------------------------------------------------------------------- #


def main():
    torch.manual_seed(0)
    d_model = 16
    max_len = 100
    pe = SinusoidalPositionalEncoding(d_model, max_len)

    print(f"pe.shape           = {tuple(pe.pe.shape)}    (expected ({max_len}, {d_model}))")
    print(f"trainable params   = {sum(p.numel() for p in pe.parameters())}    (expected 0 — pe is a buffer)")

    # Forward shape
    B, n = 2, 10
    x = torch.zeros(B, n, d_model)
    out = pe(x)
    print(f"forward out.shape  = {tuple(out.shape)}    (expected ({B}, {n}, {d_model}))")
    print(f"x=0 -> out == pe[:n] ? {torch.allclose(out[0], pe.pe[:n])}")

    # Position 0 sanity: sin(0)=0 on even dims, cos(0)=1 on odd dims
    p0 = pe.pe[0]
    print()
    print(f"PE[0] even dims (should be all 0):  {p0[0::2].tolist()}")
    print(f"PE[0] odd  dims (should be all 1):  {p0[1::2].tolist()}")

    # First few rows, first 8 dims, for eyeballing
    print()
    print(f"PE[:3, :8] (first 3 positions, first 8 dims):")
    print(pe.pe[:3, :8])


if __name__ == "__main__":
    main()
