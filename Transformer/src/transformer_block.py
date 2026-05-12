"""
T3.2 — Position-wise feed-forward network (FFN).
T3.3 — Encoder block (pre-norm) — coming after T3.2.

This file will eventually hold both the FFN and the EncoderBlock so that
they live together as 'the things on top of attention'.  T3.2 only fills
in PositionWiseFFN; EncoderBlock stays a stub for now.

Run
---
    python transformer_block.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadSelfAttention


class PositionWiseFFN(nn.Module):
    """
    Two-layer MLP applied independently to every position
    ('position-wise' = same MLP shared across the sequence dim).

        FFN(x) = Dropout(activation(x W_1 + b_1)) W_2 + b_2

    Parameters
    ----------
    d_model : int
        Input / output dim.
    d_ff : int or None, default None
        Hidden dim.  If None, defaults to 4 * d_model (paper convention).
    dropout : float, default 0.1
        Dropout probability applied AFTER the activation.
    activation : {'relu', 'gelu'}, default 'relu'
        Original paper uses ReLU; BERT / GPT / ViT use GeLU.

    Forward
    -------
    x : (B, n, d_model)
    returns : (B, n, d_model)
    """

    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff

        # ------------------------------------------------------------------ #
        # TODO 1 — Create the two Linear layers + dropout + activation fn.   #
        #                                                                     #
        # self.fc1       : nn.Linear(d_model, d_ff)    — has bias by default   #
        # self.fc2       : nn.Linear(d_ff,    d_model)                         #
        # self.dropout   : nn.Dropout(dropout)                                 #
        # self.activation: F.relu if activation == "relu" else F.gelu          #
        #                                                                     #
        # We store activation as a Python function (not nn.Module) so the     #
        # forward stays one straight pipeline.                                #
        # ------------------------------------------------------------------ #
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        # ------------------------------------------------------------------ #
        # TODO 2 — Implement the four-step pipeline.                          #
        #                                                                     #
        #   x = self.fc1(x)            ->  (B, n, d_ff)                       #
        #   x = self.activation(x)     ->  (B, n, d_ff)                       #
        #   x = self.dropout(x)        ->  (B, n, d_ff)                       #
        #   x = self.fc2(x)            ->  (B, n, d_model)                    #
        #   return x                                                          #
        #                                                                     #
        # NOTE: dropout goes BETWEEN activation and fc2 (inside-FFN style).   #
        # The dropout AFTER the residual add lives in the EncoderBlock,       #
        # not here.                                                           #
        # ------------------------------------------------------------------ #
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderBlock(nn.Module):
    """
    T3.3 — Pre-norm Transformer encoder block.

        y   = x + Dropout(MHA(LayerNorm(x), mask))
        out = y + Dropout(FFN(LayerNorm(y)))

    The residual path (the bare `x +`) is an identity — no LayerNorm sits
    on top of it.  That is the key property that makes pre-norm trainable
    at depth without warmup (see Day 3 notes).

    Parameters
    ----------
    d_model     : int   — embedding dim
    num_heads   : int   — MHA heads
    d_ff        : int   — FFN hidden dim (None => 4 * d_model)
    dropout     : float — applied at three places:
                          (i) inside attn on the attention weights,
                          (ii) inside FFN between activation and fc2,
                          (iii) on each sublayer output BEFORE the residual add.
    activation  : 'relu' | 'gelu'

    Forward
    -------
    x    : (B, n, d_model)
    mask : None or (B, n) bool tensor
    returns : (B, n, d_model)
    """

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()

        # ------------------------------------------------------------------ #
        # TODO 1 — Create the 5 submodules.                                   #
        #                                                                     #
        # self.norm1  = nn.LayerNorm(d_model)                                  #
        # self.norm2  = nn.LayerNorm(d_model)                                  #
        # self.attn   = MultiHeadSelfAttention(d_model, num_heads,             #
        #                                     dropout=dropout, bias=False)   #
        # self.ffn    = PositionWiseFFN(d_model, d_ff,                         #
        #                               dropout=dropout, activation=activation) #
        # self.resid_dropout = nn.Dropout(dropout)                             #
        #                                                                     #
        # NOTE: two LayerNorms (one per sublayer) — they are independent       #
        # modules with their own learnable γ, β.  Sharing one would be wrong. #
        # ------------------------------------------------------------------ #
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout, bias=False)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout=dropout, activation=activation)
        self.resid_dropout = nn.Dropout(dropout)    

    def forward(self, x, mask=None):
        # ------------------------------------------------------------------ #
        # TODO 2 — Implement the pre-norm two-sublayer flow.                  #
        #                                                                     #
        # Sublayer 1 (attention):                                              #
        #   attn_out, _ = self.attn(self.norm1(x), mask=mask)                  #
        #   x = x + self.resid_dropout(attn_out)                               #
        #                                                                     #
        # Sublayer 2 (FFN):                                                    #
        #   ffn_out = self.ffn(self.norm2(x))                                  #
        #   x = x + self.resid_dropout(ffn_out)                                #
        #                                                                     #
        # return x                                                             #
        #                                                                     #
        # We discard the attention weights here (the `_`).  If you later      #
        # want them for visualization, expose via a debug flag.               #
        # ------------------------------------------------------------------ #
        attn_out, _ = self.attn(self.norm1(x), mask=mask)
        x = x + self.resid_dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.resid_dropout(ffn_out)
        return x


# ---------------------------------------------------------------------------- #
# Driver: quick FFN sanity                                                     #
# ---------------------------------------------------------------------------- #


def main():
    torch.manual_seed(0)

    B, n, d_model = 2, 5, 16
    ffn = PositionWiseFFN(d_model, dropout=0.1)
    ffn.eval()  # disable dropout for shape print

    x = torch.randn(B, n, d_model)
    y = ffn(x)
    print(f"x.shape        = {tuple(x.shape)}")
    print(f"y.shape        = {tuple(y.shape)}    (expected {(B, n, d_model)})")
    print(f"d_ff (default) = {ffn.d_ff}    (expected {4 * d_model})")

    # Param count: 2 weights + 2 biases
    n_params = sum(p.numel() for p in ffn.parameters())
    expected = d_model * (4 * d_model) + (4 * d_model)   # fc1
    expected += (4 * d_model) * d_model + d_model         # fc2
    print(f"param count    = {n_params}    (expected {expected})")

    # GELU variant
    ffn_gelu = PositionWiseFFN(d_model, activation="gelu")
    ffn_gelu.eval()
    y2 = ffn_gelu(x)
    print(f"gelu out.shape = {tuple(y2.shape)}")
    print(f"relu vs gelu differ?  {not torch.allclose(y, y2)}    (expected True)")

    # ---- EncoderBlock sanity ---------------------------------------------- #
    print()
    print("=" * 60)
    print("T3.3 EncoderBlock sanity")
    print("=" * 60)
    block = EncoderBlock(d_model=d_model, num_heads=2, dropout=0.1)
    block.eval()
    out = block(x)
    print(f"block(x).shape = {tuple(out.shape)}    (expected {(B, n, d_model)})")
    print(f"block param count = {sum(p.numel() for p in block.parameters())}")

    # Identity-highway demo: zero out all sublayer weights AND biases,
    # block output must equal input (pre-norm residual passes through).
    block_zero = EncoderBlock(d_model=d_model, num_heads=2, dropout=0.0)
    block_zero.eval()
    with torch.no_grad():
        for name, p in block_zero.named_parameters():
            if name.startswith("attn.") or name.startswith("ffn."):
                p.zero_()
    out_zero = block_zero(x)
    max_diff = (out_zero - x).abs().max().item()
    print(f"zero-sublayer identity check:  max|out - x| = {max_diff:.2e}    (expected ~0)")


if __name__ == "__main__":
    main()
