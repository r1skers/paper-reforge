"""
Position-wise FFN + pre-norm Transformer encoder block (bidirectional).

VENDORED
--------
This is a verbatim copy of paper-reforge/Transformer/src/transformer_block.py
brought into CLIP/src/ to keep CLIP self-contained.

CLIP usage
----------
- PositionWiseFFN : used inside CLIP's CAUSAL block (CausalEncoderBlock
                    in text_encoder.py) — the FFN is mask-agnostic so
                    we directly reuse it.
- EncoderBlock    : used by CLIP's IMAGE encoder for the bidirectional
                    stack (image_encoder.py).

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
    Two-layer MLP applied independently to every position.

        FFN(x) = Dropout(activation(x W_1 + b_1)) W_2 + b_2

    Parameters
    ----------
    d_model    : int
    d_ff       : int or None — None defaults to 4 * d_model (paper convention)
    dropout    : float
    activation : {'relu', 'gelu'}

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
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderBlock(nn.Module):
    """
    Pre-norm Transformer encoder block (bidirectional self-attention).

        y   = x + Dropout(MHA(LayerNorm(x), mask))
        out = y + Dropout(FFN(LayerNorm(y)))

    The residual path is an identity-highway — that's why pre-norm
    trains at depth without warmup.

    Parameters
    ----------
    d_model    : int
    num_heads  : int
    d_ff       : int or None
    dropout    : float — applied (i) inside attn weights, (ii) inside FFN,
                         (iii) on each sublayer output before the residual add.
    activation : {'relu', 'gelu'}

    Forward
    -------
    x    : (B, n, d_model)
    mask : None or (B, n) BoolTensor — True=keep, False=pad
    returns : (B, n, d_model)
    """

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout, bias=False)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout=dropout, activation=activation)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(self.norm1(x), mask=mask)
        x = x + self.resid_dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.resid_dropout(ffn_out)
        return x


def main():
    torch.manual_seed(0)
    B, n, d_model = 2, 5, 16
    block = EncoderBlock(d_model=d_model, num_heads=2, dropout=0.0)
    block.eval()
    x = torch.randn(B, n, d_model)
    out = block(x)
    print(f"block(x).shape = {tuple(out.shape)}    (expected {(B, n, d_model)})")


if __name__ == "__main__":
    main()
