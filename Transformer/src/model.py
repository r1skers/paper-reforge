"""
T4.1 — Full encoder-only Transformer for the argmax-position task.

Architecture
------------
    int tokens (B, n)
        -> nn.Embedding(vocab_size -> d_model)        : (B, n, d_model)
        -> [+ SinusoidalPositionalEncoding]            : (B, n, d_model)
        -> N x EncoderBlock                            : (B, n, d_model)
        -> final LayerNorm                             : (B, n, d_model)
        -> score head nn.Linear(d_model -> 1)          : (B, n, 1)
        -> squeeze(-1)                                 : (B, n)   <- logits over positions

The output is a per-position scalar.  Training uses F.cross_entropy on
these logits with the true argmax index as the target — this is
essentially a "pointer" head, identical in spirit to Pointer Networks.

Run
---
    python model.py
"""

import torch
import torch.nn as nn

from positional_encoding import SinusoidalPositionalEncoding
from transformer_block import EncoderBlock


class ArgmaxPositionModel(nn.Module):
    """
    Encoder-only Transformer that points to the position of the argmax
    in an integer sequence.

    Parameters
    ----------
    vocab_size : int
        Size of the input vocabulary (e.g., 10 for digits 0-9).
    n_max : int
        Maximum sequence length (used to size the PE buffer).
    d_model    : int, default 64
    num_heads  : int, default 4
    num_layers : int, default 2
    d_ff       : int or None, default None (-> 4 * d_model)
    dropout    : float, default 0.1
    use_pe     : bool, default True
        Set False for the "no positional encoding" ablation in T4.4.

    Forward
    -------
    x    : LongTensor (B, n)         integer token ids
    mask : BoolTensor (B, n) or None  True = real token, False = pad
    returns : (B, n)  per-position logits
    """

    def __init__(self, vocab_size, n_max, d_model=64, num_heads=4,
                 num_layers=2, d_ff=None, dropout=0.1, use_pe=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_max = n_max
        self.d_model = d_model
        self.use_pe = use_pe

        # ------------------------------------------------------------------ #
        # TODO 1 — Create the submodules.                                     #
        #                                                                     #
        # self.token_emb  = nn.Embedding(vocab_size, d_model)                  #
        # self.pe         = SinusoidalPositionalEncoding(d_model, n_max)       #
        #                   if use_pe else None                                #
        # self.blocks     = nn.ModuleList([                                    #
        #                       EncoderBlock(d_model, num_heads, d_ff,         #
        #                                    dropout=dropout)                  #
        #                       for _ in range(num_layers)                     #
        #                   ])                                                 #
        # self.final_norm = nn.LayerNorm(d_model)                               #
        # self.score_head = nn.Linear(d_model, 1)                               #
        #                                                                     #
        # NOTE: nn.ModuleList (NOT a plain Python list) — only ModuleList      #
        # registers child modules, so .parameters() and .to(device) reach them.#
        # ------------------------------------------------------------------ #
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pe = SinusoidalPositionalEncoding(d_model, n_max) if use_pe else None
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.score_head = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        # ------------------------------------------------------------------ #
        # TODO 2 — Implement the forward pipeline.                            #
        #                                                                     #
        # h = self.token_emb(x)                                                #
        # if self.pe is not None:                                              #
        #     h = self.pe(h)                                                   #
        # for block in self.blocks:                                            #
        #     h = block(h, mask=mask)                                          #
        # h = self.final_norm(h)               # final LN required for pre-norm#
        # logits = self.score_head(h).squeeze(-1)   # (B, n, 1) -> (B, n)       #
        # return logits                                                         #
        # ------------------------------------------------------------------ #
        h = self.token_emb(x)
        if self.pe is not None:
            h = self.pe(h)
        for block in self.blocks:
            h = block(h, mask=mask)
        h = self.final_norm(h)
        logits = self.score_head(h).squeeze(-1)
        return logits


def main():
    """Forward-only smoke test."""
    torch.manual_seed(0)

    vocab_size = 10     # digits 0..9
    n_max = 32
    B, n = 4, 20

    model = ArgmaxPositionModel(
        vocab_size=vocab_size,
        n_max=n_max,
        d_model=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    model.eval()

    x = torch.randint(0, vocab_size, (B, n))
    logits = model(x)
    print(f"x.shape       = {tuple(x.shape)}")
    print(f"logits.shape  = {tuple(logits.shape)}    (expected ({B}, {n}))")
    print(f"argmax pred   = {logits.argmax(dim=-1).tolist()}")
    print(f"true argmax   = {x.argmax(dim=-1).tolist()}    (model is untrained so these should NOT match)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count   = {n_params:,}")

    # Quick ablation visibility: no-PE variant should also forward cleanly
    model_nope = ArgmaxPositionModel(
        vocab_size=vocab_size, n_max=n_max,
        d_model=32, num_heads=4, num_layers=2, dropout=0.0,
        use_pe=False,
    )
    model_nope.eval()
    logits_nope = model_nope(x)
    print(f"\nno-PE variant logits.shape = {tuple(logits_nope.shape)}    (expected ({B}, {n}))")
    print(f"no-PE pe attribute = {model_nope.pe}    (expected None)")


if __name__ == "__main__":
    main()
