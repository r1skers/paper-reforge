"""
V2 — Vision Transformer (ViT) model.

Stages 2–5 of the ViT pipeline (stage 1 lives in patch_embed.py):

    Stage 1   image  ──PatchEmbed──>            (B, N,   d)     [patch_embed.py]
    Stage 2   prepend CLS + add PE   ──>        (B, N+1, d) = z_0
    Stage 3   L × EncoderBlock        ──>       (B, N+1, d) = z_L
    Stage 4   final LN, take z_L[:, 0]  ──>     (B, d)
    Stage 5   Linear classifier       ──>       (B, num_classes)

Reuses (no rewrite):
    EncoderBlock              from Transformer/src/transformer_block.py
    MultiHeadSelfAttention    transitively

New components introduced here:
    cls_token   : nn.Parameter, shape (1, 1, d_model)
    pos_embed   : nn.Parameter, shape (1, N+1, d_model)   — learnable 1D PE
    final norm  : nn.LayerNorm(d_model)
    head        : nn.Linear(d_model, num_classes)

Run
---
    python vit.py
"""

import sys
from pathlib import Path

# Make the Transformer module's src/ importable so we can reuse EncoderBlock.
# We APPEND (not insert at 0) so that local ViT/src/ modules win on name
# collision — Transformer/src/ also has a data.py, and if it sat at position
# 0 of sys.path it would shadow ours when train.py does `from data import ...`.
_TRANSFORMER_SRC = Path(__file__).resolve().parent.parent.parent / "Transformer" / "src"
if str(_TRANSFORMER_SRC) not in sys.path:
    sys.path.append(str(_TRANSFORMER_SRC))

import torch
import torch.nn as nn

from patch_embed import PatchEmbedConv          # local — Stage 1
from transformer_block import EncoderBlock      # reused — Stage 3


class ViT(nn.Module):
    """
    Vision Transformer for image classification.

    Parameters
    ----------
    img_size      : int        — input image side (H = W)
    patch_size    : int        — patch side P; must divide img_size
    in_chans      : int        — input channels (3 RGB, 1 MNIST)
    num_classes   : int        — number of classification labels
    d_model       : int        — token embedding dim
    depth         : int        — number of EncoderBlocks (L)
    num_heads     : int        — attention heads per block
    d_ff          : int | None — FFN hidden dim (None → 4 * d_model)
    dropout       : float      — dropout in attn / FFN / residual
    activation    : str        — 'relu' | 'gelu'  (ViT paper uses 'gelu')

    Forward
    -------
    x       : (B, C, H, W)
    returns : (B, num_classes)  — class logits
    """

    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 num_classes=10, d_model=128, depth=6, num_heads=4,
                 d_ff=None, dropout=0.1, activation='gelu'):
        super().__init__()
        self.d_model = d_model

        # ------------------------------------------------------------------ #
        # TODO 1 — Patch embedding (Stage 1).                                 #
        #                                                                     #
        #   self.patch_embed = PatchEmbedConv(img_size, patch_size,           #
        #                                     in_chans, d_model)              #
        #                                                                     #
        # Stash N for the pos_embed shape on the next TODO:                   #
        #   N = self.patch_embed.num_patches                                   #
        # ------------------------------------------------------------------ #
        self.patch_embed = PatchEmbedConv(img_size, patch_size, in_chans, d_model)
        N = self.patch_embed.num_patches
        # ------------------------------------------------------------------ #
        # TODO 2 — CLS token (learnable, shared across batch).                #
        #                                                                     #
        #   self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))         #
        #                                                                     #
        # Shape (1, 1, d):                                                    #
        #   dim 0 = 1 → broadcast/expand across batch later                   #
        #   dim 1 = 1 → exactly one CLS token per image                       #
        #   dim 2 = d → same width as patch tokens                             #
        #                                                                     #
        # All-zero init is fine for CIFAR-scale runs. For ImageNet-scale,     #
        # use nn.init.trunc_normal_(self.cls_token, std=0.02).                #
        # ------------------------------------------------------------------ #
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # ------------------------------------------------------------------ #
        # TODO 3 — Learnable 1D position embedding.                            #
        #                                                                     #
        #   self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, d_model))     #
        #                                                                     #
        # Shape (1, N+1, d) — the +1 is the CLS slot at position 0.            #
        # MUST be nn.Parameter (learnable), not register_buffer.              #
        # Same trunc_normal_ option as cls_token for larger runs.              #
        # ------------------------------------------------------------------ #
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, d_model))
        # ------------------------------------------------------------------ #
        # TODO 4 — Stack of L EncoderBlocks (Stage 3).                        #
        #                                                                     #
        #   self.blocks = nn.ModuleList([                                      #
        #       EncoderBlock(d_model=d_model, num_heads=num_heads,             #
        #                    d_ff=d_ff, dropout=dropout,                       #
        #                    activation=activation)                            #
        #       for _ in range(depth)                                          #
        #   ])                                                                 #
        #                                                                     #
        # Use nn.ModuleList, not a plain Python list — only ModuleList        #
        # registers the submodule params so the optimizer can see them.       #
        # ------------------------------------------------------------------ #
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model=d_model, num_heads=num_heads,
                         d_ff=d_ff, dropout=dropout,
                         activation=activation)
            for _ in range(depth)
        ])
        # ------------------------------------------------------------------ #
        # TODO 5 — Final LayerNorm.                                           #
        #                                                                     #
        #   self.norm = nn.LayerNorm(d_model)                                  #
        #                                                                     #
        # Pre-norm puts LN INSIDE each EncoderBlock's residual, so the last   #
        # block's OUTPUT is not normed. This extra LN cleans that up before   #
        # the classifier (corresponds to LN in eqn (4) of the ViT paper).     #
        # ------------------------------------------------------------------ #
        self.norm = nn.LayerNorm(d_model)
        # ------------------------------------------------------------------ #
        # TODO 6 — Classification head (Stage 5).                              #
        #                                                                     #
        #   self.head = nn.Linear(d_model, num_classes)                       #
        #                                                                     #
        # Just one Linear. ViT deliberately drops BERT's tanh pooler — the    #
        # CLS output after L layers of attention is already linearly          #
        # separable enough, no extra non-linearity needed.                    #
        # ------------------------------------------------------------------ #
        self.head = nn.Linear(d_model, num_classes)
    def forward(self, x):
        # ------------------------------------------------------------------ #
        # Stage 1 — PatchEmbed.                                                #
        #                                                                     #
        #   x = self.patch_embed(x)               # (B, C, H, W) → (B, N, d)   #
        # ------------------------------------------------------------------ #
        x = self.patch_embed(x)
        # ------------------------------------------------------------------ #
        # Stage 2 — Prepend CLS, add PE.                                       #
        #                                                                     #
        #   B = x.shape[0]                                                     #
        #   cls = self.cls_token.expand(B, -1, -1)        # (1,1,d) → (B,1,d) #
        #   x   = torch.cat([cls, x], dim=1)              # (B, N+1, d)       #
        #   x   = x + self.pos_embed                      # broadcast over B  #
        #                                                                     #
        # Use .expand NOT .repeat:                                             #
        #   - expand : no copy, just restride. Cheap.                          #
        #   - repeat : actually copies memory B times. Wasteful.               #
        # (Side note: torch.cat will materialize a copy anyway, but the       #
        #  expand-first habit pays off in other places.)                       #
        # ------------------------------------------------------------------ #
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        # ------------------------------------------------------------------ #
        # Stage 3 — L × EncoderBlock.                                          #
        #                                                                     #
        #   for blk in self.blocks:                                            #
        #       x = blk(x)                            # (B, N+1, d) unchanged  #
        #                                                                     #
        # No mask passed — for image classification every token attends to    #
        # every other token (including CLS attending to all patches and       #
        # vice versa).                                                         #
        # ------------------------------------------------------------------ #
        for blk in self.blocks:
            x = blk(x)
        # ------------------------------------------------------------------ #
        # Stage 4 — Final LN, take CLS.                                        #
        #                                                                     #
        #   x = self.norm(x)                          # (B, N+1, d)            #
        #   cls_out = x[:, 0]                         # (B, d) — index 0 = CLS #
        # ------------------------------------------------------------------ #
        x = self.norm(x)
        # ------------------------------------------------------------------ #
        # Stage 5 — Classifier.                                                #
        #                                                                     #
        #   logits = self.head(cls_out)               # (B, num_classes)       #
        #   return logits                                                      #
        # ------------------------------------------------------------------ #
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return logits

# ---------------------------------------------------------------------------- #
# Driver: quick shape + param sanity                                           #
# ---------------------------------------------------------------------------- #


def main():
    torch.manual_seed(0)

    # CIFAR-10 sized config
    img_size, patch_size, in_chans = 32, 4, 3
    num_classes = 10
    d_model, depth, num_heads = 128, 6, 4

    model = ViT(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                num_classes=num_classes, d_model=d_model, depth=depth,
                num_heads=num_heads, dropout=0.1, activation='gelu')
    model.eval()

    B = 2
    x = torch.randn(B, in_chans, img_size, img_size)
    logits = model(x)

    N = (img_size // patch_size) ** 2
    print(f"input.shape         = {tuple(x.shape)}")
    print(f"logits.shape        = {tuple(logits.shape)}    (expected ({B}, {num_classes}))")
    print(f"num_patches N       = {model.patch_embed.num_patches}    (expected {N})")
    print(f"cls_token.shape     = {tuple(model.cls_token.shape)}    (expected (1, 1, {d_model}))")
    print(f"pos_embed.shape     = {tuple(model.pos_embed.shape)}    (expected (1, {N+1}, {d_model}))")
    print(f"depth (blocks)      = {len(model.blocks)}    (expected {depth})")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params        = {total:,}")
    print(f"trainable params    = {trainable:,}    (should equal total — no buffers here)")


if __name__ == "__main__":
    main()
