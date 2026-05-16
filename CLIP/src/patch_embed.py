"""
Image patch embedding via non-overlapping Conv2d.

VENDORED
--------
This is a stripped copy of paper-reforge/ViT/src/patch_embed.py — only
PatchEmbedConv (the production form used by CLIP's image encoder) is kept.
PatchEmbedUnfold (the math-faithful "slice → flatten → linear" version)
is omitted because CLIP doesn't use it. See the original ViT module for
the equivalence proof.

CLIP usage
----------
image_encoder.py uses PatchEmbedConv as Stage 1 of the ViT-style image tower.

Run
---
    python patch_embed.py
"""

import torch
import torch.nn as nn


class PatchEmbedConv(nn.Module):
    """
    PatchEmbed via a single Conv2d(kernel=P, stride=P).

    stride == kernel_size is what makes patches NON-OVERLAPPING — turns
    the conv from a feature extractor into a tokenizer.

    Parameters
    ----------
    img_size   : int   — H = W = img_size
    patch_size : int   — patch side P (must divide img_size)
    in_chans   : int   — 3 RGB, 1 MNIST
    d_model    : int   — output token dim

    Forward
    -------
    x : (B, C, img_size, img_size)
    returns : (B, N, d_model)   where N = (img_size // P) ** 2
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, d_model=768):
        super().__init__()
        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.d_model = d_model
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, d_model,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)             # (B, d_model, H/P, W/P)
        x = x.flatten(2)             # (B, d_model, N)
        x = x.transpose(1, 2)        # (B, N, d_model)
        return x


def main():
    torch.manual_seed(0)
    B = 2
    conv = PatchEmbedConv(img_size=32, patch_size=4, in_chans=3, d_model=128)
    conv.eval()
    x = torch.randn(B, 3, 32, 32)
    y = conv(x)
    print(f"input.shape   = {tuple(x.shape)}")
    print(f"output.shape  = {tuple(y.shape)}    (expected {(B, 64, 128)})")
    print(f"num_patches   = {conv.num_patches}    (expected 64)")


if __name__ == "__main__":
    main()
