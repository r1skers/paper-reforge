"""
V1 — Patch embedding (the only "new" input-side module ViT adds on top of
NLP Transformer).

Two equivalent implementations, both producing (B, N, d_model) tokens
from an input image (B, C, H, W):

    PatchEmbedConv    : nn.Conv2d(kernel=P, stride=P) — the standard,
                        production-grade, one-liner version.
    PatchEmbedUnfold  : F.unfold(kernel=P, stride=P) + nn.Linear — the
                        "math-faithful" version, makes the
                          'slice → flatten → linear project'
                        three-step decomposition explicit.

Equivalence under weight copy:
    With  W_lin = W_conv.flatten(1)  and  b_lin = b_conv  ,
    both implementations are numerically bit-exact (up to fp precision).
    A test in tests/test_patch_embed.py asserts this with float64.

This file does NOT touch CLS / pos_embed — those live in vit.py.
PatchEmbed only handles the   image → token sequence   step.

Run
---
    python patch_embed.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedConv(nn.Module):
    """
    PatchEmbed via a single Conv2d(kernel=P, stride=P).

    This is the form everyone ships in production (timm, torchvision, etc.).
    It is one nn.Conv2d call followed by a flatten+transpose.

    Parameters
    ----------
    img_size   : int   — input image side length (H = W = img_size assumed)
    patch_size : int   — patch side length P; must divide img_size
    in_chans   : int   — input channels (3 for RGB, 1 for MNIST)
    d_model    : int   — output token dim

    Forward
    -------
    x : (B, C, img_size, img_size)
    returns : (B, N, d_model)  where N = (img_size // P) ** 2
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

        # ------------------------------------------------------------------ #
        # TODO 1 — Create the conv projection.                                #
        #                                                                     #
        #   self.proj = nn.Conv2d(in_chans, d_model,                          #
        #                         kernel_size=patch_size,                     #
        #                         stride=patch_size)                          #
        #                                                                     #
        # KEY POINT: stride == kernel_size. This is what makes the patches    #
        # NON-OVERLAPPING and turns the conv from a "feature extractor"       #
        # into a "tokenizer". Default bias=True is fine (timm matches).       #
        # ------------------------------------------------------------------ #
        self.proj = nn.Conv2d(in_chans, d_model, kernel_size=patch_size, stride=patch_size)
        # self.proj = nn.Linear(in_chans * patch_size * patch_size, d_model)
        # 这两种实现方式在数学上是等价的，卷积层的功能可以通过一个线性层来实现。卷积层的尺寸是(d_model, in_chans, patch_size, patch_size)，当我们将其展平后得到(d_model, in_chans * patch_size * patch_size)，这正好是线性层的权重尺寸(d_model, patch_dim)。线性层的输入尺寸是(patch_dim = in_chans * patch_size * patch_size)，输出尺寸是d_model，这与卷积层的输入输出尺寸完全对应。因此，通过一个线性映射，我们可以实现卷积层的功能。
        # 数据量也是一样的，卷积层的参数量是d_model * in_chans * patch_size * patch_size + d_model（权重和偏置），线性层的参数量也是d_model * (in_chans * patch_size * patch_size) + d_model（权重和偏置）。因此，这两种实现方式在参数量上是等价的。
    def forward(self, x):
        # ------------------------------------------------------------------ #
        # TODO 2 — Three-step forward.                                        #
        #                                                                     #
        # Input:                                                              #
        #   x : (B, C, H, W)   with  H = W = img_size                         #
        #                                                                     #
        # Pipeline:                                                           #
        #   x = self.proj(x)            # (B, d_model, H/P, W/P)              #
        #   x = x.flatten(2)            # (B, d_model, N)                     #
        #   x = x.transpose(1, 2)       # (B, N, d_model)                     #
        #   return x                                                          #
        #                                                                     #
        # NOTE: shape sanity check is optional — the conv itself will         #
        # complain if H/W are wrong sizes.                                    #
        # ------------------------------------------------------------------ #
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class PatchEmbedUnfold(nn.Module):
    """
    PatchEmbed via F.unfold(kernel=P, stride=P) + nn.Linear.

    Same math as PatchEmbedConv, but the "slice → flatten → linear"
    decomposition is now three explicit operations. Useful for:
      (a) teaching: makes the math fully transparent,
      (b) the bit-exact equivalence test (a sanity check that you
          understand what Conv2d-with-stride=kernel actually computes).

    Parameters
    ----------
    Same as PatchEmbedConv.

    Forward
    -------
    Same I/O as PatchEmbedConv.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, d_model=768):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.d_model = d_model
        self.num_patches = (img_size // patch_size) ** 2

        # ------------------------------------------------------------------ #
        # TODO 1 — Create the linear projection.                              #
        #                                                                     #
        #   patch_dim = in_chans * patch_size * patch_size                    #
        #   self.proj = nn.Linear(patch_dim, d_model)                         #
        #                                                                     #
        # Default bias=True. The weight will be (d_model, patch_dim) — this   #
        # is exactly W_conv.flatten(1) up to a copy, which is what makes      #
        # the two implementations bit-exact under weight transfer.            #
        # ------------------------------------------------------------------ #
        patch_dim = in_chans * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, d_model)
    def forward(self, x):
        # ------------------------------------------------------------------ #
        # TODO 2 — Three-step forward using F.unfold.                         #
        #                                                                     #
        # Input:                                                              #
        #   x : (B, C, H, W)                                                  #
        #                                                                     #
        # Step A — slice into non-overlapping patches:                        #
        #   patches = F.unfold(x,                                             #
        #                      kernel_size=self.patch_size,                   #
        #                      stride=self.patch_size)                        #
        #   # shape: (B, C * P * P, N)                                        #
        #   #                                                                 #
        #   # F.unfold flattens each patch into a single column.              #
        #   # The C*P*P axis is in CHANNEL-MAJOR, ROW-MAJOR order — exactly   #
        #   # the same flattening order PyTorch uses for Conv2d weight.       #
        #   # That alignment is WHY the bit-exact test below works.           #
        #                                                                     #
        # Step B — transpose so tokens are the second axis:                   #
        #   patches = patches.transpose(1, 2)                                 #
        #   # shape: (B, N, C * P * P)                                        #
        #                                                                     #
        # Step C — linear project to d_model:                                 #
        #   tokens = self.proj(patches)                                       #
        #   # shape: (B, N, d_model)                                          #
        #                                                                     #
        # return tokens                                                       #
        # ------------------------------------------------------------------ #
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2)
        tokens = self.proj(patches)
        return tokens


# ---------------------------------------------------------------------------- #
# Driver: quick shape + param sanity                                           #
# ---------------------------------------------------------------------------- #


def main():
    torch.manual_seed(0)

    # ImageNet-Base style config
    img_size, patch_size, in_chans, d_model = 224, 16, 3, 768
    B = 2

    conv = PatchEmbedConv(img_size, patch_size, in_chans, d_model)
    unfold = PatchEmbedUnfold(img_size, patch_size, in_chans, d_model)
    conv.eval()
    unfold.eval()

    x = torch.randn(B, in_chans, img_size, img_size)

    y_conv = conv(x)
    y_unfold = unfold(x)

    print(f"input.shape       = {tuple(x.shape)}")
    print(f"conv  out.shape   = {tuple(y_conv.shape)}    (expected {(B, 196, d_model)})")
    print(f"unfold out.shape  = {tuple(y_unfold.shape)}    (expected {(B, 196, d_model)})")

    # Param count: both should be d_model * (C*P*P) + d_model
    conv_params = sum(p.numel() for p in conv.parameters())
    unfold_params = sum(p.numel() for p in unfold.parameters())
    expected = d_model * (in_chans * patch_size * patch_size) + d_model
    print(f"conv  param count = {conv_params}    (expected {expected})")
    print(f"unfold param count = {unfold_params}    (expected {expected})")

    # CIFAR-10 style sanity
    print()
    print("=" * 60)
    print("CIFAR-10 style: 32x32 image, patch=4 -> 64 tokens")
    print("=" * 60)
    conv32 = PatchEmbedConv(img_size=32, patch_size=4, in_chans=3, d_model=128)
    conv32.eval()
    x32 = torch.randn(B, 3, 32, 32)
    y32 = conv32(x32)
    print(f"out.shape         = {tuple(y32.shape)}    (expected {(B, 64, 128)})")
    print(f"num_patches       = {conv32.num_patches}    (expected 64)")


if __name__ == "__main__":
    main()
