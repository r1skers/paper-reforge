"""
M2.1 — Image encoder for CLIP, built by composing ViT's leaf components.

为什么不直接 instantiate paper-reforge/ViT/src/vit.py 的 ViT 然后 skip head?
-----------------------------------------------------------------------
两条路:
    (a) instantiate ViT(num_classes=任意) → forward 里跳过 self.head        ← 简单但浪费一个 head 的参数
    (b) 不 import ViT 类, 直接复用它的 leaf 模块 (PatchEmbedConv, EncoderBlock)
        在本文件里自己拼装 — 没有 head, 没有冗余                              ← 我们选这个

(b) 路线的代价: __init__ 里要重复 ViT.__init__ 的几行 (CLS、PE、blocks、norm).
但好处是: 没有 wasted head 参数, 没有 monkey-patch, vit.py 完全不动.
和 M2 整体设计哲学 ("外部 wrapper, 原模块不动") 一致.

Pipeline:
    images          (B, 3, H, W)
        │  PatchEmbedConv
        ▼
    patches         (B, N, d_model)             N = (H/P)^2
        │  prepend CLS, add learned PE
        ▼
    z_0             (B, N+1, d_model)
        │  depth × EncoderBlock(bidirectional, no mask)
        ▼
    z_L             (B, N+1, d_model)
        │  final LayerNorm
        ▼
    z_L_norm        (B, N+1, d_model)
        │  take CLS slice: x[:, 0]
        ▼
    image_feat      (B, d_model)
        │  image projection: Linear(d_model, d_shared, bias=False)
        ▼
    image_embed     (B, d_shared)

跟 text encoder 镜像
--------------------
    text:   token_emb + PE → causal blocks → final_norm → x[arange(B), eos_pos] → text_proj
    image:  patch_emb + PE + CLS → bidirectional blocks → final_norm → x[:, 0]   → image_proj

唯一结构差异: causal mask vs no mask, EOS indexing vs CLS slice.
"对齐"在 d_shared 这一层第一次发生.

Run
---
    python image_encoder.py
"""

import torch
import torch.nn as nn

# 所有依赖都在本 src/ 目录 (patch_embed.py 和 transformer_block.py 都是 vendored).
# CLIP 是 self-contained 的 — 不依赖 paper-reforge 内的任何其他模块.
from patch_embed import PatchEmbedConv
from transformer_block import EncoderBlock


class ImageEncoderViT(nn.Module):
    """
    ViT-style image encoder ending in a shared-space projection (no classifier head).

    Parameters
    ----------
    img_size     : int   — input image side (assume H = W)
    patch_size   : int   — patch side P (must divide img_size)
    in_chans     : int   — 3 RGB, 1 MNIST
    d_model      : int   — internal embedding dim
    d_shared     : int   — 共享空间维度 (与 TextTransformer 的 d_shared 必须一致)
    depth        : int   — EncoderBlock 个数
    num_heads    : int
    d_ff         : int | None  — FFN 隐藏维度, None → 4 * d_model
    dropout      : float
    activation   : str   — 'gelu' (CLIP / ViT 现代约定) | 'relu'

    Forward
    -------
    images       : (B, in_chans, img_size, img_size)

    Returns
    -------
    image_embed  : (B, d_shared)   — 未 normalize, 未乘 temperature.
                                     这两步留给 M2.2 CLIPModel.
    """

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3,
                 d_model: int = 128, d_shared: int = 128,
                 depth: int = 4, num_heads: int = 4,
                 d_ff: int = None, dropout: float = 0.0,
                 activation: str = "gelu"):
        super().__init__()
        self.d_model = d_model
        self.d_shared = d_shared

        # ------------------------------------------------------------------ #
        # TODO 1 — Patch embedding.                                            #
        #                                                                     #
        #   self.patch_embed = PatchEmbedConv(img_size, patch_size,            #
        #                                     in_chans, d_model)               #
        #   N = self.patch_embed.num_patches                                    #
        #                                                                     #
        # N 待会儿 pos_embed 要用. 不存为 self.N — 直接从 patch_embed 拿即可.    #
        # ------------------------------------------------------------------ #
        self.patch_embed = PatchEmbedConv(img_size, patch_size, in_chans, d_model)

        # 在下面用 — 提前算出来:
        N = self.patch_embed.num_patches

        # ------------------------------------------------------------------ #
        # TODO 2 — CLS token (learnable, 整 batch 共享).                       #
        #                                                                     #
        #   self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))          #
        #                                                                     #
        # 形状 (1, 1, d): batch 维 broadcast, 序列维只有 1 (一张图一个 CLS).    #
        # 全零初始化在 toy-scale OK; 大模型建议 trunc_normal_(std=0.02).        #
        # ------------------------------------------------------------------ #
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # ------------------------------------------------------------------ #
        # TODO 3 — Learnable positional embedding (N + 1 slots, +1 是 CLS 的).  #
        #                                                                     #
        #   self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, d_model))      #
        #                                                                     #
        # 与 TextTransformer 的 pos_emb 镜像. 必须 nn.Parameter, 不是 buffer.   #
        # ------------------------------------------------------------------ #
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, d_model))

        # ------------------------------------------------------------------ #
        # TODO 4 — Stack of L bidirectional EncoderBlocks.                     #
        #                                                                     #
        #   self.blocks = nn.ModuleList([                                      #
        #       EncoderBlock(d_model=d_model, num_heads=num_heads,             #
        #                    d_ff=d_ff, dropout=dropout,                       #
        #                    activation=activation)                            #
        #       for _ in range(depth)                                          #
        #   ])                                                                 #
        #                                                                     #
        # 注意这里复用的是 Transformer/src/transformer_block.py 的 EncoderBlock  #
        # — bidirectional 版本, mask 参数我们 forward 里不传 (None).             #
        # ------------------------------------------------------------------ #
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, activation=activation)
            for _ in range(depth)
        ])

        # ------------------------------------------------------------------ #
        # TODO 5 — Final LayerNorm.                                            #
        #                                                                     #
        #   self.final_norm = nn.LayerNorm(d_model)                            #
        #                                                                     #
        # 跟 ViT / TextTransformer 一致, pre-norm 后处理.                       #
        # ------------------------------------------------------------------ #
        self.final_norm = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------ #
        # TODO 6 — Image projection 到共享空间.                                  #
        #                                                                     #
        #   self.image_projection = nn.Linear(d_model, d_shared, bias=False)   #
        #                                                                     #
        # bias=False — 跟 text_projection 一致 (CLIP 几何约定, M0.5).            #
        # 当 d_model == d_shared 时这看起来"多余", 但保留它有两个理由:           #
        #   1. image 和 text 的 internal d_model 不一定一致, projection 让      #
        #      它们对齐到同一个 d_shared.                                        #
        #   2. projection 是 image / text 之间唯一可以"独立调节"的桥梁,         #
        #      训练时它承担把两个 representation space 拉到对齐的工作.            #
        # ------------------------------------------------------------------ #
        self.image_projection = nn.Linear(d_model, d_shared, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # ------------------------------------------------------------------ #
        # TODO 7 — Forward pass. 几乎就是 ViT.forward, 末尾换 image_projection. #
        #                                                                     #
        #   x = self.patch_embed(images)              # (B, N, d_model)        #
        #   B = x.shape[0]                                                     #
        #   cls = self.cls_token.expand(B, -1, -1)    # (B, 1, d_model)        #
        #   x = torch.cat([cls, x], dim=1)            # (B, N+1, d_model)      #
        #   x = x + self.pos_embed                                              #
        #                                                                     #
        #   for blk in self.blocks:                                            #
        #       x = blk(x)               # 不传 mask, bidirectional             #
        #                                                                     #
        #   x = self.final_norm(x)                                              #
        #   cls_out = x[:, 0]                          # (B, d_model)           #
        #   image_embed = self.image_projection(cls_out)  # (B, d_shared)       #
        #   return image_embed                                                  #
        #                                                                     #
        # NOTE: .expand 而非 .repeat — expand 不复制内存只 restride.             #
        # cat 之后会自然 materialize, 但保持 expand 习惯, 之后传播开销低.          #
        # ------------------------------------------------------------------ #
        x = self.patch_embed(images)              # (B, N, d_model)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)            # (B, N+1, d_model)
        x = x + self.pos_embed                                              # (B, N+1, d_model) 

        for blk in self.blocks:                                            #
            x = blk(x)               # 不传 mask, bidirectional             #
        x = self.final_norm(x)                                              # (B, N+1, d_model)
        cls_out = x[:, 0]                          # (B, d_model)
        image_embed = self.image_projection(cls_out)  # (B, d_shared)
        return image_embed


# ---------------------------------------------------------------------------- #
# Driver                                                                       #
# ---------------------------------------------------------------------------- #


def main():
    torch.manual_seed(0)

    img_size, patch_size, in_chans = 32, 4, 3
    d_model, d_shared = 128, 64
    depth, num_heads = 2, 4

    model = ImageEncoderViT(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans,
        d_model=d_model, d_shared=d_shared,
        depth=depth, num_heads=num_heads,
        dropout=0.0, activation="gelu",
    )
    model.eval()

    B = 4
    images = torch.randn(B, in_chans, img_size, img_size)
    image_embed = model(images)

    N = (img_size // patch_size) ** 2
    print(f"images.shape       = {tuple(images.shape)}")
    print(f"num_patches N      = {model.patch_embed.num_patches}    (expected {N})")
    print(f"cls_token.shape    = {tuple(model.cls_token.shape)}     (expected (1, 1, {d_model}))")
    print(f"pos_embed.shape    = {tuple(model.pos_embed.shape)}     (expected (1, {N+1}, {d_model}))")
    print(f"image_embed.shape  = {tuple(image_embed.shape)}         (expected ({B}, {d_shared}))")
    print()
    total = sum(p.numel() for p in model.parameters())
    print(f"total params       = {total:,}")
    print(f"image_proj         = {model.image_projection.weight.numel():,}    "
          f"({d_shared} × {d_model}, no bias)")


if __name__ == "__main__":
    main()
