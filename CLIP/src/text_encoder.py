"""
M1.3 — Text encoder for CLIP (causal Transformer + EOS feature extraction).

Pipeline:
    token_ids       (B, L)
        │  token_embedding + learned positional embedding
        ▼
    z_0             (B, L, d_model)
        │  N × CausalEncoderBlock(causal_mask)
        ▼
    z_N             (B, L, d_model)
        │  final LayerNorm
        ▼
    z_N_norm        (B, L, d_model)
        │  pick EOS position: x[arange(B), eos_pos]
        ▼
    sentence_feat   (B, d_model)
        │  text projection: Linear(d_model, d_shared, bias=False)
        ▼
    text_embed      (B, d_shared)

Design notes
------------
- Position embedding is LEARNED (nn.Parameter), 沿用 CLIP / ViT 的现代做法.
  Transformer 模块用的是 sinusoidal (固定) — 这里换 learnable, M0.4 讨论过.
- Final LayerNorm 跟 ViT 一致 — pre-norm 架构最后一层 block 输出未归一化,
  需要补一个 final norm 再喂 projection.
- text_projection 用 bias=False, 这是 CLIP 原版 convention (M0.5 提过).
  几何理由: 共享空间里我们关心方向, 不关心偏移.

Run
---
    python text_encoder.py
"""

import torch
import torch.nn as nn

# 所有依赖都在本 src/ 目录 (含 vendored 的 transformer_block.py),
# 不再跨模块 sys.path 操作 — CLIP 是 self-contained 的.
from transformer_block import PositionWiseFFN
from causal_attention import (
    CausalMultiHeadSelfAttention,
    build_causal_mask,
)


class CausalEncoderBlock(nn.Module):
    """
    Pre-norm Transformer encoder block with causal self-attention.

    跟 Transformer/src/transformer_block.py 的 EncoderBlock 结构一样,
    只是 attention 子层换成 CausalMultiHeadSelfAttention, 接受 additive mask.

        y   = x + Dropout(CausalMHA(LayerNorm(x), attn_mask))
        out = y + Dropout(FFN(LayerNorm(y)))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int = None,
                 dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()

        # ------------------------------------------------------------------ #
        # TODO 1 — 5 个子模块, 跟 EncoderBlock 镜像但 attention 换成 causal 版.  #
        #                                                                     #
        #   self.norm1 = nn.LayerNorm(d_model)                                 #
        #   self.norm2 = nn.LayerNorm(d_model)                                 #
        #   self.attn  = CausalMultiHeadSelfAttention(                         #
        #                   d_model, num_heads,                                #
        #                   dropout=dropout, bias=True)                        #
        #   self.ffn   = PositionWiseFFN(d_model, d_ff,                        #
        #                                dropout=dropout, activation=activation) #
        #   self.resid_dropout = nn.Dropout(dropout)                           #
        #                                                                     #
        # NOTE: CLIP 原版用 GeLU activation. 默认值已经是 "gelu".              #
        # ------------------------------------------------------------------ #
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = CausalMultiHeadSelfAttention(
                        d_model, num_heads,
                        dropout=dropout, bias=True)
        self.ffn   = PositionWiseFFN(d_model, d_ff,
                                     dropout=dropout, activation=activation)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # ------------------------------------------------------------------ #
        # TODO 2 — pre-norm two-sublayer flow, 跟 EncoderBlock 一致:           #
        #                                                                     #
        #   attn_out, _ = self.attn(self.norm1(x), attn_mask=attn_mask)        #
        #   x = x + self.resid_dropout(attn_out)                               #
        #                                                                     #
        #   ffn_out = self.ffn(self.norm2(x))                                  #
        #   x = x + self.resid_dropout(ffn_out)                                #
        #                                                                     #
        #   return x                                                           #
        # ------------------------------------------------------------------ #
        attn_out, _ = self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.resid_dropout(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.resid_dropout(ffn_out)

        return x


class TextTransformer(nn.Module):
    """
    CLIP text encoder.

    Parameters
    ----------
    vocab_size  : int   — tokenizer.vocab_size
    max_len     : int   — context length, 训练 / 推理时 token 序列固定到这个长度
    d_model     : int   — internal embedding dim (CLIP 原版 base 用 512)
    d_shared    : int   — 投影到共享空间的目标维度 (M2 会和 image side 对齐)
    depth       : int   — CausalEncoderBlock 数量
    num_heads   : int   — attention heads per block
    d_ff        : int | None  — FFN 隐藏维度, None → 4 * d_model
    dropout     : float — attention / FFN / residual dropout
    activation  : str   — 'gelu' | 'relu', CLIP 原版用 gelu

    Forward
    -------
    token_ids  : (B, L) LongTensor — tokenizer.encode 出来的
    eos_pos    : (B,)  LongTensor — 每条样本 EOS 所在 index

    Returns
    -------
    text_embed : (B, d_shared)  — 未 normalize, 未乘 temperature.
                                  L2 normalize + temperature 留给 M2 (CLIPModel).
    """

    def __init__(self, vocab_size: int, max_len: int,
                 d_model: int = 256, d_shared: int = 128,
                 depth: int = 4, num_heads: int = 4,
                 d_ff: int = None, dropout: float = 0.0,
                 activation: str = "gelu"):
        super().__init__()
        self.d_model = d_model
        self.d_shared = d_shared
        self.max_len = max_len

        # ------------------------------------------------------------------ #
        # TODO 3 — Token embedding.                                            #
        #                                                                     #
        #   self.token_emb = nn.Embedding(vocab_size, d_model)                 #
        #                                                                     #
        # 是 learnable lookup table, shape (vocab_size, d_model).              #
        # ------------------------------------------------------------------ #
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # ------------------------------------------------------------------ #
        # TODO 4 — Learnable positional embedding.                             #
        #                                                                     #
        #   self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))      #
        #                                                                     #
        # Shape (1, max_len, d_model) — leading 1 用于 batch broadcast.         #
        # 与 Transformer 阶段的 sinusoidal PE 不同, 这里 learnable.            #
        # 全零初始化 OK; 大规模训练时可换 trunc_normal_(..., std=0.01).        #
        # ------------------------------------------------------------------ #
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        # ------------------------------------------------------------------ #
        # TODO 5 — Stack of CausalEncoderBlock.                                #
        #                                                                     #
        #   self.blocks = nn.ModuleList([                                      #
        #       CausalEncoderBlock(d_model=d_model, num_heads=num_heads,       #
        #                          d_ff=d_ff, dropout=dropout,                 #
        #                          activation=activation)                      #
        #       for _ in range(depth)                                          #
        #   ])                                                                 #
        #                                                                     #
        # 必须用 nn.ModuleList — 普通 list 不会注册参数, 优化器看不到.         #
        # ------------------------------------------------------------------ #
        self.blocks = nn.ModuleList([
            CausalEncoderBlock(d_model=d_model, num_heads=num_heads,
                               d_ff=d_ff, dropout=dropout,
                               activation=activation)
            for _ in range(depth)
        ])

        # ------------------------------------------------------------------ #
        # TODO 6 — Final LayerNorm.                                            #
        #                                                                     #
        #   self.final_norm = nn.LayerNorm(d_model)                            #
        #                                                                     #
        # 跟 ViT 一样 — pre-norm 架构最后一层 block 输出还没 norm 过.          #
        # ------------------------------------------------------------------ #
        self.final_norm = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------ #
        # TODO 7 — Text projection 到共享空间.                                  #
        #                                                                     #
        #   self.text_projection = nn.Linear(d_model, d_shared, bias=False)    #
        #                                                                     #
        # CLIP convention: bias=False. M0.5 几何讨论过 — 共享空间里我们关心方向, #
        # bias 会引入偏移破坏 cosine 几何.                                       #
        # ------------------------------------------------------------------ #
        self.text_projection = nn.Linear(d_model, d_shared, bias=False)

        # ------------------------------------------------------------------ #
        # TODO 8 — 提前构造 causal mask 注册为 buffer.                         #
        #                                                                     #
        #   self.register_buffer(                                              #
        #       "causal_mask",                                                 #
        #       build_causal_mask(max_len),                                    #
        #       persistent=False,                                              #
        #   )                                                                  #
        #                                                                     #
        # 为什么用 buffer 而不是每次 forward 现造:                              #
        # 1. mask 不依赖 input, 只依赖 max_len, 不需要每步重建.                 #
        # 2. register_buffer 让它跟着 model.to(device) 自动转设备.              #
        # 3. persistent=False — mask 是固定的 (上三角 -inf), 不需要存进         #
        #    state_dict (省 checkpoint 空间, 也避免 load_state_dict 报错).      #
        # ------------------------------------------------------------------ #
        self.register_buffer("causal_mask", build_causal_mask(max_len), persistent=False)

    def forward(self, token_ids: torch.Tensor, eos_pos: torch.Tensor):
        B, L = token_ids.shape
        assert L == self.max_len, (
            f"TextTransformer 期望固定长度 {self.max_len}, 收到 {L}. "
            f"用 tokenizer.encode(text, max_len={self.max_len}) padding."
        )

        # ------------------------------------------------------------------ #
        # TODO 9 — Token embedding + positional embedding.                     #
        #                                                                     #
        #   x = self.token_emb(token_ids)        # (B, L, d_model)             #
        #   x = x + self.pos_emb                  # broadcast (1, L, d) → batch #
        #                                                                     #
        # pos_emb 形状 (1, max_len, d_model), broadcast 沿 batch 维.            #
        # ------------------------------------------------------------------ #
        x = self.token_emb(token_ids) + self.pos_emb

        # ------------------------------------------------------------------ #
        # TODO 10 — 过 N 个 CausalEncoderBlock, 每个都传入 causal mask.         #
        #                                                                     #
        #   for blk in self.blocks:                                            #
        #       x = blk(x, attn_mask=self.causal_mask)                         #
        #                                                                     #
        # self.causal_mask 形状 (L, L), 在 attention 里 broadcast 到 (B, h, L, L). #
        # ------------------------------------------------------------------ #
        for blk in self.blocks:
            x = blk(x, attn_mask=self.causal_mask)

        # ------------------------------------------------------------------ #
        # TODO 11 — Final LayerNorm.                                           #
        #                                                                     #
        #   x = self.final_norm(x)                # (B, L, d_model)             #
        # ------------------------------------------------------------------ #
        x = self.final_norm(x)

        # ------------------------------------------------------------------ #
        # TODO 12 — 用 advanced indexing 取 EOS 位置 — CLIP 核心 trick.        #
        #                                                                     #
        #   sentence_feat = x[torch.arange(B, device=x.device), eos_pos]       #
        #                                          # (B, d_model)              #
        #                                                                     #
        # 为什么取 EOS 不取 SOS:                                                 #
        #   causal mask 下, 位置 0 只能看到自己,                                 #
        #   只有 EOS 位置看得到前面所有 token (M0.4).                            #
        #                                                                     #
        # 为什么不直接 x[:, -1]:                                                 #
        #   不同样本的 EOS 位置不同 (短文本 EOS 在前, 长文本 EOS 在后),          #
        #   末位置可能是 PAD. 必须用 eos_pos 精确索引.                          #
        #                                                                     #
        # eos_pos 必须在 x 的 device 上, 否则 advanced indexing 报错.            #
        # ------------------------------------------------------------------ #
        sentence_feat = x[torch.arange(B, device=x.device), eos_pos]

        # ------------------------------------------------------------------ #
        # TODO 13 — Project to shared space.                                   #
        #                                                                     #
        #   text_embed = self.text_projection(sentence_feat)   # (B, d_shared)  #
        #   return text_embed                                                   #
        #                                                                     #
        # NOTE: 这里 NOT normalize, NOT temperature-scale.                      #
        # 这两步留给 M2 的 CLIPModel — text encoder 只负责出 raw embedding.    #
        # ------------------------------------------------------------------ #
        text_embed = self.text_projection(sentence_feat)
        return text_embed


# ---------------------------------------------------------------------------- #
# Driver: shape + param count                                                  #
# ---------------------------------------------------------------------------- #


def main():
    torch.manual_seed(0)

    vocab_size = 100
    max_len = 16
    d_model, d_shared = 128, 64
    depth, num_heads = 2, 4

    model = TextTransformer(
        vocab_size=vocab_size, max_len=max_len,
        d_model=d_model, d_shared=d_shared,
        depth=depth, num_heads=num_heads,
        dropout=0.0, activation="gelu",
    )
    model.eval()

    B = 3
    token_ids = torch.randint(0, vocab_size, (B, max_len))
    eos_pos = torch.tensor([5, 10, max_len - 1])
    text_embed = model(token_ids, eos_pos)

    print(f"token_ids.shape  = {tuple(token_ids.shape)}")
    print(f"eos_pos          = {eos_pos.tolist()}")
    print(f"text_embed.shape = {tuple(text_embed.shape)}    (expected {(B, d_shared)})")
    print(f"causal_mask.shape = {tuple(model.causal_mask.shape)}    (expected {(max_len, max_len)})")
    print()
    total = sum(p.numel() for p in model.parameters())
    print(f"total params     = {total:,}")
    print(f"token_emb params = {model.token_emb.weight.numel():,}    "
          f"({vocab_size} × {d_model})")
    print(f"pos_emb params   = {model.pos_emb.numel():,}    "
          f"(1 × {max_len} × {d_model})")
    print(f"text_proj params = {model.text_projection.weight.numel():,}    "
          f"({d_shared} × {d_model}, no bias)")


if __name__ == "__main__":
    main()
