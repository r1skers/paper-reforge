"""
M1.2 — Causal multi-head self-attention for the CLIP text encoder.

为什么不直接复用 Transformer/src/attention.py 里的 MultiHeadSelfAttention？
-----------------------------------------------------------------------
那个版本只接受 (B, n) 布尔型的 padding mask, 沿 key 维度做 broadcast.
而 causal mask 是 PER-QUERY 的, 形状 (L, L), 上三角 -inf.
两种 mask 的 broadcast 维度不一样 — 与其修改已经测好的 Transformer 模块,
不如在 CLIP 里独立写一个支持 causal mask 的版本 (沿用 ViT 不动原则).

API 差异
--------
    Transformer.MultiHeadSelfAttention:
        forward(x, mask: (B, n) bool, True=keep)         ← multiplicative
    CLIP.CausalMultiHeadSelfAttention:
        forward(x, attn_mask: (L, L) float, -inf 标记屏蔽)  ← additive

Additive mask 是 OpenAI 官方 CLIP 的约定, 直接加到 scaled-dot scores 上,
经 softmax 后 -inf 位置变 0. 它和 causal 的天然形式 (torch.triu) 完美匹配.

Run
---
    python causal_attention.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_causal_mask(seq_len: int) -> torch.Tensor:
    """
    构造一个 (L, L) additive causal mask.

    输出:
        mask[i, j] = 0       if j <= i   (query i 允许看 key j, 包括自己和过去)
        mask[i, j] = -inf    if j >  i   (query i 屏蔽 key j, 即未来 token)

    例 (L=4):
        [[ 0,  -inf, -inf, -inf],
         [ 0,    0,  -inf, -inf],
         [ 0,    0,    0,  -inf],
         [ 0,    0,    0,    0 ]]

    """
    # ------------------------------------------------------------------ #
    # TODO 1 — 用 torch.triu + torch.full 造 mask.                          #
    #                                                                     #
    #   mask = torch.full((seq_len, seq_len), float('-inf'))              #
    #   mask = torch.triu(mask, diagonal=1)                               #
    #                                                                     #
    # diagonal=1 表示主对角线之上 (严格上三角) 保留 -inf, 其他位置为 0.   #
    # 主对角线 (j == i) 保留 0 — query 必须能看到自己, 否则信息流就断了.    #
    # ------------------------------------------------------------------ #
    mask = torch.full((seq_len, seq_len), float('-inf'))
    mask = torch.triu(mask, diagonal=1)
    return mask


class CausalMultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with an ADDITIVE attention mask.

    Parameters
    ----------
    d_model   : int      — embedding dim, 必须能整除 num_heads
    num_heads : int      — h, attention 头数
    dropout   : float    — 应用在 attention weights 上的 dropout
    bias      : bool     — Q/K/V/O 投影是否带 bias. CLIP 原版用 True.

    Forward
    -------
    x         : (B, L, d_model)
    attn_mask : (L, L) additive float mask, -inf 标记屏蔽位置, 0 标记允许.
                None 时退化成无 mask 的标准 self-attention.

    Returns
    -------
    output    : (B, L, d_model)
    attn      : (B, h, L, L) — softmax 后的 attention weights, 保留供可视化
    """

    def __init__(self, d_model: int, num_heads: int,
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads

        # ------------------------------------------------------------------ #
        # TODO 2 — 创建 4 个 Linear (W_Q, W_K, W_V, W_O) + attention dropout.  #
        #                                                                     #
        #   self.W_Q = nn.Linear(d_model, d_model, bias=bias)                 #
        #   self.W_K = nn.Linear(d_model, d_model, bias=bias)                 #
        #   self.W_V = nn.Linear(d_model, d_model, bias=bias)                 #
        #   self.W_O = nn.Linear(d_model, d_model, bias=bias)                 #
        #   self.attn_dropout = nn.Dropout(dropout)                           #
        #                                                                     #
        # NOTE: 这里 bias 默认 True (CLIP 原版的选择). Transformer/attention.py #
        # 用 bias=False (原 Transformer paper 的选择). 两个项目的 convention   #
        # 不一样 — 不是 bug.                                                   #
        # ------------------------------------------------------------------ #
        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)
        self.W_O = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        B, L, d_model = x.shape
        h, d_k = self.h, self.d_k

        # ------------------------------------------------------------------ #
        # TODO 3 — 投影到 Q, K, V 并 reshape 成 multi-head.                    #
        #                                                                     #
        #   Q = self.W_Q(x).reshape(B, L, h, d_k).transpose(1, 2)              #
        #   K = self.W_K(x).reshape(B, L, h, d_k).transpose(1, 2)              #
        #   V = self.W_V(x).reshape(B, L, h, d_k).transpose(1, 2)              #
        #                                                                     #
        # 输出形状: (B, h, L, d_k). 和 Transformer 阶段完全一样.                #
        # ------------------------------------------------------------------ #
        Q = self.W_Q(x).reshape(B, L, h, d_k).transpose(1, 2)
        K = self.W_K(x).reshape(B, L, h, d_k).transpose(1, 2)
        V = self.W_V(x).reshape(B, L, h, d_k).transpose(1, 2)

        # ------------------------------------------------------------------ #
        # TODO 4 — Scaled dot-product scores + ADDITIVE mask + softmax.       #
        #                                                                     #
        #   S = Q @ K.transpose(-2, -1) / math.sqrt(d_k)                       #
        #       → (B, h, L, L)                                                 #
        #                                                                     #
        #   if attn_mask is not None:                                          #
        #       S = S + attn_mask           # broadcast (L, L) → (B, h, L, L)  #
        #                                                                     #
        #   attn = F.softmax(S, dim=-1)                                        #
        #   attn = self.attn_dropout(attn)                                     #
        #                                                                     #
        # 为什么 additive mask 不会出 NaN:                                       #
        # padding mask 是 multiplicative (~mask 整行 -inf 会让 softmax 出 NaN), #
        # 但 causal mask 至少保留主对角线 (query 看自己), 每行至少有一个 0,    #
        # softmax 永远 well-defined.                                            #
        # ------------------------------------------------------------------ #
        S = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

        if attn_mask is not None:
            S = S + attn_mask

        attn = F.softmax(S, dim=-1)
        attn = self.attn_dropout(attn)

        # ------------------------------------------------------------------ #
        # TODO 5 — Attention @ V, 还原 multi-head, 输出投影.                   #
        #                                                                     #
        #   head_out = attn @ V                          # (B, h, L, d_k)      #
        #   out = head_out.transpose(1, 2).reshape(B, L, d_model)              #
        #   out = self.W_O(out)                          # (B, L, d_model)     #
        #                                                                     #
        #   return out, attn                                                   #
        #                                                                     #
        # 用 .reshape 不用 .view — transpose 后 tensor 可能 non-contiguous.      #
        # ------------------------------------------------------------------ #
        head_out = attn @ V
        out = head_out.transpose(1, 2).reshape(B, L, d_model)
        out = self.W_O(out)
        return out, attn


# ---------------------------------------------------------------------------- #
# Driver: shape + causal property sanity                                       #
# ---------------------------------------------------------------------------- #


def main():
    torch.manual_seed(0)

    B, L, d_model, h = 2, 6, 16, 4
    attn = CausalMultiHeadSelfAttention(d_model, h, dropout=0.0)
    attn.eval()

    x = torch.randn(B, L, d_model)
    mask = build_causal_mask(L)
    out, weights = attn(x, attn_mask=mask)

    print(f"x.shape         = {tuple(x.shape)}")
    print(f"out.shape       = {tuple(out.shape)}    (expected {(B, L, d_model)})")
    print(f"weights.shape   = {tuple(weights.shape)}    (expected {(B, h, L, L)})")
    print()
    print(f"causal mask (L={L}):")
    print(mask)
    print()
    print(f"weights[0, 0] (head 0, batch 0) — 注意上三角应该是 0:")
    print(weights[0, 0].detach())
    print()
    # 验证 causal: 未来位置的 attention weight 必须严格为 0
    upper_tri = torch.triu(weights[0, 0], diagonal=1)
    print(f"max attention to future positions = {upper_tri.abs().max().item():.2e}    (expect 0)")


if __name__ == "__main__":
    main()
