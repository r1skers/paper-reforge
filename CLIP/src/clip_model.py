"""
M2.2 — CLIPModel: dual-tower fusion + L2 normalize + learnable temperature.

把 M1 的 TextTransformer 和 M2.1 的 ImageEncoderViT 拼起来, 加两件 CLIP 特有的
组件:
    1. L2 normalize → 把两塔输出推到单位球面 (M0.5 几何)
    2. Learnable temperature τ → 控制 softmax 软硬度 (M0.3)
       参数化为 logit_scale = log(1/τ), 用 exp() 保正性 + log-scale 跨数量级.

Forward 输出
-----------
跟 OpenAI 官方 CLIP API 对齐, 返回两个 logit 矩阵:
    logits_per_image : (B, B) — 沿行做 softmax 得 image→text 概率
    logits_per_text  : (B, B) — 沿列做 softmax 得 text→image 概率, 等于上面转置

InfoNCE loss 留给 M3 (在 trainer / experiments 里算), 这里只出 logits.
这是 OpenAI / HuggingFace 的 convention — model 出 logits, loss 在外面算,
方便切换不同 loss (symmetric, asymmetric, with hard-negative weighting, ...).

Toy-scale 配置建议 (M0.3 讨论过)
-------------------------------
- batch 小 (CPU 上 128 已经偏大) → 应该用偏大的 τ
- 不要照搬 CLIP 原版 init τ=0.07 (1/τ=14)
- 推荐 init τ=0.2 (logit_scale_init = log(5) ≈ 1.609)
- clamp 上限收紧到 log(20) (即 τ 最低 0.05)

Run
---
    python clip_model.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_encoder import ImageEncoderViT
from text_encoder import TextTransformer


class CLIPModel(nn.Module):
    """
    Two-tower CLIP wrapper.

    Parameters
    ----------
    image_encoder         : ImageEncoderViT — 必须输出 (B, d_shared)
    text_encoder          : TextTransformer — 必须输出 (B, d_shared), 同样 d_shared
    init_temperature      : float, default 0.2
        初始 τ. CLIP 原版 0.07, 我们 toy-scale 用 0.2 (更大 = softmax 更软, 适合小 batch).
    max_logit_scale       : float, default math.log(20)
        clamp 1/τ 的上限 (logit_scale.data 的最大值). 原版 log(100), 我们收紧到 log(20).
        即 τ 最低 0.05. 这是保险丝 — 不让 τ 在训练中爆冲到 0.

    Forward
    -------
    images     : (B, C, H, W)
    token_ids  : (B, L) LongTensor
    eos_pos    : (B,)  LongTensor

    Returns
    -------
    logits_per_image : (B, B) — 已经乘 1/τ, 可直接送 cross_entropy
    logits_per_text  : (B, B) — = logits_per_image.T

    Note
    ----
    返回的 logits 已经 scale, 没有再 normalize.
    InfoNCE loss 在 trainer (M3) 里:
        labels = torch.arange(B)
        loss = (F.cross_entropy(logits_per_image, labels)
                + F.cross_entropy(logits_per_text,  labels)) / 2
    """

    def __init__(self,
                 image_encoder: ImageEncoderViT,
                 text_encoder:  TextTransformer,
                 init_temperature: float = 0.2,
                 max_logit_scale:  float = math.log(20)):
        super().__init__()
        assert image_encoder.d_shared == text_encoder.d_shared, (
            f"d_shared 不匹配: image={image_encoder.d_shared}, "
            f"text={text_encoder.d_shared}. 两塔必须投到同一个空间."
        )
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.max_logit_scale = max_logit_scale

        # ------------------------------------------------------------------ #
        # TODO 1 — Learnable logit_scale = log(1/τ) as a scalar Parameter.    #
        #                                                                     #
        # 注意三件事:                                                          #
        #   (a) 存的是 log(1/τ), 不是 τ 本身. forward 里用 .exp() 还原.        #
        #       理由 (M0.3): log-scale 让 1/τ 在不同数量级上的梯度尺度匹配,    #
        #       同时 .exp() 保证 1/τ 永远正.                                   #
        #   (b) 形状是 scalar — torch.tensor(value), 不是 (1,) 也不是 (1,1).   #
        #   (c) requires_grad=True (Parameter 默认就是).                       #
        #                                                                     #
        #   init_value = math.log(1.0 / init_temperature)                     #
        #   self.logit_scale = nn.Parameter(torch.tensor(init_value))         #
        # ------------------------------------------------------------------ #
        init_value = math.log(1.0 / init_temperature)
        self.logit_scale = nn.Parameter(torch.tensor(init_value))

    # ------------------------------------------------------------------ #
    # Single-tower 接口 — 评估 / retrieval 时单独编码一边                  #
    # ------------------------------------------------------------------ #

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns L2-normalized image embeddings (B, d_shared).

        Inference convention: 输出已 L2 normalize, 直接和 encode_text 的输出
        点积就是 cosine similarity. 注意这里 NOT 乘 temperature — 因为在
        retrieval / zero-shot classification 阶段不需要 temperature scaling
        (排序不变). 只在训练 forward 里乘 temperature.
        """
        # ------------------------------------------------------------------ #
        # TODO 2 — image_embed = image_encoder(images), L2 normalize, return. #
        #                                                                     #
        #   x = self.image_encoder(images)                # (B, d_shared)      #
        #   x = F.normalize(x, dim=-1)                    # 单位球面            #
        #   return x                                                            #
        # ------------------------------------------------------------------ #
        x = self.image_encoder(images)
        x = F.normalize(x, dim=-1)
        return x

    def encode_text(self, token_ids: torch.Tensor, eos_pos: torch.Tensor) -> torch.Tensor:
        """
        Returns L2-normalized text embeddings (B, d_shared).

        和 encode_image 完全镜像. zero-shot classification 时:
            text_features = model.encode_text(class_prompts, eos_pos_per_prompt)
            similarities  = model.encode_image(image) @ text_features.T
            prediction    = similarities.argmax()
        """
        # ------------------------------------------------------------------ #
        # TODO 3 — text_embed = text_encoder(token_ids, eos_pos), normalize.  #
        # ------------------------------------------------------------------ #
        x = self.text_encoder(token_ids, eos_pos)
        x = F.normalize(x, dim=-1)
        return x

    # ------------------------------------------------------------------ #
    # Training forward — 出 similarity logits                              #
    # ------------------------------------------------------------------ #

    def forward(self, images, token_ids, eos_pos):
        # ------------------------------------------------------------------ #
        # TODO 4 — 编码两塔 + L2 normalize.                                    #
        #                                                                     #
        #   image_features = self.encode_image(images)        # (B, d_shared)  #
        #   text_features  = self.encode_text(token_ids, eos_pos)              #
        #                                                                     #
        # 两个都已经在 encode_* 里 L2 normalize 过了. 不要重复.                  #
        # ------------------------------------------------------------------ #
        image_features = self.encode_image(images)
        text_features = self.encode_text(token_ids, eos_pos)

        # ------------------------------------------------------------------ #
        # TODO 5 — Clamp logit_scale 然后 exp() 得 1/τ.                        #
        #                                                                     #
        # Clamp 必须在 .data 上原地做, 否则会破坏梯度图:                          #
        #   with torch.no_grad():                                              #
        #       self.logit_scale.clamp_(max=self.max_logit_scale)              #
        #                                                                     #
        # 然后:                                                                #
        #   logit_scale = self.logit_scale.exp()                               #
        #                                                                     #
        # CLIP 原版每个 forward 都做这个 clamp. 它是 in-place 修改 Parameter,    #
        # 不影响该步的 forward / backward (clamp 在前向 happen 之前生效, 而且    #
        # 是 hard floor/ceiling 不可微).                                         #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            self.logit_scale.clamp_(max=self.max_logit_scale)
        logit_scale = self.logit_scale.exp()

        # ------------------------------------------------------------------ #
        # TODO 6 — Similarity matrix → logits.                                 #
        #                                                                     #
        #   logits_per_image = logit_scale * image_features @ text_features.T  #
        #                                                                     #
        # 形状: (B, d) @ (d, B) = (B, B).                                       #
        # 因为已 normalize, 这就是 cosine * (1/τ) — M0.5 推导.                  #
        #                                                                     #
        # text-to-image 方向直接转置:                                            #
        #   logits_per_text = logits_per_image.T                                #
        #                                                                     #
        # 数学上 logits_per_text 也可以独立算成 text @ image.T, 完全等价.        #
        # 转置版本省了一次 matmul, 是 OpenAI / HF 的实际写法.                     #
        #                                                                     #
        #   return logits_per_image, logits_per_text                            #
        # ------------------------------------------------------------------ #
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text

    # ------------------------------------------------------------------ #
    # Convenience: 当前 τ 值, 训练 log 用                                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def get_temperature(self) -> float:
        """返回当前 τ 值 (= 1 / exp(logit_scale)). 训练时 logging 用."""
        return float(1.0 / self.logit_scale.exp())


# ---------------------------------------------------------------------------- #
# Driver: shape + logit_scale init sanity                                      #
# ---------------------------------------------------------------------------- #


def main():
    torch.manual_seed(0)

    d_shared = 64
    image_encoder = ImageEncoderViT(
        img_size=32, patch_size=4, in_chans=3,
        d_model=128, d_shared=d_shared,
        depth=2, num_heads=4, dropout=0.0,
    )
    text_encoder = TextTransformer(
        vocab_size=50, max_len=16,
        d_model=128, d_shared=d_shared,
        depth=2, num_heads=4, dropout=0.0,
    )
    clip = CLIPModel(image_encoder, text_encoder,
                     init_temperature=0.2)
    clip.eval()

    B = 3
    images = torch.randn(B, 3, 32, 32)
    token_ids = torch.randint(0, 50, (B, 16))
    eos_pos = torch.tensor([5, 8, 15])

    logits_i2t, logits_t2i = clip(images, token_ids, eos_pos)

    print(f"images.shape       = {tuple(images.shape)}")
    print(f"token_ids.shape    = {tuple(token_ids.shape)}")
    print(f"logits_per_image   = {tuple(logits_i2t.shape)}    (expected ({B}, {B}))")
    print(f"logits_per_text    = {tuple(logits_t2i.shape)}    (expected ({B}, {B}))")
    print()
    print(f"logit_scale (= log(1/τ))  = {clip.logit_scale.item():.4f}    "
          f"(expected log(1/0.2) = {math.log(1/0.2):.4f})")
    print(f"current temperature τ      = {clip.get_temperature():.4f}    (expected 0.2)")
    print()
    print(f"logits_per_image:\n{logits_i2t.detach()}")
    print(f"logits_per_text == logits_per_image.T?  "
          f"{torch.allclose(logits_t2i, logits_i2t.T)}")

    # 单塔 normalize 检查
    img_feat = clip.encode_image(images)
    txt_feat = clip.encode_text(token_ids, eos_pos)
    print()
    print(f"|image_feat|_2 per row  = {img_feat.norm(dim=-1).tolist()}    (expected all ~1.0)")
    print(f"|text_feat|_2 per row   = {txt_feat.norm(dim=-1).tolist()}    (expected all ~1.0)")


if __name__ == "__main__":
    main()
