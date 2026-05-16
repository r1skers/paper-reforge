"""
Tests for CLIP/src/clip_model.py — CLIPModel.

最关键的几条 invariant
---------------------
1. forward 输出 (logits_per_image, logits_per_text) 形状是 (B, B)
2. logits_per_text == logits_per_image.T (CLIP 对称性)
3. encode_image / encode_text 出来的向量 L2 norm = 1 (单位球面)
4. logit_scale 是 learnable + log-scale + 上限 clamp
5. d_shared 不匹配时构造器报错 (sanity guard)

Run
---
    cd D:/Dev/repos/paper-reforge/CLIP
    python -m pytest tests/test_clip_model.py -v
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from clip_model import CLIPModel
from image_encoder import ImageEncoderViT
from text_encoder import TextTransformer


# ---------------------------------------------------------------------------- #
# Fixtures                                                                     #
# ---------------------------------------------------------------------------- #


def _make_clip(d_shared=64, max_len=16, vocab_size=50,
               init_temperature=0.2):
    image_encoder = ImageEncoderViT(
        img_size=32, patch_size=4, in_chans=3,
        d_model=64, d_shared=d_shared, depth=2, num_heads=4,
        dropout=0.0,
    )
    text_encoder = TextTransformer(
        vocab_size=vocab_size, max_len=max_len,
        d_model=64, d_shared=d_shared, depth=2, num_heads=4,
        dropout=0.0,
    )
    return CLIPModel(image_encoder, text_encoder,
                     init_temperature=init_temperature)


def _make_inputs(B=3, max_len=16, vocab_size=50):
    images = torch.randn(B, 3, 32, 32)
    token_ids = torch.randint(0, vocab_size, (B, max_len))
    eos_pos = torch.randint(1, max_len, (B,))
    return images, token_ids, eos_pos


# ---------------------------------------------------------------------------- #
# Shape + symmetric output                                                     #
# ---------------------------------------------------------------------------- #


def test_forward_output_shapes():
    B = 4
    clip = _make_clip()
    clip.eval()
    images, token_ids, eos_pos = _make_inputs(B=B)
    logits_i2t, logits_t2i = clip(images, token_ids, eos_pos)
    assert logits_i2t.shape == (B, B)
    assert logits_t2i.shape == (B, B)


def test_logits_per_text_is_transpose_of_per_image():
    """CLIP 对称性: logits_per_text == logits_per_image.T."""
    clip = _make_clip()
    clip.eval()
    images, token_ids, eos_pos = _make_inputs()
    logits_i2t, logits_t2i = clip(images, token_ids, eos_pos)
    assert torch.allclose(logits_t2i, logits_i2t.T, atol=1e-6)


# ---------------------------------------------------------------------------- #
# L2 normalize                                                                 #
# ---------------------------------------------------------------------------- #


def test_encode_image_unit_norm():
    clip = _make_clip()
    clip.eval()
    images, _, _ = _make_inputs()
    feat = clip.encode_image(images)
    norms = feat.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"image features 不是单位向量, norms = {norms.tolist()}"
    )


def test_encode_text_unit_norm():
    clip = _make_clip()
    clip.eval()
    _, token_ids, eos_pos = _make_inputs()
    feat = clip.encode_text(token_ids, eos_pos)
    norms = feat.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"text features 不是单位向量, norms = {norms.tolist()}"
    )


# ---------------------------------------------------------------------------- #
# Logits bound — normalize 之后 cosine ∈ [-1, 1], scale 后绝对值 ≤ 1/τ          #
# ---------------------------------------------------------------------------- #


def test_logits_bounded_by_temperature():
    clip = _make_clip(init_temperature=0.2)
    clip.eval()
    images, token_ids, eos_pos = _make_inputs()
    logits, _ = clip(images, token_ids, eos_pos)

    inv_tau = 1.0 / 0.2  # = 5
    # 严格上界 |cosine| <= 1, 所以 |logits| <= inv_tau
    assert logits.abs().max().item() <= inv_tau + 1e-5, (
        f"logits 超过 1/τ = {inv_tau}, 说明 encode_* 没 normalize."
    )


# ---------------------------------------------------------------------------- #
# Learnable temperature                                                        #
# ---------------------------------------------------------------------------- #


def test_logit_scale_init_value():
    """init_temperature=0.2 → logit_scale 初值 = log(5)."""
    clip = _make_clip(init_temperature=0.2)
    expected = math.log(1.0 / 0.2)
    assert abs(clip.logit_scale.item() - expected) < 1e-6


def test_logit_scale_is_learnable():
    """logit_scale 必须出现在 .parameters() 里, 且 requires_grad=True."""
    clip = _make_clip()
    names = {n for n, _ in clip.named_parameters()}
    assert "logit_scale" in names
    assert clip.logit_scale.requires_grad


def test_logit_scale_is_scalar():
    """logit_scale 是 0-d scalar, 不是 (1,) 或 (1, 1)."""
    clip = _make_clip()
    assert clip.logit_scale.dim() == 0


def test_logit_scale_clamp_in_forward():
    """如果用户把 logit_scale 设到天上, forward 时应该被 clamp 回 max_logit_scale."""
    clip = _make_clip(init_temperature=0.2)
    clip.eval()
    # 手动把 logit_scale 推到非常大
    with torch.no_grad():
        clip.logit_scale.fill_(100.0)
    images, token_ids, eos_pos = _make_inputs()
    _ = clip(images, token_ids, eos_pos)
    # forward 后应该被 clamp 到 max_logit_scale
    assert clip.logit_scale.item() <= clip.max_logit_scale + 1e-6
    assert clip.logit_scale.item() == pytest.approx(clip.max_logit_scale, abs=1e-6)


def test_get_temperature():
    clip = _make_clip(init_temperature=0.2)
    assert abs(clip.get_temperature() - 0.2) < 1e-5


# ---------------------------------------------------------------------------- #
# d_shared mismatch guard                                                       #
# ---------------------------------------------------------------------------- #


def test_d_shared_mismatch_raises():
    """两塔 d_shared 不一致时构造器必须报错."""
    image_encoder = ImageEncoderViT(
        img_size=32, patch_size=4, in_chans=3,
        d_model=64, d_shared=64, depth=1, num_heads=2,
    )
    text_encoder = TextTransformer(
        vocab_size=50, max_len=16,
        d_model=64, d_shared=32, depth=1, num_heads=2,
    )
    with pytest.raises(AssertionError, match="d_shared"):
        CLIPModel(image_encoder, text_encoder)


# ---------------------------------------------------------------------------- #
# 端到端 sanity: InfoNCE loss 可以反传 (M3 验证用, 这里先 smoke)                  #
# ---------------------------------------------------------------------------- #


def test_loss_backward_smoke():
    """完整 forward → symmetric InfoNCE → backward, 检查所有 trainable 参数
    都拿到了非零梯度. 这是 M3 真正写 trainer 前的最后保险."""
    torch.manual_seed(0)
    B = 4
    clip = _make_clip()
    clip.train()
    images, token_ids, eos_pos = _make_inputs(B=B)

    logits_i2t, logits_t2i = clip(images, token_ids, eos_pos)
    labels = torch.arange(B)
    loss = (F.cross_entropy(logits_i2t, labels)
            + F.cross_entropy(logits_t2i, labels)) / 2

    loss.backward()

    # 检查关键参数都收到了梯度
    has_grad = {}
    for name, p in clip.named_parameters():
        has_grad[name] = (p.grad is not None) and (p.grad.abs().sum().item() > 0)

    # logit_scale 必须有梯度 — 这是 contrastive loss 的核心 learnable scalar
    assert has_grad["logit_scale"], "logit_scale 没拿到梯度"
    # 两塔都必须有梯度
    assert any(has_grad[n] for n in has_grad if "image_encoder" in n), (
        "image_encoder 整个没拿到梯度"
    )
    assert any(has_grad[n] for n in has_grad if "text_encoder" in n), (
        "text_encoder 整个没拿到梯度"
    )
    # projection layers 必须有梯度 — 它们是 d_shared 空间的桥梁
    assert has_grad.get("image_encoder.image_projection.weight", False)
    assert has_grad.get("text_encoder.text_projection.weight", False)
