"""
Tests for CLIP/src/image_encoder.py — ImageEncoderViT.

Run
---
    cd D:/Dev/repos/paper-reforge/CLIP
    python -m pytest tests/test_image_encoder.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from image_encoder import ImageEncoderViT


# ---------------------------------------------------------------------------- #
# Shape                                                                        #
# ---------------------------------------------------------------------------- #


def test_output_shape():
    """(B, 3, H, W) → (B, d_shared)"""
    B, img_size, in_chans = 4, 32, 3
    d_shared = 64
    model = ImageEncoderViT(
        img_size=img_size, patch_size=4, in_chans=in_chans,
        d_model=128, d_shared=d_shared, depth=2, num_heads=4,
    )
    model.eval()

    images = torch.randn(B, in_chans, img_size, img_size)
    out = model(images)

    assert out.shape == (B, d_shared)


def test_num_patches():
    """N = (img_size / patch_size)^2, pos_embed 应该有 N+1 个 slot."""
    model = ImageEncoderViT(
        img_size=32, patch_size=4, in_chans=3,
        d_model=64, d_shared=32, depth=1, num_heads=2,
    )
    N = (32 // 4) ** 2
    assert model.patch_embed.num_patches == N
    assert model.pos_embed.shape == (1, N + 1, 64)
    assert model.cls_token.shape == (1, 1, 64)


def test_grayscale_input():
    """MNIST 风格 (1 通道) 也应该工作."""
    model = ImageEncoderViT(
        img_size=28, patch_size=7, in_chans=1,
        d_model=64, d_shared=32, depth=1, num_heads=2,
    )
    model.eval()

    images = torch.randn(2, 1, 28, 28)
    out = model(images)
    assert out.shape == (2, 32)


# ---------------------------------------------------------------------------- #
# Parameter registration                                                       #
# ---------------------------------------------------------------------------- #


def test_all_params_registered():
    model = ImageEncoderViT(
        img_size=32, patch_size=4, in_chans=3,
        d_model=64, d_shared=32, depth=2, num_heads=4,
    )
    names = {n for n, _ in model.named_parameters()}

    # patch_embed 内部有 Conv2d 权重
    assert any("patch_embed" in n for n in names)
    # CLS / PE
    assert "cls_token" in names
    assert "pos_embed" in names
    # 至少一个 block 的子模块
    assert any("blocks.0.attn.W_Q.weight" in n for n in names)
    assert any("blocks.0.norm1.weight" in n for n in names)
    # final norm
    assert "final_norm.weight" in names
    # image_projection (no bias)
    assert "image_projection.weight" in names
    assert "image_projection.bias" not in names


def test_no_classification_head():
    """ImageEncoderViT 不应该有 'head.weight' (那是 ViT 原版的, 我们不要)."""
    model = ImageEncoderViT(
        img_size=32, patch_size=4, in_chans=3,
        d_model=64, d_shared=32, depth=2, num_heads=4,
    )
    names = {n for n, _ in model.named_parameters()}
    assert not any(n.startswith("head.") for n in names), (
        "应该没有 classification head. 我们要 d_shared 投影, 不要 num_classes 投影."
    )


# ---------------------------------------------------------------------------- #
# Behavioral                                                                   #
# ---------------------------------------------------------------------------- #


def test_different_images_give_different_outputs():
    """两张不同的图编码后必须不同."""
    torch.manual_seed(0)
    model = ImageEncoderViT(
        img_size=32, patch_size=4, in_chans=3,
        d_model=64, d_shared=32, depth=2, num_heads=4,
    )
    model.eval()

    img_a = torch.randn(1, 3, 32, 32)
    img_b = torch.randn(1, 3, 32, 32)

    out_a = model(img_a)
    out_b = model(img_b)
    assert (out_a - out_b).abs().max().item() > 1e-3


def test_batch_independent():
    """同一张图无论单独跑还是和别的图一起 batch 跑, 输出应该一致 (确认没有
    意外的 cross-sample 信息泄漏, 比如 BatchNorm)."""
    torch.manual_seed(0)
    model = ImageEncoderViT(
        img_size=32, patch_size=4, in_chans=3,
        d_model=64, d_shared=32, depth=2, num_heads=4,
        dropout=0.0,
    )
    model.eval()

    img = torch.randn(1, 3, 32, 32)
    other = torch.randn(2, 3, 32, 32)

    out_single = model(img)
    out_batched = model(torch.cat([img, other], dim=0))[:1]

    diff = (out_single - out_batched).abs().max().item()
    assert diff < 1e-5, f"batch 行为不一致 max diff = {diff:.2e}"
