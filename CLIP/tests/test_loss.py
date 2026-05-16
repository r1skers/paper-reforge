"""
Tests for CLIP/src/loss.py — info_nce_symmetric.

测试的数学性质
-------------
1. shape: 输入 (B, B) → 输出 scalar tensor
2. default 行为: 不传 logits_per_text 时自动取转置
3. boundary 1: uniform logits → loss = log(B)             ← learn-nothing baseline
4. boundary 2: 强对角线 (identity * 大数) → loss → 0       ← 完美对齐下界
5. boundary 3: anti-diagonal → loss > log(B)              ← 学反时的反向信号
6. symmetric in transpose: L(S) == L(Sᵀ)                  ← M0.2 symmetric 的数学结果
7. gradient 通畅 + 收敛点梯度 → 0
8. 跟 CLIPModel forward 兼容 (end-to-end smoke)

Run
---
    cd D:/Dev/repos/paper-reforge/CLIP
    python -m pytest tests/test_loss.py -v
"""

import math
import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from loss import info_nce_symmetric


# ---------------------------------------------------------------------------- #
# Shape + scalar                                                                #
# ---------------------------------------------------------------------------- #


def test_loss_returns_scalar_tensor():
    logits = torch.randn(4, 4)
    loss = info_nce_symmetric(logits)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_loss_default_logits_per_text_is_transpose():
    """不传 logits_per_text 时, 内部应该用 logits_per_image.T."""
    logits = torch.randn(4, 4)
    loss_default = info_nce_symmetric(logits)
    loss_explicit = info_nce_symmetric(logits, logits.T)
    assert torch.allclose(loss_default, loss_explicit, atol=1e-12)


# ---------------------------------------------------------------------------- #
# Boundary behavior                                                            #
# ---------------------------------------------------------------------------- #


def test_uniform_logits_gives_log_B():
    """logits 全 0 → softmax 每行均匀 1/B → -log(1/B) = log(B). 这是 InfoNCE
    在'啥都没学到'状态下的理论 baseline. 训练 loss 第一个 epoch 应该接近 log(B),
    后面才慢慢下降. 用来判断模型是否真的在学习."""
    for B in [2, 4, 8, 32]:
        logits = torch.zeros(B, B)
        loss = info_nce_symmetric(logits).item()
        expected = math.log(B)
        assert abs(loss - expected) < 1e-6, (
            f"B={B}: loss={loss:.6f}, expected log({B})={expected:.6f}"
        )


def test_strong_diagonal_gives_near_zero_loss():
    """logits 是 identity * 大数 → 对角线主导 → 完美匹配 → loss → 0.
    这是 InfoNCE 的 lower bound 行为, 训练理想情况收敛到这里."""
    B = 8
    logits = torch.eye(B, dtype=torch.float64) * 50.0
    loss = info_nce_symmetric(logits).item()
    assert loss < 1e-15, f"identity*50 时 loss = {loss:.2e}, expected ~0"


def test_anti_diagonal_gives_high_loss():
    """logits 让 OFF-diagonal 高、对角线低 → 完全学反 → loss 应该比 log(B) 还高."""
    B = 4
    logits = torch.ones(B, B) * 10.0
    # 把对角线压低
    logits[torch.arange(B), torch.arange(B)] = -10.0
    loss = info_nce_symmetric(logits).item()
    # 'random guess' baseline 是 log(B), 学反了应该明显高于这个
    assert loss > math.log(B) + 1.0, (
        f"完全学反时 loss = {loss:.4f} 应该 >> log({B}) = {math.log(B):.4f}"
    )


# ---------------------------------------------------------------------------- #
# Symmetric in transpose: L(S) == L(Sᵀ)                                         #
# ---------------------------------------------------------------------------- #


def test_loss_symmetric_in_transpose():
    """
    *** M0.2 推过的 symmetric 性质的数学体现 ***

    Symmetric loss 的形式 L = (L_i2t(S) + L_t2i(S)) / 2 = (L_i2t(S) + L_i2t(Sᵀ)) / 2
    那么 L(S) 和 L(Sᵀ) 应该完全相等 — 把 S 转置就是把 image 和 text 角色对换,
    symmetric loss 对这种对换 invariant.
    """
    torch.manual_seed(0)
    for B in [3, 5, 8]:
        logits = torch.randn(B, B, dtype=torch.float64) * 2.0
        loss_S = info_nce_symmetric(logits).item()
        loss_ST = info_nce_symmetric(logits.T).item()
        diff = abs(loss_S - loss_ST)
        assert diff < 1e-12, f"B={B}: L(S)={loss_S:.10f}, L(Sᵀ)={loss_ST:.10f}, diff={diff:.2e}"


# ---------------------------------------------------------------------------- #
# Gradient flow                                                                 #
# ---------------------------------------------------------------------------- #


def test_loss_backward_produces_grad():
    """loss 必须能反传到 logits 上 — 这是 trainer 能运行的基础."""
    torch.manual_seed(0)
    logits = torch.randn(4, 4, requires_grad=True)
    loss = info_nce_symmetric(logits)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0


def test_zero_gradient_at_strong_diagonal():
    """完美匹配时 (loss ~ 0), 对 logits 的梯度应该接近 0 — 收敛点的几何性质."""
    B = 4
    logits = (torch.eye(B) * 50.0).clone().requires_grad_(True)
    loss = info_nce_symmetric(logits)
    loss.backward()
    assert logits.grad.abs().max().item() < 1e-15, (
        f"identity*50 时梯度 = {logits.grad.abs().max().item():.2e}, expected ~0"
    )


# ---------------------------------------------------------------------------- #
# Integration: 跟 CLIPModel forward 输出兼容                                     #
# ---------------------------------------------------------------------------- #


def test_loss_with_clip_model_output():
    """end-to-end smoke: CLIPModel.forward → info_nce_symmetric → backward.
    确认 loss API 跟 CLIPModel 的 forward signature 完美对齐."""
    from clip_model import CLIPModel
    from image_encoder import ImageEncoderViT
    from text_encoder import TextTransformer

    torch.manual_seed(0)
    d_shared = 64
    img_enc = ImageEncoderViT(
        img_size=32, patch_size=4, in_chans=3,
        d_model=64, d_shared=d_shared, depth=1, num_heads=2,
    )
    txt_enc = TextTransformer(
        vocab_size=50, max_len=16,
        d_model=64, d_shared=d_shared, depth=1, num_heads=2,
    )
    clip = CLIPModel(img_enc, txt_enc, init_temperature=0.2)
    clip.train()

    B = 3
    images = torch.randn(B, 3, 32, 32)
    token_ids = torch.randint(0, 50, (B, 16))
    eos_pos = torch.randint(1, 16, (B,))

    logits_i2t, logits_t2i = clip(images, token_ids, eos_pos)
    loss = info_nce_symmetric(logits_i2t, logits_t2i)
    loss.backward()

    # logit_scale 必须收到非零梯度 — temperature 是 InfoNCE 的核心 learnable
    assert clip.logit_scale.grad is not None
    assert clip.logit_scale.grad.abs().item() > 0
