"""
M3 — InfoNCE symmetric loss, PyTorch implementation.

核心就一行: 两个方向的 F.cross_entropy 取平均.
PyTorch 的 F.cross_entropy 已经在 log-sum-exp 稳定化 + 按 label 取对角 + 取平均
这三件事内部都做了.

为什么不直接把这一行写在 trainer 里
----------------------------------
- 让 loss 有自己的测试 (boundary behavior + gradient + transpose invariance)
- 方便将来扩展 (asymmetric loss, hard-negative weighting, label smoothing 等)
  时只改这个文件, trainer 不动.

API
---
跟 OpenAI / HF 的 CLIPModel 输出对齐:
    loss = info_nce_symmetric(logits_per_image, logits_per_text)
其中 logits_per_image 和 logits_per_text 来自 CLIPModel.forward.
两者数学上 = transpose, 但接受两个参数是为了灵活性 (比如未来 asymmetric loss).

Run
---
    python loss.py
"""

import torch
import torch.nn.functional as F


def info_nce_symmetric(logits_per_image: torch.Tensor,
                       logits_per_text:  torch.Tensor = None) -> torch.Tensor:
    """
    Symmetric InfoNCE loss.

    Args
    ----
    logits_per_image : (B, B)  — temperature-scaled similarity matrix,
                                 rows = images, cols = texts.
                                 沿行 softmax → image→text 分类.
    logits_per_text  : (B, B) or None
                       sim 矩阵 from text's POV. 一般 = logits_per_image.T.
                       传 None 时本函数自动取转置.

    Returns
    -------
    loss : scalar tensor  — (loss_i2t + loss_t2i) / 2,
                            和 numpy gold standard 在 float64 下 bit-exact.
    """
    if logits_per_text is None:
        logits_per_text = logits_per_image.T

    # ------------------------------------------------------------------ #
    # TODO 1 — 构造 labels = arange(B), device 跟 logits 对齐.             #
    #                                                                     #
    #   B = logits_per_image.shape[0]                                     #
    #   labels = torch.arange(B, device=logits_per_image.device)          #
    #                                                                     #
    # 为什么 labels 是 arange(B):                                           #
    #   对第 i 行做 softmax, "正确类" 就是 j=i (匹配 pair).                  #
    #   所以一整列 label = [0, 1, 2, ..., B-1] = arange(B).                #
    # device 必须对齐, 否则 GPU 训练时 F.cross_entropy 会报错.               #
    # ------------------------------------------------------------------ #
    B = logits_per_image.shape[0]
    labels = torch.arange(B, device=logits_per_image.device)

    # ------------------------------------------------------------------ #
    # TODO 2 — 两个方向的 cross-entropy.                                    #
    #                                                                     #
    #   loss_i2t = F.cross_entropy(logits_per_image, labels)               #
    #   loss_t2i = F.cross_entropy(logits_per_text,  labels)               #
    #                                                                     #
    # F.cross_entropy 已经把 "log_softmax + 按 label 取对角 + 取平均"        #
    # 这三步合在一起, 不需要手写.                                            #
    # 它内部用 log-sum-exp trick, 数值稳定 (跟 numpy gold standard 一致).    #
    # ------------------------------------------------------------------ #
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    # ------------------------------------------------------------------ #
    # TODO 3 — symmetric: 平均两个方向, return 一个 scalar tensor.          #
    #                                                                     #
    #   return 0.5 * (loss_i2t + loss_t2i)                                #
    # ------------------------------------------------------------------ #
    return 0.5 * (loss_i2t + loss_t2i)


# ---------------------------------------------------------------------------- #
# Driver: boundary 行为 smoke check                                              #
# ---------------------------------------------------------------------------- #


def main():
    import math

    # 边界 1: logits 全 0 → softmax 每行均匀 1/B → loss = log(B).
    # 这是 InfoNCE 在 "啥都没学" 状态下的理论 baseline.
    B = 5
    z = torch.zeros(B, B)
    loss_uniform = info_nce_symmetric(z)
    print(f"uniform logits → loss = {loss_uniform.item():.6f}    "
          f"(expected log({B}) = {math.log(B):.6f})")

    # 边界 2: identity * 大 scale → 对角线主导 → loss → 0.
    # 这是 InfoNCE 的 lower bound, 完美对齐.
    logits_strong = torch.eye(B, dtype=torch.float64) * 50.0
    loss_strong = info_nce_symmetric(logits_strong)
    print(f"strong diagonal → loss = {loss_strong.item():.2e}    (expected ~0)")

    # 边界 3: symmetric in transpose — L(S) = L(Sᵀ)
    torch.manual_seed(0)
    logits = torch.randn(B, B, dtype=torch.float64) * 2.0
    loss_S = info_nce_symmetric(logits).item()
    loss_ST = info_nce_symmetric(logits.T).item()
    print(f"transpose invariance: |L(S) - L(Sᵀ)| = {abs(loss_S - loss_ST):.2e}    "
          f"(expected < 1e-12 在 float64)")


if __name__ == "__main__":
    main()
