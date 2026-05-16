"""
M4.3 — Visualize CLIP similarity matrix (before vs after training).

输出
----
outputs/<run_name>/similarity.png  : 1×2 子图
    左:  untrained random init 的 (B, B) similarity heatmap
    右:  训完 (load best.pt) 的 (B, B) similarity heatmap

读图的方法:
    - 训练前应该是个 noise 矩阵, 对角线和非对角线没明显差异
    - 训练后对角线应该明显比非对角亮 (一条对角线的"亮线"出现)
    - 如果训练后还是 noise → 训练失败 (loss 没下降 / temperature 失控 / 数据有问题)
    - 如果训练后整张图都很均匀 (没有亮对角) → 温度太大 / 学习率太小 / encoder 容量不足

可视化的小细节: 我们用 batch 内固定 16 个样本 (4 张图 × 每个 class 不重复采)
来画 heatmap, 这样不同 epoch 横向对比时 "意义" 一致.

Run
---
    python visualize_similarity.py --run_name cifar_sanity
"""

import argparse
import sys
from pathlib import Path

# 让 src/ 可 import
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.pyplot as plt
import numpy as np
import torch

from image_encoder import ImageEncoderViT
from text_encoder import TextTransformer
from clip_model import CLIPModel
from data import (
    CIFAR10ClipDataset, CIFAR10_CLASSES, TEMPLATES,
    build_cifar_tokenizer,
)


def gather_eval_batch(B: int = 10, tokenizer=None, max_len: int = 16):
    """
    取一个固定的 eval batch — 每个 class 一张图, 共 10 张.
    每张图用同一个 template "a photo of a {class}", 这样 visualize 时
    对角线对应的 caption 就是 image 的真实类别.

    Returns
    -------
    images    : (10, 3, 32, 32)
    token_ids : (10, max_len)
    eos_pos   : (10,)
    captions  : List[str]    # 用来画 axis label
    classes   : List[str]
    """
    if tokenizer is None:
        tokenizer = build_cifar_tokenizer()

    ds = CIFAR10ClipDataset(train=False, tokenizer=tokenizer,
                            max_len=max_len, augment=False, seed=0)

    # 每个 class 取一张图. ds.cifar.targets 是 label list.
    targets = ds.cifar.targets
    per_class_idx = {}
    for i, t in enumerate(targets):
        if t not in per_class_idx:
            per_class_idx[t] = i
        if len(per_class_idx) == 10:
            break

    images_list = []
    captions_list = []
    classes_list = []
    token_ids_list = []
    eos_pos_list = []

    # 用最干净的模板, 不要随机
    template = "a photo of a {}"

    for class_idx in range(10):
        idx = per_class_idx[class_idx]
        image, _ = ds.cifar[idx]   # 注意: 直接走 ds.cifar (跳过 ds 的随机 template)
        images_list.append(image)

        cls_name = CIFAR10_CLASSES[class_idx]
        caption = template.format(cls_name)
        captions_list.append(caption)
        classes_list.append(cls_name)

        ids, eos = tokenizer.encode(caption, max_len)
        token_ids_list.append(ids)
        eos_pos_list.append(eos)

    images = torch.stack(images_list)
    token_ids = torch.stack(token_ids_list)
    eos_pos = torch.tensor(eos_pos_list, dtype=torch.long)
    return images, token_ids, eos_pos, captions_list, classes_list


@torch.no_grad()
def compute_similarity(model, images, token_ids, eos_pos):
    """
    Returns (B, B) cosine similarity matrix (没乘 temperature, 因为
    可视化只关心相对结构, 不关心绝对尺度).
    """
    model.eval()
    img_feat = model.encode_image(images)
    txt_feat = model.encode_text(token_ids, eos_pos)
    sim = img_feat @ txt_feat.T                # (B, B), L2-normed 所以 = cosine
    return sim.cpu().numpy()


def plot_two_heatmaps(sim_before: np.ndarray, sim_after: np.ndarray,
                      classes: list, save_path: Path):
    """
    Side-by-side heatmap. 两边用同一个 colormap range 方便对比.
    """
    vmin = min(sim_before.min(), sim_after.min())
    vmax = max(sim_before.max(), sim_after.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, sim, title in zip(axes,
                               [sim_before, sim_after],
                               ["Before training (random init)",
                                "After training (best ckpt)"]):
        im = ax.imshow(sim, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels([f'"a photo of a {c}"' for c in classes],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel("Text features (per class)")
        ax.set_ylabel("Image features")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # 标对角线 diag mean / off mean
        diag = np.diag(sim).mean()
        off_mask = ~np.eye(len(sim), dtype=bool)
        off = sim[off_mask].mean()
        ax.text(0.02, 0.98,
                f"diag={diag:+.3f}\noff={off:+.3f}\ngap={diag - off:+.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    plt.suptitle("CLIP similarity matrix: image features × text features",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {save_path}")


def build_model_from_ckpt(ckpt_path: Path, tokenizer):
    """根据 ckpt 里的 args 重建 model 然后 load state_dict."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    image_encoder = ImageEncoderViT(
        img_size=32, patch_size=args["patch_size"], in_chans=3,
        d_model=args["img_d_model"], d_shared=args["d_shared"],
        depth=args["img_depth"], num_heads=args["img_num_heads"],
        dropout=0.0,
    )
    text_encoder = TextTransformer(
        vocab_size=tokenizer.vocab_size, max_len=args["max_len"],
        d_model=args["txt_d_model"], d_shared=args["d_shared"],
        depth=args["txt_depth"], num_heads=args["txt_num_heads"],
        dropout=0.0,
    )
    model = CLIPModel(image_encoder, text_encoder,
                      init_temperature=args["init_temperature"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model, args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="cifar_sanity",
                        help="对应 train_cifar.py 的 --run_name")
    parser.add_argument("--max_len", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / args.run_name
    ckpt_path = out_dir / "best.pt"
    save_path = out_dir / "similarity.png"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"找不到 ckpt: {ckpt_path}\n"
            f"先跑 train_cifar.py --run_name {args.run_name}"
        )

    # 准备 fixed eval batch
    tokenizer = build_cifar_tokenizer()
    images, token_ids, eos_pos, captions, classes = gather_eval_batch(
        B=10, tokenizer=tokenizer, max_len=args.max_len,
    )

    print("Eval batch (每个 class 一张图):")
    for c, cap in zip(classes, captions):
        print(f"  [{c:12s}] caption = '{cap}'")
    print()

    # ---- before training (random init) ------------------------------------ #
    # 重建一个空 model (用 ckpt 里的 args, 但不 load state_dict)
    _, train_args = build_model_from_ckpt(ckpt_path, tokenizer)  # 只拿 args
    image_encoder = ImageEncoderViT(
        img_size=32, patch_size=train_args["patch_size"], in_chans=3,
        d_model=train_args["img_d_model"], d_shared=train_args["d_shared"],
        depth=train_args["img_depth"], num_heads=train_args["img_num_heads"],
        dropout=0.0,
    )
    text_encoder = TextTransformer(
        vocab_size=tokenizer.vocab_size, max_len=train_args["max_len"],
        d_model=train_args["txt_d_model"], d_shared=train_args["d_shared"],
        depth=train_args["txt_depth"], num_heads=train_args["txt_num_heads"],
        dropout=0.0,
    )
    model_random = CLIPModel(image_encoder, text_encoder,
                              init_temperature=train_args["init_temperature"])
    torch.manual_seed(0)
    # 重置 random encoder 让结果可复现
    sim_before = compute_similarity(model_random, images, token_ids, eos_pos)

    # ---- after training --------------------------------------------------- #
    model_trained, _ = build_model_from_ckpt(ckpt_path, tokenizer)
    sim_after = compute_similarity(model_trained, images, token_ids, eos_pos)

    # ---- plot ------------------------------------------------------------- #
    plot_two_heatmaps(sim_before, sim_after, classes, save_path)

    # 文字总结
    print()
    print("Similarity gap (diag_mean - offdiag_mean):")
    for sim, label in [(sim_before, "Before"), (sim_after, "After")]:
        diag = np.diag(sim).mean()
        off = sim[~np.eye(len(sim), dtype=bool)].mean()
        print(f"  {label:7s}: diag={diag:+.4f}  off={off:+.4f}  gap={diag - off:+.4f}")


if __name__ == "__main__":
    main()
