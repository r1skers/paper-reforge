"""
M4.2 — Train CLIP on CIFAR-10 with template captions.

这是 CLIP pipeline 的 sanity check (不是真正的 retrieval 训练).
目标:
    1. 跑通完整 forward / backward / optimizer 链路
    2. loss 真的从 ~log(B) 下降
    3. similarity matrix 对角线均值 > 非对角均值, 且 gap 随 epoch 增长

CPU 时间预估
-----------
小 config (d_model=128, depth=2, batch=128):
    50k images / 128 batch ≈ 390 step/epoch
    CPU per-step ~0.5-1.5s
    单 epoch ~3-10 min
推荐先 --epochs 3 跑一遍 sanity, 再决定要不要长跑.

输出
----
outputs/<run_name>/
    log.csv           : per-epoch loss / temperature / sim diagonal vs off-diag
    best.pt           : 最好 epoch 的 model state
    args.txt          : 本次运行的配置

Run
---
    # 快速 sanity (3 epoch, 小 config)
    python train_cifar.py --epochs 3 --run_name smoke

    # 标准跑 (10 epoch)
    python train_cifar.py --epochs 10 --run_name default
"""

import argparse
import csv
import sys
import time
from pathlib import Path

# 让 src/ 可 import
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.optim as optim

from image_encoder import ImageEncoderViT
from text_encoder import TextTransformer
from clip_model import CLIPModel
from loss import info_nce_symmetric
from data import get_cifar_clip_loaders


# ---------------------------------------------------------------------------- #
# Training primitives                                                          #
# ---------------------------------------------------------------------------- #


def train_one_epoch(model, loader, optimizer, device, log_every=50):
    """
    一个 epoch 的训练. 返回该 epoch 的 (avg_loss, last_temperature).

    对每个 batch 做的事:
        1. 三 tensor 搬到 device
        2. forward → (logits_per_image, logits_per_text)
        3. info_nce_symmetric loss
        4. 反传 + step + zero_grad
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (images, token_ids, eos_pos) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        token_ids = token_ids.to(device, non_blocking=True)
        eos_pos = eos_pos.to(device, non_blocking=True)

        # ------------------------------------------------------------------ #
        # TODO 1 — 三步 optimizer 更新 (跟 ViT 一致, 只是 forward / loss 不同).  #
        #                                                                     #
        #   optimizer.zero_grad()                                              #
        #   logits_i2t, logits_t2i = model(images, token_ids, eos_pos)         #
        #   loss = info_nce_symmetric(logits_i2t, logits_t2i)                  #
        #   loss.backward()                                                    #
        #   optimizer.step()                                                   #
        #                                                                     #
        # NOTE: 不再像 ViT 那样有 "accuracy" 这个 metric — contrastive 不算     #
        # accuracy. 看 loss + similarity gap 即可 (后者在 evaluate 算).         #
        # ------------------------------------------------------------------ #
        optimizer.zero_grad()
        logits_i2t, logits_t2i = model(images, token_ids, eos_pos)
        loss = info_nce_symmetric(logits_i2t, logits_t2i)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % log_every == 0:
            tau = model.get_temperature()
            print(f"    step {batch_idx:4d}/{len(loader)}  "
                  f"loss={loss.item():.4f}  τ={tau:.4f}")

    return total_loss / n_batches, model.get_temperature()


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Eval: 在 test loader 上跑一遍, 返回:
        - avg_loss        : 平均 InfoNCE loss
        - sim_diag_mean   : similarity matrix 对角线均值 (越大说明匹配 pair 越近)
        - sim_offdiag_mean: 非对角均值 (越小说明负 pair 越远)
        - sim_gap         : diag_mean - offdiag_mean (越大说明 alignment 越好)

    sim_gap 是 toy-scale CLIP 学习进度最直接的 metric.
    训练完美时 diag_mean → 1.0, offdiag → 0 附近, gap → 1.0.
    """
    model.eval()
    total_loss = 0.0
    total_diag = 0.0
    total_offdiag = 0.0
    n_batches = 0
    n_offdiag = 0

    for images, token_ids, eos_pos in loader:
        images = images.to(device, non_blocking=True)
        token_ids = token_ids.to(device, non_blocking=True)
        eos_pos = eos_pos.to(device, non_blocking=True)

        # ------------------------------------------------------------------ #
        # TODO 2 — Eval forward + 收集 metric.                                  #
        #                                                                     #
        #   logits_i2t, logits_t2i = model(images, token_ids, eos_pos)         #
        #   loss = info_nce_symmetric(logits_i2t, logits_t2i)                  #
        #                                                                     #
        # 算 similarity (不要带 temperature, 我们要看原始 cosine):              #
        #   img_feat = model.encode_image(images)         # 已 L2 normalize    #
        #   txt_feat = model.encode_text(token_ids, eos_pos)                  #
        #   sim = img_feat @ txt_feat.T                    # (B, B), ∈ [-1, 1]  #
        #                                                                     #
        # 取对角和非对角:                                                       #
        #   B = sim.shape[0]                                                   #
        #   diag = sim.diagonal()                          # (B,)              #
        #   off_mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)      #
        #   off = sim[off_mask]                            # (B*(B-1),)        #
        #                                                                     #
        # 累加:                                                                #
        #   total_loss    += loss.item()                                       #
        #   total_diag    += diag.sum().item()                                 #
        #   total_offdiag += off.sum().item()                                  #
        #   n_batches += 1                                                     #
        #   n_offdiag += off.numel()    # 每个 batch 累计 B*(B-1) 个负样本    #
        #                                                                     #
        # (注: 这里直接拿 batch 内 B 个样本算 diag/off — 不是 dataset-wide     #
        # 的 retrieval Recall, 而是 batch-level "对角线-非对角"差距.           #
        # 真正的 retrieval Recall 留给 M5.)                                    #
        # ------------------------------------------------------------------ #
        logits_i2t, logits_t2i = model(images, token_ids, eos_pos)
        loss = info_nce_symmetric(logits_i2t, logits_t2i)

        img_feat = model.encode_image(images)
        txt_feat = model.encode_text(token_ids, eos_pos)
        sim = img_feat @ txt_feat.T

        B = sim.shape[0]
        diag = sim.diagonal()
        off_mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
        off = sim[off_mask]

        total_loss += loss.item()
        total_diag += diag.sum().item()
        total_offdiag += off.sum().item()
        n_batches += 1
        n_offdiag += off.numel()

    avg_loss = total_loss / n_batches
    # 对角均值: 每 batch B 个对角, 总 batch * B 个 — 但每 batch B 不固定 (最后
    # 一个 batch 可能少几个), 所以更稳是用累加 / 总样本数.
    n_total_diag = sum(min(loader.batch_size, len(loader.dataset)
                            - i * loader.batch_size)
                       for i in range(n_batches))
    diag_mean = total_diag / n_total_diag
    off_mean = total_offdiag / n_offdiag
    return avg_loss, diag_mean, off_mean, diag_mean - off_mean


# ---------------------------------------------------------------------------- #
# Main                                                                          #
# ---------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_len",   type=int, default=16)

    # Image encoder
    p.add_argument("--img_d_model",   type=int, default=128)
    p.add_argument("--img_depth",     type=int, default=2)
    p.add_argument("--img_num_heads", type=int, default=4)
    p.add_argument("--patch_size",    type=int, default=4)

    # Text encoder
    p.add_argument("--txt_d_model",   type=int, default=128)
    p.add_argument("--txt_depth",     type=int, default=2)
    p.add_argument("--txt_num_heads", type=int, default=4)

    # Shared
    p.add_argument("--d_shared",  type=int,   default=128)
    p.add_argument("--dropout",   type=float, default=0.0)
    p.add_argument("--init_temperature", type=float, default=0.2)

    # Optim
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)

    # Misc
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--run_name", type=str, default="cifar_sanity")
    p.add_argument("--device",   type=str, default="cpu")
    p.add_argument("--log_every", type=int, default=50)

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 60)
    print(f"CLIP M4 — CIFAR-10 + template captions sanity training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Args:   {vars(args)}")
    print()

    # ---- Data ------------------------------------------------------------- #
    print("Building data loaders ...")
    tokenizer, train_loader, test_loader = get_cifar_clip_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_len=args.max_len, augment_train=True,
    )
    print(f"  vocab_size       = {tokenizer.vocab_size}")
    print(f"  len(train_set)   = {len(train_loader.dataset)}")
    print(f"  len(test_set)    = {len(test_loader.dataset)}")
    print(f"  steps per epoch  = {len(train_loader)}")
    print()

    # ---- Model ------------------------------------------------------------ #
    image_encoder = ImageEncoderViT(
        img_size=32, patch_size=args.patch_size, in_chans=3,
        d_model=args.img_d_model, d_shared=args.d_shared,
        depth=args.img_depth, num_heads=args.img_num_heads,
        dropout=args.dropout,
    )
    text_encoder = TextTransformer(
        vocab_size=tokenizer.vocab_size, max_len=args.max_len,
        d_model=args.txt_d_model, d_shared=args.d_shared,
        depth=args.txt_depth, num_heads=args.txt_num_heads,
        dropout=args.dropout,
    )
    model = CLIPModel(
        image_encoder, text_encoder,
        init_temperature=args.init_temperature,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: CLIP with {n_params:,} params")
    print(f"  init τ = {args.init_temperature}    "
          f"(logit_scale = log(1/τ) = {model.logit_scale.item():.4f})")
    print()

    # ---- Optim ------------------------------------------------------------ #
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Logging setup ---------------------------------------------------- #
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "log.csv"
    ckpt_path = out_dir / "best.pt"
    args_path = out_dir / "args.txt"

    with open(args_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "test_loss",
            "sim_diag_mean", "sim_offdiag_mean", "sim_gap",
            "temperature", "lr", "epoch_seconds",
        ])

    # ---- 初始 baseline (epoch 0, 没训练) ----------------------------------- #
    import math
    print("Evaluating untrained model (baseline) ...")
    te_loss, diag_m, off_m, gap = evaluate(model, test_loader, device)
    print(f"  baseline test_loss = {te_loss:.4f}    "
          f"(uniform-prior baseline ≈ log({args.batch_size}) = {math.log(args.batch_size):.4f})")
    print(f"  sim diag_mean    = {diag_m:+.4f}")
    print(f"  sim offdiag_mean = {off_m:+.4f}")
    print(f"  sim gap          = {gap:+.4f}    (训练前应该 ≈ 0)")
    print()

    best_gap = -float("inf")

    # ---- Train loop ------------------------------------------------------- #
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"--- Epoch {epoch}/{args.epochs} ---")
        tr_loss, cur_tau = train_one_epoch(
            model, train_loader, optimizer, device, log_every=args.log_every,
        )
        te_loss, diag_m, off_m, gap = evaluate(model, test_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  train_loss = {tr_loss:.4f}    test_loss = {te_loss:.4f}")
        print(f"  sim diag={diag_m:+.4f}  off={off_m:+.4f}  gap={gap:+.4f}")
        print(f"  τ={cur_tau:.4f}  lr={cur_lr:.2e}  t={elapsed:.1f}s")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, tr_loss, te_loss,
                diag_m, off_m, gap,
                cur_tau, cur_lr, elapsed,
            ])

        if gap > best_gap:
            best_gap = gap
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "vocab_size": tokenizer.vocab_size,
                "args": vars(args),
                "sim_gap": gap,
            }, ckpt_path)
            print(f"  → new best  sim_gap={gap:+.4f}  saved to {ckpt_path}")
        print()

    print(f"Done. Best sim_gap = {best_gap:+.4f}")
    print(f"Logs:       {csv_path}")
    print(f"Best ckpt:  {ckpt_path}")


if __name__ == "__main__":
    main()
