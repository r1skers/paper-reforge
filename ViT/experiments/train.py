"""
V4 — Train ViT on MNIST (smoke test) or CIFAR-10.

Default config is tuned for CPU on a small ViT — should run MNIST 10 epochs
in roughly 20–40 minutes on a modern laptop CPU.

Logs per-epoch train/test loss and accuracy to stdout and to a CSV file
under ../outputs/<run_name>/.

Run
---
    # MNIST defaults (recommended first run):
    python train.py

    # CIFAR-10:
    python train.py --dataset cifar10 --patch_size 4 --img_size 32 \
                    --in_chans 3 --epochs 30

Key design choices baked in:
    - AdamW optimizer  (ViT standard)
    - Cosine annealing LR schedule
    - Cross-entropy loss
    - Best-checkpoint saving (by test accuracy)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

# Make src/ importable.
_SRC = Path(__file__).resolve().parent.parent / "src"
_DATA = Path(__file__).resolve().parent.parent / "src" / "data.py"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn as nn

from vit import ViT
from data import get_mnist_loaders, get_cifar10_loaders


# ---------------------------------------------------------------------------- #
# Train / eval primitives                                                      #
# ---------------------------------------------------------------------------- #


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Run one full pass over the training data. Returns (avg_loss, accuracy).

    accuracy is the running fraction of correct predictions.
    """
    model.train()
    total_loss = 0.0
    n_correct = 0
    n_seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # ------------------------------------------------------------------ #
        # TODO 1 — The three-step optimizer update.                           #
        #                                                                     #
        #   optimizer.zero_grad()                                              #
        #   logits = model(x)                       # (B, num_classes)         #
        #   loss = criterion(logits, y)                                        #
        #   loss.backward()                                                    #
        #   optimizer.step()                                                   #
        #                                                                     #
        # IMPORTANT: zero_grad BEFORE forward, not after — gradients          #
        # accumulate by default in PyTorch.                                    #
        # ------------------------------------------------------------------ #
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        # ------------------------------------------------------------------ #
        # TODO 2 — Accumulate running stats for the epoch.                    #
        #                                                                     #
        #   total_loss += loss.item() * x.size(0)        # weighted by batch  #
        #   preds = logits.argmax(dim=1)                                       #
        #   n_correct += (preds == y).sum().item()                             #
        #   n_seen    += x.size(0)                                             #
        #                                                                     #
        # We multiply loss.item() by batch size so the final average is       #
        # correct even if the last batch is smaller than the rest.            #
        # ------------------------------------------------------------------ #
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        n_correct += (preds == y).sum().item()
        n_seen    += x.size(0)

    avg_loss = total_loss / n_seen
    acc = n_correct / n_seen
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on a loader. No gradients, no parameter updates.
    Returns (avg_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # ------------------------------------------------------------------ #
        # TODO 3 — Eval forward + stats (no backward, no step).               #
        #                                                                     #
        #   logits = model(x)                                                  #
        #   loss   = criterion(logits, y)                                      #
        #                                                                     #
        #   total_loss += loss.item() * x.size(0)                              #
        #   preds = logits.argmax(dim=1)                                       #
        #   n_correct += (preds == y).sum().item()                             #
        #   n_seen    += x.size(0)                                             #
        # ------------------------------------------------------------------ #
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        n_correct += (preds == y).sum().item()
        n_seen    += x.size(0)
    avg_loss = total_loss / n_seen
    acc = n_correct / n_seen
    return avg_loss, acc


# ---------------------------------------------------------------------------- #
# Main                                                                          #
# ---------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--dataset",    type=str, default="mnist",
                   choices=["mnist", "cifar10"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)

    # Model
    p.add_argument("--img_size",   type=int, default=28)
    p.add_argument("--patch_size", type=int, default=7)
    p.add_argument("--in_chans",   type=int, default=1)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--d_model",    type=int, default=64)
    p.add_argument("--depth",      type=int, default=4)
    p.add_argument("--num_heads",  type=int, default=4)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--activation", type=str, default="gelu")

    # Optim
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)

    # Misc
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--run_name", type=str, default="default")
    p.add_argument("--device",   type=str, default="cpu")

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Args:   {vars(args)}")

    # ---- Data ------------------------------------------------------------- #
    if args.dataset == "mnist":
        train_loader, test_loader = get_mnist_loaders(
            batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # ---- Model ------------------------------------------------------------ #
    model = ViT(
        img_size=args.img_size, patch_size=args.patch_size,
        in_chans=args.in_chans, num_classes=args.num_classes,
        d_model=args.d_model, depth=args.depth, num_heads=args.num_heads,
        dropout=args.dropout, activation=args.activation,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ViT with {n_params:,} params  "
          f"(N={model.patch_embed.num_patches}, d={args.d_model}, "
          f"L={args.depth}, h={args.num_heads})")

    # ---- Optim + schedule + loss ----------------------------------------- #
    # ------------------------------------------------------------------ #
    # TODO 4 — Build optimizer, scheduler, and loss.                      #
    #                                                                     #
    #   optimizer = torch.optim.AdamW(model.parameters(),                  #
    #                                 lr=args.lr,                          #
    #                                 weight_decay=args.weight_decay)      #
    #                                                                     #
    #   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(             #
    #       optimizer, T_max=args.epochs)                                  #
    #   # CosineAnnealingLR: lr starts at args.lr, decays to 0 by epoch     #
    #   # T_max via half-cosine. One .step() per epoch.                     #
    #                                                                     #
    #   criterion = nn.CrossEntropyLoss()                                   #
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ---- Logging setup ---------------------------------------------------- #
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "log.csv"
    ckpt_path = out_dir / "best.pt"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "test_loss", "test_acc", "lr", "epoch_seconds"])

    best_acc = 0.0

    # ---- Train loop ------------------------------------------------------- #
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        epoch_seconds = time.time() - t0

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[ep {epoch:3d}/{args.epochs}] "
              f"train  loss={tr_loss:.4f}  acc={tr_acc*100:.2f}%   "
              f"test  loss={te_loss:.4f}  acc={te_acc*100:.2f}%   "
              f"lr={cur_lr:.2e}   t={epoch_seconds:.1f}s")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr_loss, tr_acc, te_loss, te_acc, cur_lr, epoch_seconds])

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "test_acc": te_acc,
                "args": vars(args),
            }, ckpt_path)
            print(f"  → new best  test_acc={te_acc*100:.2f}%  saved to {ckpt_path}")

    print(f"\nDone. Best test_acc = {best_acc*100:.2f}%")
    print(f"Logs: {csv_path}")
    print(f"Best ckpt: {ckpt_path}")


if __name__ == "__main__":
    main()
