"""
T4.3 — Training script for the argmax-position Transformer.

Run
---
    python experiments/train.py

Expected
--------
- Starting accuracy ~ 1/n (random), ~5% for n=20.
- Should climb to 95%+ within 1000-2000 steps.
- CPU on a modern Ryzen: ~1-2 minutes total.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Make `src/` importable.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model import ArgmaxPositionModel
from data import gen_argmax_batch


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

CONFIG = {
    # task
    "vocab_size": 10,
    "n":          20,
    # model
    "d_model":    32,
    "num_heads":  4,
    "num_layers": 2,
    "dropout":    0.0,
    "use_pe":     True,
    # training
    "batch_size":   64,
    "num_steps":    2000,
    "lr":           1e-3,
    "log_interval": 100,
    "eval_batches": 20,        # how many batches per eval call
    "seed":         0,
}


# --------------------------------------------------------------------------- #
# Evaluation helper                                                           #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model, n_batches, B, n, vocab_size, device):
    """Return (mean_loss, accuracy) over n_batches freshly sampled batches."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for _ in range(n_batches):
        x, y = gen_argmax_batch(B, n, vocab_size, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)

        total_loss += loss.item() * B
        total_correct += (pred == y).sum().item()
        total_samples += B
    return total_loss / total_samples, total_correct / total_samples


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #

def main():
    cfg = CONFIG

    # Device — CPU on AMD, GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # Seed for reproducibility.
    torch.manual_seed(cfg["seed"])

    # Build model.
    model = ArgmaxPositionModel(
        vocab_size=cfg["vocab_size"],
        n_max=cfg["n"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        use_pe=cfg["use_pe"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params = {n_params:,}")

    # Optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Initial accuracy (should be ~ 1/n).
    init_loss, init_acc = evaluate(model, cfg["eval_batches"],
                                   cfg["batch_size"], cfg["n"],
                                   cfg["vocab_size"], device)
    print(f"step    0:  init_loss = {init_loss:.4f}   init_acc = {init_acc:.3f}   "
          f"(random baseline ~ {1/cfg['n']:.3f})")

    # ---- Training loop --------------------------------------------------- #
    start_time = time.time()
    for step in range(1, cfg["num_steps"] + 1):
        model.train()

        # Sample fresh batch (synthetic — no need for a DataLoader).
        x, y = gen_argmax_batch(cfg["batch_size"], cfg["n"],
                                cfg["vocab_size"], device=device)

        # ------------------------------------------------------------------ #
        # TODO 1 — Forward pass and loss.                                     #
        #                                                                     #
        #   logits = model(x)                       # (B, n)                   #
        #   loss   = F.cross_entropy(logits, y)     # scalar                   #
        #                                                                     #
        # Note: logits shape is (B, n) — n positions act as "n_classes".      #
        # F.cross_entropy expects (B, num_classes) and target (B,).            #
        # That matches our setup exactly — no reshape needed.                 #
        # ------------------------------------------------------------------ #
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        # ------------------------------------------------------------------ #
        # TODO 2 — The three-step optimizer update (THE PyTorch idiom).       #
        #                                                                     #
        #   optimizer.zero_grad()        # clear last step's .grad             #
        #   loss.backward()              # populate new .grad via autograd     #
        #   optimizer.step()             # apply Adam update using .grad       #
        #                                                                     #
        # Order matters.  Skipping zero_grad accumulates gradients across     #
        # steps (sometimes intentional in grad-accumulation training, but     #
        # NEVER what you want by default).                                    #
        # ------------------------------------------------------------------ #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Periodic logging.
        if step % cfg["log_interval"] == 0:
            val_loss, val_acc = evaluate(model, cfg["eval_batches"],
                                         cfg["batch_size"], cfg["n"],
                                         cfg["vocab_size"], device)
            elapsed = time.time() - start_time
            print(f"step {step:>4d}:  train_loss = {loss.item():.4f}   "
                  f"val_loss = {val_loss:.4f}   val_acc = {val_acc:.3f}   "
                  f"[{elapsed:.1f}s]")

    # Final summary.
    final_loss, final_acc = evaluate(model, cfg["eval_batches"] * 5,
                                     cfg["batch_size"], cfg["n"],
                                     cfg["vocab_size"], device)
    print()
    print(f"final accuracy = {final_acc:.3f}   (target: >= 0.95)")
    print(f"total time     = {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
