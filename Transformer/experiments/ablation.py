"""
T4.4 — Ablation sweep for ArgmaxPositionModel.

For each variant we train from scratch with the same seed and the same
training schedule, then report:
  - final accuracy (large-sample eval)
  - steps_to_95: first log step whose val_acc reached >= 0.95
                 (None means never reached within num_steps)
  - wall-clock time

Run
---
    python experiments/ablation.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model import ArgmaxPositionModel
from data import gen_argmax_batch


# --------------------------------------------------------------------------- #
# Config + variant sweep                                                      #
# --------------------------------------------------------------------------- #

BASE_CONFIG = {
    "vocab_size":   10,
    "n":            100,
    "d_model":      32,
    "num_heads":    4,
    "num_layers":   2,
    "dropout":      0.0,
    "use_pe":       True,
    "batch_size":   64,
    "num_steps":    2000,
    "lr":           1e-3,
    "log_interval": 100,
    "eval_batches": 20,
    "seed":         0,
}

# (name, override-dict) — each variant inherits BASE_CONFIG then applies overrides.
VARIANTS = [
    ("baseline",  {}),
    ("no_pe",     {"use_pe": False}),
    ("1_layer",   {"num_layers": 1}),
    ("4_layers",  {"num_layers": 4}),
    ("1_head",    {"num_heads": 1}),
    ("8_heads",   {"num_heads": 8}),
]


# --------------------------------------------------------------------------- #
# Eval helper (same as train.py)                                              #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model, n_batches, B, n, vocab_size, device):
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
# Single-run training                                                         #
# --------------------------------------------------------------------------- #

def train_one_run(cfg, device):
    """Train one variant; return dict of {final_acc, final_loss, steps_to_95, time_s}."""
    torch.manual_seed(cfg["seed"])

    model = ArgmaxPositionModel(
        vocab_size=cfg["vocab_size"],
        n_max=cfg["n"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        use_pe=cfg["use_pe"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    steps_to_95 = None
    start = time.time()

    for step in range(1, cfg["num_steps"] + 1):
        model.train()
        x, y = gen_argmax_batch(cfg["batch_size"], cfg["n"],
                                cfg["vocab_size"], device=device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % cfg["log_interval"] == 0:
            _, val_acc = evaluate(model, cfg["eval_batches"],
                                  cfg["batch_size"], cfg["n"],
                                  cfg["vocab_size"], device)
            if steps_to_95 is None and val_acc >= 0.95:
                steps_to_95 = step

    # Big-sample final eval
    final_loss, final_acc = evaluate(model, cfg["eval_batches"] * 5,
                                     cfg["batch_size"], cfg["n"],
                                     cfg["vocab_size"], device)
    return {
        "final_acc":    final_acc,
        "final_loss":   final_loss,
        "steps_to_95":  steps_to_95,
        "time_s":       time.time() - start,
    }


# --------------------------------------------------------------------------- #
# Main sweep                                                                  #
# --------------------------------------------------------------------------- #

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    print(f"running {len(VARIANTS)} variants (each ~10-30s on CPU)")
    print()

    results = []
    for name, override in VARIANTS:
        cfg = {**BASE_CONFIG, **override}
        print(f"--- {name}  (use_pe={cfg['use_pe']}, "
              f"layers={cfg['num_layers']}, heads={cfg['num_heads']}) ---")
        result = train_one_run(cfg, device)
        s95 = result["steps_to_95"] if result["steps_to_95"] is not None else "----"
        print(f"    final_acc={result['final_acc']:.3f}   "
              f"steps_to_95={s95}   time={result['time_s']:.1f}s")
        print()
        results.append((name, cfg, result))

    # ---- Comparison table ----
    print()
    print("=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)
    print(f"{'variant':<12}{'use_pe':<10}{'layers':<10}{'heads':<10}"
          f"{'final_acc':<14}{'steps_to_95':<14}{'time(s)':<10}")
    print("-" * 80)
    for name, cfg, r in results:
        s95 = str(r["steps_to_95"]) if r["steps_to_95"] is not None else "----"
        print(f"{name:<12}{str(cfg['use_pe']):<10}{cfg['num_layers']:<10}"
              f"{cfg['num_heads']:<10}{r['final_acc']:<14.3f}{s95:<14}"
              f"{r['time_s']:<10.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
