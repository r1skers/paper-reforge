"""
T4.5 — Visualize learned attention patterns for both tasks.

What this does
--------------
1. Train a baseline (PE on, 2 layers, 4 heads) on each of the two tasks
   (argmax and second-max).
2. Pick one example sequence and run forward, capturing per-layer
   attention weights via forward hooks.
3. Render a figure per task: input sequence (bar chart) + per-layer
   head-averaged attention map.

Saves PNGs to  experiments/figures/

Run
---
    python experiments/visualize_attention.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model import ArgmaxPositionModel
from data import gen_argmax_batch, gen_second_max_batch


# --------------------------------------------------------------------------- #
# Training (compact, matches ablation.py baseline config)                     #
# --------------------------------------------------------------------------- #

def train_baseline(task_fn, vocab_size=10, n=20, num_steps=1000,
                   d_model=32, num_heads=4, num_layers=2, device="cpu"):
    """Train a baseline (PE on) model on a given task_fn data generator."""
    torch.manual_seed(0)
    model = ArgmaxPositionModel(
        vocab_size=vocab_size, n_max=n,
        d_model=d_model, num_heads=num_heads, num_layers=num_layers,
        dropout=0.0, use_pe=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(num_steps):
        model.train()
        x, y = task_fn(64, n, vocab_size, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


# --------------------------------------------------------------------------- #
# Capture attention via forward hooks                                         #
# --------------------------------------------------------------------------- #

def capture_attention(model, x):
    """
    Run model.forward(x) once and return (logits, list_of_attn).

    Each attn entry has shape (B, h, n, n) — one tensor per encoder block.
    We use forward hooks so the model code stays untouched.
    """
    captured = {}

    def make_hook(idx):
        def hook(module, inputs, outputs):
            # outputs is (out_tensor, attn_tensor) from MultiHeadSelfAttention
            captured[idx] = outputs[1].detach().cpu()
        return hook

    handles = []
    for i, block in enumerate(model.blocks):
        h = block.attn.register_forward_hook(make_hook(i))
        handles.append(h)

    model.eval()
    with torch.no_grad():
        logits = model(x)

    for h in handles:
        h.remove()

    return logits, [captured[i] for i in range(len(model.blocks))]


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #

def plot_attention(x_sample, y_label, pred_label, attns, title, save_path,
                   max_token=9):
    """
    Render a 1-row figure: input bar chart + one head-averaged attention
    matrix per layer.
    """
    n = x_sample.shape[0]
    num_layers = len(attns)
    fig, axes = plt.subplots(1, num_layers + 1,
                             figsize=(4.5 * (num_layers + 1), 4.2))

    # ---- Panel 0: input sequence ---- #
    ax = axes[0]
    x_list = x_sample.tolist()
    colors = ["crimson" if v == max_token else "steelblue" for v in x_list]
    ax.bar(range(n), x_list, color=colors)
    ax.axvline(y_label,    color="green",  linestyle="--",
               label=f"label = {y_label}")
    ax.axvline(pred_label, color="orange", linestyle=":",
               label=f"pred  = {pred_label}")
    ax.set_xlabel("position")
    ax.set_ylabel("token value")
    ax.set_title("input sequence (red = max token)")
    ax.legend(loc="upper right")

    # ---- Panels 1..L: head-averaged attention per layer ---- #
    for layer_idx, attn in enumerate(attns):
        ax = axes[layer_idx + 1]
        # attn: (1, h, n, n) → average over heads → (n, n)
        attn_mat = attn[0].mean(dim=0).numpy()
        im = ax.imshow(attn_mat, aspect="auto", cmap="viridis")
        ax.set_xlabel("key position (where attention LOOKS)")
        ax.set_ylabel("query position (who is asking)")
        ax.set_title(f"layer {layer_idx} — head-averaged attention")
        # Mark the label key column for orientation
        ax.axvline(y_label, color="green", linestyle="--", alpha=0.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {save_path}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)

    n        = 20
    vocab_sz = 10
    V_MAX    = vocab_sz - 1

    # ---- Task 1: argmax ---- #
    print("=" * 60)
    print("Task 1: argmax (find position of UNIQUE max value)")
    print("=" * 60)
    print("training baseline (1000 steps)...")
    model_a = train_baseline(gen_argmax_batch, vocab_size=vocab_sz,
                             n=n, device=device)

    torch.manual_seed(42)  # different seed so the sample isn't training data
    x, y = gen_argmax_batch(1, n, vocab_sz, device=device)
    logits, attns = capture_attention(model_a, x)
    pred = logits.argmax(dim=-1)
    print(f"  input  = {x[0].tolist()}")
    print(f"  label  = {y[0].item()}    pred = {pred[0].item()}    "
          f"{'OK' if y[0] == pred[0] else 'XX'}")
    plot_attention(
        x_sample=x[0],
        y_label=y[0].item(),
        pred_label=pred[0].item(),
        attns=attns,
        title="Argmax task — every query attends to the (unique) max token",
        save_path=out_dir / "attn_argmax.png",
        max_token=V_MAX,
    )

    # ---- Task 2: second-max ---- #
    print()
    print("=" * 60)
    print("Task 2: second-max (find position of SECOND occurrence of max)")
    print("=" * 60)
    print("training baseline (1000 steps)...")
    model_s = train_baseline(gen_second_max_batch, vocab_size=vocab_sz,
                             n=n, device=device)

    torch.manual_seed(7)
    x, y = gen_second_max_batch(1, n, vocab_sz, device=device)
    logits, attns = capture_attention(model_s, x)
    pred = logits.argmax(dim=-1)
    max_positions = (x[0] == V_MAX).nonzero(as_tuple=False).squeeze(-1).tolist()
    print(f"  input          = {x[0].tolist()}")
    print(f"  max positions  = {max_positions}")
    print(f"  label (second) = {y[0].item()}    pred = {pred[0].item()}    "
          f"{'OK' if y[0] == pred[0] else 'XX'}")
    plot_attention(
        x_sample=x[0],
        y_label=y[0].item(),
        pred_label=pred[0].item(),
        attns=attns,
        title="Second-max task — model must distinguish two identical tokens by POSITION",
        save_path=out_dir / "attn_second_max.png",
        max_token=V_MAX,
    )

    print()
    print(f"figures saved under: {out_dir}")


if __name__ == "__main__":
    main()
