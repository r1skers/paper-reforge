"""
V5 — Visualize ViT attention maps.

Load the best MNIST checkpoint, run a small batch of test images through it,
and for each image extract the [CLS] token's attention to all patches in the
last EncoderBlock. Reshape the patch attention back into a 2D grid, upsample
to image resolution, and plot side-by-side:

    original digit   |   attention heatmap   |   overlay

Method:
    - Register a forward hook on each EncoderBlock's MultiHeadSelfAttention
      to capture (output, attn_weights) — we keep the weights.
    - attn[:, :, 0, 1:] = CLS query attending to patch keys, shape (B, H, N).
    - Average over heads (simple summary; could also show per-head).
    - Reshape (N,) → (√N, √N), upsample to (H, W) via F.interpolate(bilinear).
    - Overlay with alpha.

Run
---
    python visualize_attention.py
    python visualize_attention.py --ckpt ../outputs/default/best.pt --n_images 8
"""

import argparse
import sys
from pathlib import Path

# Make src/ importable.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import math

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from vit import ViT
from data import get_mnist_loaders


# ---------------------------------------------------------------------------- #
# Helpers                                                                       #
# ---------------------------------------------------------------------------- #


def load_model_from_ckpt(ckpt_path, device="cpu"):
    """Rebuild a ViT from the args saved in the checkpoint and load weights."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = ViT(
        img_size=a["img_size"], patch_size=a["patch_size"],
        in_chans=a["in_chans"], num_classes=a["num_classes"],
        d_model=a["d_model"], depth=a["depth"], num_heads=a["num_heads"],
        dropout=0.0,            # disable dropout for inference
        activation=a["activation"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    print(f"Loaded ckpt from {ckpt_path}")
    print(f"  saved at epoch {ckpt['epoch']}, test_acc={ckpt['test_acc']*100:.2f}%")
    return model, a


def register_attention_hooks(model):
    """
    Register forward hooks on each EncoderBlock's attn submodule.
    Returns a dict that will be filled with attention tensors on every forward.

        attn_maps[layer_idx] : tensor of shape (B, num_heads, N+1, N+1)

    The hook captures the SECOND element of the attn module's output tuple
    — that's the attention weights. The first element is the attn output
    feature that flows down the model.
    """
    attn_maps = {}

    # ------------------------------------------------------------------ #
    # TODO 1 — Register a forward hook on each block's .attn submodule.   #
    #                                                                     #
    # The trick is binding `i` properly via a default arg or closure       #
    # factory — otherwise all hooks would write to the same key.           #
    #                                                                     #
    #   def make_hook(layer_idx):                                          #
    #       def hook(module, inputs, outputs):                             #
    #           # MultiHeadSelfAttention returns (out, attn_weights)       #
    #           attn_maps[layer_idx] = outputs[1].detach().cpu()           #
    #       return hook                                                    #
    #                                                                     #
    #   for i, blk in enumerate(model.blocks):                             #
    #       blk.attn.register_forward_hook(make_hook(i))                   #
    # ------------------------------------------------------------------ #
    def make_hook(layer_idx):
        def hook(module, inputs, outputs):
            # MultiHeadSelfAttention returns (out, attn_weights)
            attn_maps[layer_idx] = outputs[1].detach().cpu()
        return hook
    for i, blk in enumerate(model.blocks):
        blk.attn.register_forward_hook(make_hook(i))

    return attn_maps


# ---------------------------------------------------------------------------- #
# CLS → patches attention extraction                                            #
# ---------------------------------------------------------------------------- #


def cls_to_patch_attention(attn_layer, head_reduce="mean"):
    """
    Pull out the CLS-to-patch attention from one layer.

    Parameters
    ----------
    attn_layer  : tensor of shape (B, H, N+1, N+1)
                  from one captured EncoderBlock attention
    head_reduce : 'mean'  → average over heads (simple summary)
                  'max'   → max over heads (sharpest map)

    Returns
    -------
    attn : tensor of shape (B, N) — for each image, the attention weight that
           the CLS query placed on each of the N patch keys (excludes the
           CLS-to-CLS weight at position 0)
    """
    # ------------------------------------------------------------------ #
    # TODO 2 — Slice CLS query, slice patch keys, reduce heads.           #
    #                                                                     #
    #   # CLS is at sequence position 0.                                   #
    #   # Patches are at positions 1..N.                                   #
    #   #                                                                 #
    #   # attn_layer shape: (B, H, N+1, N+1)                               #
    #   # [b, h, i, j] = attention weight from query i to key j            #
    #                                                                     #
    #   cls_attn = attn_layer[:, :, 0, 1:]    # (B, H, N)                  #
    #                                                                     #
    #   if head_reduce == "mean":                                          #
    #       return cls_attn.mean(dim=1)        # (B, N)                    #
    #   elif head_reduce == "max":                                         #
    #       return cls_attn.max(dim=1).values  # (B, N)                    #
    #   else:                                                              #
    #       raise ValueError(f"unknown head_reduce: {head_reduce}")        #
    # ------------------------------------------------------------------ #
    cls_attn = attn_layer[:, :, 0, 1:]    # (B, H, N)
    if head_reduce == "mean":
        return cls_attn.mean(dim=1)        # (B, N)
    elif head_reduce == "max":
        return cls_attn.max(dim=1).values  # (B, N)
    else:
        raise ValueError(f"unknown head_reduce: {head_reduce}")


# ---------------------------------------------------------------------------- #
# Reshape sequence attention → 2D grid → upsample to image                     #
# ---------------------------------------------------------------------------- #


def attn_to_image_grid(attn_1d, img_size):
    """
    Take per-patch attention as a 1D sequence and turn it into a 2D heatmap
    upsampled to image resolution.

    Parameters
    ----------
    attn_1d  : tensor of shape (B, N) — attention weight per patch
    img_size : int — output spatial size (H = W = img_size)

    Returns
    -------
    heatmap : tensor of shape (B, img_size, img_size), values in [0, 1]
              after per-image min-max normalization (so each image's map
              fills its own dynamic range)
    """
    # ------------------------------------------------------------------ #
    # TODO 3 — Reshape (B, N) -> (B, 1, √N, √N), interpolate to image,     #
    # min-max normalize per image so each heatmap fills [0,1].            #
    #                                                                     #
    #   B, N = attn_1d.shape                                              #
    #   side = int(round(N ** 0.5))                                        #
    #   assert side * side == N, f"N={N} not a perfect square"             #
    #                                                                     #
    #   grid = attn_1d.reshape(B, 1, side, side)                           #
    #   grid = F.interpolate(grid, size=(img_size, img_size),              #
    #                        mode="bilinear", align_corners=False)         #
    #   grid = grid.squeeze(1)                            # (B, H, W)      #
    #                                                                     #
    #   # Per-image min-max normalize so faint maps still visible.        #
    #   B = grid.shape[0]                                                  #
    #   flat = grid.reshape(B, -1)                                         #
    #   mn   = flat.min(dim=1, keepdim=True).values                        #
    #   mx   = flat.max(dim=1, keepdim=True).values                        #
    #   grid = (grid - mn.view(B,1,1)) / (mx - mn + 1e-8).view(B,1,1)      #
    #   return grid                                                        #
    # ------------------------------------------------------------------ #
    B, N = attn_1d.shape
    side = int(round(N ** 0.5))
    assert side * side == N, f"N={N} not a perfect square"  

    grid = attn_1d.reshape(B, 1, side, side)
    grid = F.interpolate(grid, size=(img_size, img_size),
                         mode="bilinear", align_corners=False)
    grid = grid.squeeze(1)                            # (B, H, W)   
    # Per-image min-max normalize so faint maps still visible.
    B = grid.shape[0]
    flat = grid.reshape(B, -1)
    mn   = flat.min(dim=1, keepdim=True).values
    mx   = flat.max(dim=1, keepdim=True).values
    grid = (grid - mn.view(B,1,1)) / (mx - mn + 1e-8).view(B,1,1)
    return grid



# ---------------------------------------------------------------------------- #
# Attention rollout (Abnar & Zuidema 2020, used in ViT paper Fig 6)             #
# ---------------------------------------------------------------------------- #


def attention_rollout(attn_maps, head_reduce="mean"):
    """
    Compose attention across all layers into a single 'rollout' matrix that
    summarizes how information flows from input tokens to output tokens.

    Idea
    ----
    Each EncoderBlock applies (roughly):
        z_l = A_l @ z_{l-1}   plus a residual identity path
    To account for the residual, replace each A_l by
        Ã_l = 0.5 * A_l + 0.5 * I              (eqn from the paper)
    Then the composed transformation from z_0 to z_L is
        rollout = Ã_L · Ã_{L-1} · ... · Ã_1
    The CLS row of `rollout` tells us how the final CLS state was assembled
    from all the input patches.

    Parameters
    ----------
    attn_maps   : dict[int -> tensor]
                  attn_maps[l] has shape (B, num_heads, N+1, N+1)
                  collected by `register_attention_hooks`.
    head_reduce : 'mean' or 'max' — how to collapse the head dim before
                  multiplying. 'mean' is the original paper recipe.

    Returns
    -------
    cls_rollout : tensor of shape (B, N) — CLS query's *cumulative*
                  attention to each of the N patch keys after L layers.
                  Excludes CLS-to-CLS at position 0.
    """
    # ------------------------------------------------------------------ #
    # TODO 4 — Implement attention rollout.                               #
    #                                                                     #
    # Step A — collapse heads in each layer.                              #
    #   Each attn_maps[l] is (B, H, N+1, N+1). After head reduce it's     #
    #   (B, N+1, N+1).                                                    #
    #                                                                     #
    #   def reduce_heads(a):                                              #
    #       if head_reduce == "mean":                                     #
    #           return a.mean(dim=1)                                      #
    #       elif head_reduce == "max":                                    #
    #           return a.max(dim=1).values                                #
    #       else:                                                         #
    #           raise ValueError(head_reduce)                             #
    #                                                                     #
    # Step B — add identity for residual, then matmul-chain.              #
    #   L = len(attn_maps)                                                #
    #   first = reduce_heads(attn_maps[0])     # (B, N+1, N+1)            #
    #   B_, n, _ = first.shape                                            #
    #   I = torch.eye(n).expand(B_, n, n)                                 #
    #                                                                     #
    #   rollout = 0.5 * first + 0.5 * I                                   #
    #   for l in range(1, L):                                             #
    #       A_l = 0.5 * reduce_heads(attn_maps[l]) + 0.5 * I              #
    #       rollout = A_l @ rollout                # left-multiply        #
    #                                                                     #
    # Step C — slice CLS row, drop CLS→CLS column.                        #
    #   cls_rollout = rollout[:, 0, 1:]            # (B, N)                #
    #   return cls_rollout                                                 #
    #                                                                     #
    # Why left-multiply: if z_l = A_l z_{l-1}, then                       #
    #   z_L = A_L A_{L-1} ... A_1 z_0                                     #
    # so each new A_l should be applied on the LEFT of the running product.#
    # ------------------------------------------------------------------ #
    def reduce_heads(a):
        if head_reduce == "mean":
            return a.mean(dim=1)
        elif head_reduce == "max":
            return a.max(dim=1).values
        else:
            raise ValueError(head_reduce)
    L = len(attn_maps)
    first = reduce_heads(attn_maps[0])     # (B, N+1, N+1)
    B_, n, _ = first.shape
    I = torch.eye(n).expand(B_, n, n)

    rollout = 0.5 * first + 0.5 * I
    for l in range(1, L):
        A_l = 0.5 * reduce_heads(attn_maps[l]) + 0.5 * I
        rollout = A_l @ rollout                # left-multiply

    cls_rollout = rollout[:, 0, 1:]            # (B, N)
    return cls_rollout



# ---------------------------------------------------------------------------- #
# Plot                                                                          #
# ---------------------------------------------------------------------------- #


def plot_attention_grid(images, heatmaps, labels, preds, save_path):
    """
    Save a figure with 3 columns per image: original, heatmap, overlay.

    Parameters
    ----------
    images   : tensor (B, 1, H, W)  — denormalized to [0, 1] for display
    heatmaps : tensor (B, H, W)     — already in [0, 1]
    labels   : list of int          — ground truth
    preds    : list of int          — model predictions
    save_path: path-like
    """
    B = images.shape[0]
    fig, axes = plt.subplots(B, 3, figsize=(6, 2 * B))
    if B == 1:
        axes = axes[None, :]   # make indexing uniform

    for b in range(B):
        img = images[b, 0].cpu().numpy()
        hm  = heatmaps[b].cpu().numpy()

        axes[b, 0].imshow(img, cmap="gray")
        axes[b, 0].set_title(f"gt={labels[b]}  pred={preds[b]}", fontsize=9)
        axes[b, 0].axis("off")

        axes[b, 1].imshow(hm, cmap="jet")
        axes[b, 1].set_title("CLS→patches attn", fontsize=9)
        axes[b, 1].axis("off")

        axes[b, 2].imshow(img, cmap="gray")
        axes[b, 2].imshow(hm, cmap="jet", alpha=0.5)
        axes[b, 2].set_title("overlay", fontsize=9)
        axes[b, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path}")


# ---------------------------------------------------------------------------- #
# Main                                                                          #
# ---------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       type=str, default="../outputs/default/best.pt")
    p.add_argument("--n_images",   type=int, default=8)
    p.add_argument("--mode",       type=str, default="single_layer",
                   choices=["single_layer", "rollout"],
                   help="single_layer: attention from one chosen layer; "
                        "rollout: cumulative attention across all layers "
                        "(Abnar & Zuidema)")
    p.add_argument("--layer",      type=int, default=-1,
                   help="which layer's attention to visualize when mode=single_layer "
                        "(-1 = last)")
    p.add_argument("--head_reduce", type=str, default="mean",
                   choices=["mean", "max"])
    p.add_argument("--out",        type=str, default="../outputs/default/attention.png")
    p.add_argument("--device",     type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ---- Load model + hooks ---------------------------------------------- #
    ckpt_path = Path(__file__).resolve().parent / args.ckpt
    model, model_args = load_model_from_ckpt(ckpt_path, device=device)
    attn_maps = register_attention_hooks(model)

    # ---- Pick a small batch of test images ------------------------------- #
    _, test_loader = get_mnist_loaders(batch_size=args.n_images)
    x, y = next(iter(test_loader))
    x = x.to(device)

    # ---- Forward → fills attn_maps -------------------------------------- #
    with torch.no_grad():
        logits = model(x)
    preds = logits.argmax(dim=1).cpu().tolist()
    labels = y.cpu().tolist()

    # Sanity print
    L = len(attn_maps)
    print(f"Captured attention from {L} layers")
    for i in sorted(attn_maps.keys()):
        print(f"  layer {i}: shape {tuple(attn_maps[i].shape)}")

    # ---- Pull CLS→patch attention based on mode ------------------------- #
    if args.mode == "single_layer":
        layer_idx = args.layer if args.layer >= 0 else L - 1
        print(f"Mode: single_layer, layer {layer_idx} ({args.head_reduce} over heads)")
        cls_attn = cls_to_patch_attention(
            attn_maps[layer_idx], head_reduce=args.head_reduce)
    elif args.mode == "rollout":
        print(f"Mode: rollout across {L} layers ({args.head_reduce} over heads)")
        cls_attn = attention_rollout(attn_maps, head_reduce=args.head_reduce)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    heatmaps = attn_to_image_grid(cls_attn, img_size=model_args["img_size"])

    # ---- Denormalize images for display --------------------------------- #
    # MNIST was normalized with mean=0.1307, std=0.3081
    mean, std = 0.1307, 0.3081
    images_disp = (x.cpu() * std + mean).clamp(0, 1)

    out_path = Path(__file__).resolve().parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_attention_grid(images_disp, heatmaps, labels, preds, out_path)


if __name__ == "__main__":
    main()
