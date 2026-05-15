"""
Tests for V2 — Vision Transformer model.

Five tests:
    1. Forward output shape:  (B, C, H, W) → (B, num_classes)
    2. cls_token shape  is (1, 1, d_model)
    3. pos_embed shape  is (1, N+1, d_model)
    4. pos_embed is learnable (Parameter, not Buffer) and requires_grad
    5. EncoderBlock stack length equals `depth`
"""

import sys
from pathlib import Path

# Make src/ importable.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn as nn

from vit import ViT


# ---------------------------------------------------------------------------- #
# Test 1 — Forward output shape                                                #
# ---------------------------------------------------------------------------- #
def test_forward_output_shape():
    """End-to-end shape sanity: image batch → class logits."""
    # ------------------------------------------------------------------ #
    # TODO 1 — Build model, push a random batch through, assert shape.    #
    #                                                                     #
    #   B, C, H = 2, 3, 32                                                 #
    #   model = ViT(img_size=H, patch_size=4, in_chans=C,                  #
    #               num_classes=10, d_model=64, depth=2, num_heads=2)      #
    #   model.eval()                                                        #
    #   x = torch.randn(B, C, H, H)                                         #
    #   logits = model(x)                                                   #
    #   assert logits.shape == (B, 10), f"got {tuple(logits.shape)}"        #
    # ------------------------------------------------------------------ #
    B, C, H = 2, 3, 32
    model = ViT(img_size=H, patch_size=4, in_chans=C,
                num_classes=10, d_model=64, depth=2, num_heads=2)
    model.eval()
    x = torch.randn(B, C, H, H)
    logits = model(x)
    assert logits.shape == (B, 10), f"got {tuple(logits.shape)}"


# ---------------------------------------------------------------------------- #
# Test 2 — CLS token shape                                                     #
# ---------------------------------------------------------------------------- #
def test_cls_token_shape():
    """cls_token must be a Parameter of shape (1, 1, d_model)."""
    # ------------------------------------------------------------------ #
    # TODO 2 — Inspect model.cls_token.                                    #
    #                                                                     #
    #   d = 64                                                              #
    #   model = ViT(img_size=32, patch_size=4, in_chans=3,                  #
    #               num_classes=10, d_model=d, depth=2, num_heads=2)        #
    #   assert isinstance(model.cls_token, nn.Parameter)                    #
    #   assert model.cls_token.shape == (1, 1, d), (                        #
    #       f"got {tuple(model.cls_token.shape)}"                           #
    #   )                                                                   #
    # ------------------------------------------------------------------ #
    d = 64
    model = ViT(img_size=32, patch_size=4, in_chans=3,
                num_classes=10, d_model=d, depth=2, num_heads=2)
    assert isinstance(model.cls_token, nn.Parameter)
    assert model.cls_token.shape == (1, 1, d), (
        f"got {tuple(model.cls_token.shape)}"
    )


# ---------------------------------------------------------------------------- #
# Test 3 — pos_embed shape with the +1 slot for CLS                            #
# ---------------------------------------------------------------------------- #
def test_pos_embed_shape_has_cls_slot():
    """pos_embed must have N+1 rows (one extra for the CLS slot)."""
    # ------------------------------------------------------------------ #
    # TODO 3 — Compute N and assert pos_embed.shape == (1, N+1, d).       #
    #                                                                     #
    #   H, P, d = 32, 4, 64                                                #
    #   N = (H // P) ** 2     # = 64                                       #
    #   model = ViT(img_size=H, patch_size=P, in_chans=3,                  #
    #               num_classes=10, d_model=d, depth=2, num_heads=2)       #
    #   assert model.pos_embed.shape == (1, N + 1, d), (                   #
    #       f"got {tuple(model.pos_embed.shape)}"                          #
    #   )                                                                  #
    # ------------------------------------------------------------------ #
    H, P, d = 32, 4, 64
    N = (H // P) ** 2
    model = ViT(img_size=H, patch_size=P, in_chans=3,
                num_classes=10, d_model=d, depth=2, num_heads=2)
    assert model.pos_embed.shape == (1, N + 1, d), (
        f"got {tuple(model.pos_embed.shape)}"
    )


# ---------------------------------------------------------------------------- #
# Test 4 — pos_embed is learnable                                              #
# ---------------------------------------------------------------------------- #
def test_pos_embed_is_learnable():
    """pos_embed must be a Parameter (nn.Parameter), not a Buffer.

    This is the silent-bug check: if you accidentally register_buffer it,
    pos_embed stays at its init value forever and the model behaves as if
    it has no PE — see the lecture on the 13% accuracy cliff.
    """
    # ------------------------------------------------------------------ #
    # TODO 4 — Assert .requires_grad and the named_parameters membership. #
    #                                                                     #
    #   model = ViT(img_size=32, patch_size=4, in_chans=3,                 #
    #               num_classes=10, d_model=64, depth=2, num_heads=2)      #
    #   assert isinstance(model.pos_embed, nn.Parameter), (                #
    #       "pos_embed must be nn.Parameter — silent bug if it isn't"      #
    #   )                                                                  #
    #   assert model.pos_embed.requires_grad                                #
    #   names = {n for n, _ in model.named_parameters()}                    #
    #   assert "pos_embed" in names                                         #
    # ------------------------------------------------------------------ #
    H, P, d = 32, 4, 64
    model = ViT(img_size=H, patch_size=P, in_chans=3,
                num_classes=10, d_model=d, depth=2, num_heads=2)
    assert isinstance(model.pos_embed, nn.Parameter), (
        "pos_embed must be nn.Parameter — silent bug if it isn't"
    )
    assert model.pos_embed.requires_grad
    names = {n for n, _ in model.named_parameters()}
    assert "pos_embed" in names


# ---------------------------------------------------------------------------- #
# Test 5 — Encoder depth                                                       #
# ---------------------------------------------------------------------------- #
def test_blocks_count_equals_depth():
    """self.blocks must contain exactly `depth` EncoderBlocks."""
    # ------------------------------------------------------------------ #
    # TODO 5 — Build a model with depth=L and check len(model.blocks).    #
    #                                                                     #
    #   L = 4                                                              #
    #   model = ViT(img_size=32, patch_size=4, in_chans=3,                 #
    #               num_classes=10, d_model=64, depth=L, num_heads=2)      #
    #   assert len(model.blocks) == L, f"got {len(model.blocks)}"          #
    # ------------------------------------------------------------------ #
    H, P, d, L = 32, 4, 64, 4
    model = ViT(img_size=H, patch_size=P, in_chans=3,
                num_classes=10, d_model=d, depth=L, num_heads=2)
    assert len(model.blocks) == L, f"got {len(model.blocks)}"


if __name__ == "__main__":
    test_forward_output_shape()
    test_cls_token_shape()
    test_pos_embed_shape_has_cls_slot()
    test_pos_embed_is_learnable()
    test_blocks_count_equals_depth()
    print("All ViT tests passed!")
