import sys
from pathlib import Path
import torch
import numpy as np

# Make `src/` importable whether we run via pytest or directly.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from transformer_block import PositionWiseFFN
from transformer_block import EncoderBlock


def test_ffn_output_shape():
    """The FFN forward output must have shape (B, n, d_model), same as input."""
    B, n, d_model = 2, 5, 16
    ffn = PositionWiseFFN(d_model, dropout=0.1)
    assert ffn(torch.randn(B, n, d_model)).shape == (B, n, d_model)

def test_ffn_default_d_ff_is_4x():
    """The default d_ff should be 4 * d_model."""
    assert PositionWiseFFN(d_model=16).d_ff == 64, f"Expected default d_ff to be 64, got {PositionWiseFFN(d_model=16).d_ff}"

def test_ffn_gradient_flow():
    """Check that gradients flow through the FFN without error."""
    B, n, d_model = 2, 5, 16
    ffn = PositionWiseFFN(d_model, dropout=0.1)
    x = torch.randn(B, n, d_model)
    loss = ffn(x).sum()
    loss.backward()  
    for name, param in ffn.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert torch.any(param.grad != 0), f"Parameter {name} has zero gradient"

def test_ffn_dropout_train_vs_eval():
    B, n, d_model = 2, 5, 16
    ffn = PositionWiseFFN(d_model, dropout=0.5)
    x = torch.randn(B, n, d_model)

    ffn.eval()                                  # eval 模式 = dropout off
    y1 = ffn(x); y2 = ffn(x)
    assert torch.allclose(y1, y2)               # 必须相同 (确定性)

    ffn.train()                                 # train 模式 = dropout on
    y3 = ffn(x); y4 = ffn(x)
    assert not torch.allclose(y3, y4)  

def test_block_output_shape():
    """The TransformerBlock forward output must have shape (B, n, d_model), same as input."""
    B, n, d_model = 2, 5, 16
    block = EncoderBlock(d_model, num_heads=4, dropout=0.1)
    assert block(torch.randn(B, n, d_model)).shape == (B, n, d_model)

def test_block_with_mask():
    """Check that the block can process a mask without error."""
    B, n, d_model = 2, 5, 16
    block = EncoderBlock(d_model, num_heads=4, dropout=0.1)
    x = torch.randn(B, n, d_model)
    mask = torch.ones(B, n).bool()  
    output = block(x, mask=mask)
    assert not torch.isnan(output).any(), "Output contains NaNs when using a mask"

def test_block_gradient_flow():
    """Check that gradients flow through the TransformerBlock without error."""
    B, n, d_model = 2, 5, 16
    block = EncoderBlock(d_model, num_heads=4, dropout=0.1)
    x = torch.randn(B, n, d_model)
    loss = block(x).sum()
    loss.backward()  
    for name, param in block.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert torch.any(param.grad != 0), f"Parameter {name} has zero gradient"

def test_block_identity_when_sublayers_zero():
    """Pre-norm: zeroing sublayers should make block(x) == x (identity highway)."""
    d_model = 16
    block = EncoderBlock(d_model=d_model, num_heads=2, dropout=0.0)
    block.eval()

    with torch.no_grad():
        for name, p in block.named_parameters():
            if name.startswith("attn.") or name.startswith("ffn."):
                p.zero_()

    x = torch.randn(2, 5, d_model)
    out = block(x)
    assert torch.allclose(out, x, atol=1e-6), \
        f"identity broke: max diff = {(out - x).abs().max().item()}"

if __name__ == "__main__":
    test_ffn_output_shape()
    test_ffn_default_d_ff_is_4x()
    test_ffn_gradient_flow()
    test_ffn_dropout_train_vs_eval()
    test_block_output_shape()
    test_block_with_mask()
    test_block_gradient_flow()
    test_block_identity_when_sublayers_zero()
    print("All tests passed!")
   