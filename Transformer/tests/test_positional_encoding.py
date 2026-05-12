import sys
from pathlib import Path

# Make `src/` importable whether we run via pytest or directly.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch

from positional_encoding import SinusoidalPositionalEncoding



def test_pe_buffer_shape():
    """The PE table must have shape (max_len, d_model)."""
    d_model = 16
    max_len = 50
    pe_module = SinusoidalPositionalEncoding(d_model, max_len)
    assert pe_module.pe.shape == (max_len, d_model), f"Expected shape ({max_len}, {d_model}), got {tuple(pe_module.pe.shape)}"

def test_pe_has_no_trainable_params():
    """The PE table must be a buffer, not a parameter — no trainable params."""
    d_model = 16
    max_len = 50
    pe_module = SinusoidalPositionalEncoding(d_model, max_len)
    num_params = sum(p.numel() for p in pe_module.parameters())
    assert num_params == 0, f"Expected 0 trainable params, got {num_params}"

def test_forward_output_shape():
    """The forward output must have shape (B, n, d_model), same as input."""
    d_model = 16
    max_len = 50
    pe_module = SinusoidalPositionalEncoding(d_model, max_len)

    B, n = 4, 30
    x = torch.zeros(B, n, d_model)
    out = pe_module(x)
    assert out.shape == (B, n, d_model), f"Expected shape ({B}, {n}, {d_model}), got {tuple(out.shape)}"

def test_pe_at_position_zero():
    """At position 0, even dims should be 0 (sin(0)=0) and odd dims should be 1 (cos(0)=1)."""
    d_model = 16
    max_len = 50
    pe_module = SinusoidalPositionalEncoding(d_model, max_len)

    pe = pe_module.pe
    even_dims = pe[0, 0::2]  # even dims at position 0
    odd_dims = pe[0, 1::2]  # odd dims at position 0

    assert torch.allclose(even_dims, torch.zeros_like(even_dims)), f"Expected even dims to be 0, got {even_dims}"
    assert torch.allclose(odd_dims, torch.ones_like(odd_dims)), f"Expected odd dims to be 1, got {odd_dims}"

def test_forward_zero_input_returns_pe():
    """If the input x is all zeros, the output should equal the corresponding slice of the PE table."""
    d_model = 16
    max_len = 50
    pe_module = SinusoidalPositionalEncoding(d_model, max_len)

    B, n = 4, 30
    x = torch.zeros(B, n, d_model)
    out = pe_module(x)

    expected_pe_slice = pe_module.pe[:n]  # (n, d_model)
    assert torch.allclose(out, expected_pe_slice.unsqueeze(0).expand(B, -1, -1))

if __name__ == "__main__":
    # Run the tests when this script is executed directly.
    test_pe_buffer_shape()
    test_pe_has_no_trainable_params()
    test_forward_output_shape()
    test_pe_at_position_zero()
    test_forward_zero_input_returns_pe()
    print("All tests passed!")
    