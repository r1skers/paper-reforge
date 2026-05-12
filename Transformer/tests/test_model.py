import torch
import sys
from pathlib import Path
# Make `src/` importable whether we run via pytest or directly.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model import ArgmaxPositionModel
from data import gen_argmax_batch

def test_output_shape():
    """The model forward output must have shape (B, n), same as input."""
    B, n = 2, 5
    vocab_size = 10
    model = ArgmaxPositionModel(vocab_size=vocab_size, n_max=32, d_model=16, num_heads=4, num_layers=2)
    x = torch.randint(0, vocab_size, (B, n))
    assert model(x).shape == (B, n)


def test_no_pe_runs():
    """The model should run without PE if use_pe=False."""
    B, n = 2, 5
    vocab_size = 10
    model = ArgmaxPositionModel(vocab_size=vocab_size, n_max=32, d_model=16, num_heads=4, num_layers=2, use_pe=False)
    x = torch.randint(0, vocab_size, (B, n))
    assert model(x).shape == (B, n)


def test_gradient_to_embedding():
    """
    Gradient must propagate from loss back to token_emb.weight.

    token_emb is the DEEPEST learnable layer in the forward chain.
    If gradient reaches here, the whole forward path is differentiable.
    One assertion replaces N per-layer checks — high signal, low effort.
    """
    B, n, vocab_size = 2, 5, 10
    model = ArgmaxPositionModel(vocab_size=vocab_size, n_max=32,
                                d_model=16, num_heads=4, num_layers=2,
                                dropout=0.0)
    x = torch.randint(0, vocab_size, (B, n))
    loss = model(x).sum()
    loss.backward()

    grad = model.token_emb.weight.grad
    assert grad is not None, "token_emb.weight.grad is None — autograd chain broken"
    assert grad.abs().sum().item() > 0, "token_emb.weight.grad is all zero — gradient died before reaching here"

def test_data_shape():
    """Check that the data generation function produces the expected shapes."""
    B, n, vocab_size = 4, 8, 10
    x, y = gen_argmax_batch(B, n, vocab_size)
    assert x.shape == (B, n), f"Expected x shape {(B, n)}, got {x.shape}"
    assert y.shape == (B,), f"Expected y shape {(B,)}, got {y.shape}"

def test_argmax_matches_label():
    """Check that the argmax of x matches the label y."""
    B, n, vocab_size = 4, 8, 10
    x, y = gen_argmax_batch(B, n, vocab_size)
    pred = x.argmax(dim=-1)
    assert torch.equal(pred, y), f"Expected argmax(x) to equal y, but got {pred.tolist()} vs {y.tolist()}"

def test_max_value_is_vocab_minus_one():
    """Check that the max value in each row of x is vocab_size - 1."""
    B, n, vocab_size = 4, 8, 10
    x, _ = gen_argmax_batch(B, n, vocab_size)
    max_vals = x.max(dim=-1).values
    expected_max = vocab_size - 1
    assert torch.all(max_vals == expected_max), f"Expected max value per row to be {expected_max}, but got {max_vals.tolist()}"

if __name__ == "__main__":
    test_output_shape()
    test_no_pe_runs()
    test_gradient_to_embedding()
    test_data_shape()
    test_argmax_matches_label()
    test_max_value_is_vocab_minus_one()
    print("All tests passed.")