"""
T4.2 — Synthetic dataset for the argmax-position task.

Task
----
Input  : a sequence of n integers from a small vocab.
Output : the position (0..n-1) of the (unique) maximum value.

Design
------
To avoid label ambiguity from ties:
  - sample base values uniformly from [0, vocab_size - 1)  (exclusive of max),
  - pick one random position per row,
  - set that position to vocab_size - 1  (globally unique max).
Label = the chosen position.

This keeps the labels deterministic, the max-position uniform over
positions, and forces the model to look at token content (not just
position) to find the answer.

Run
---
    python data.py
"""

import torch


def gen_argmax_batch(B, n, vocab_size, device="cpu", generator=None):
    """
    Generate one batch for the argmax-position task.

    Parameters
    ----------
    B : int        batch size
    n : int        sequence length
    vocab_size : int   vocabulary size (must be >= 2)
    device : str | torch.device
    generator : torch.Generator | None
        Pass a seeded generator for reproducibility.

    Returns
    -------
    x : LongTensor (B, n)   values in [0, vocab_size)
    y : LongTensor (B,)     true argmax position per row
    """
    assert vocab_size >= 2, "vocab_size must be at least 2 (need at least one non-max value)"

    # Sample base values strictly less than the max token.
    x = torch.randint(0, vocab_size - 1, (B, n), device=device, generator=generator)

    # Random target position per row.
    y = torch.randint(0, n, (B,), device=device, generator=generator)

    # Place the unique max at the chosen position via advanced indexing.
    # x[torch.arange(B), y] selects ONE element per row — same idiom as
    # gather/scatter but more readable for "one per row".
    x[torch.arange(B, device=device), y] = vocab_size - 1
    return x, y


def gen_second_max_batch(B, n, vocab_size, device="cpu", generator=None):
    """
    Generate one batch for the "find SECOND occurrence of the max" task.

    Strategy
    --------
    - Sample base values uniformly from [0, vocab_size - 1)  (excludes max).
    - Pick TWO distinct positions p1 < p2 per row.
    - Set x[i, p1] = x[i, p2] = vocab_size - 1  (two identical max tokens).
    - Label y[i] = p2  (the SECOND occurrence).

    Why this task strictly requires positional encoding
    ---------------------------------------------------
    The two max positions hold the *same token id*.  Without PE, their
    embeddings are identical, their attention outputs are identical, and
    score_head produces identical scores at both — argmax then breaks
    the tie deterministically (PyTorch picks the lower index), so the
    model returns p1, but the answer is p2.  Without PE the model
    *cannot* solve this task above ~50% (random tie-break).

    Parameters / Returns: same convention as gen_argmax_batch.
    """
    assert n >= 2,           "need at least 2 positions for two distinct max"
    assert vocab_size >= 2,  "vocab_size must be >= 2"

    # Sample base values in [0, vocab_size - 1) — i.e., < the max token.
    x = torch.randint(0, vocab_size - 1, (B, n), device=device, generator=generator)

    # Pick two distinct random positions per row.
    # Trick: torch.randperm is single-row only.  For batched distinct
    # picks, generate random scores then argsort — argsort gives a
    # random permutation row by row.
    rand_scores = torch.rand(B, n, device=device, generator=generator)
    perm = rand_scores.argsort(dim=-1)          # (B, n)  random permutation per row
    pos = perm[:, :2]                            # (B, 2)  take any two
    pos, _ = pos.sort(dim=-1)                    # ensure p1 < p2 per row
    p1, p2 = pos[:, 0], pos[:, 1]                # each (B,)

    # Plant the (identical) max token at both positions.
    batch_idx = torch.arange(B, device=device)
    x[batch_idx, p1] = vocab_size - 1
    x[batch_idx, p2] = vocab_size - 1

    return x, p2


def main():
    torch.manual_seed(0)
    B, n, V = 4, 8, 10
    x, y = gen_argmax_batch(B, n, V)

    print(f"x.shape = {tuple(x.shape)}")
    print(f"y.shape = {tuple(y.shape)}")
    print(f"x =\n{x}")
    print(f"y = {y.tolist()}")

    # Sanity: every row's argmax should equal y
    pred = x.argmax(dim=-1)
    print(f"\nargmax(x)         = {pred.tolist()}")
    print(f"y (label)         = {y.tolist()}")
    print(f"all match?        = {torch.equal(pred, y)}")

    # Sanity: every row's max value equals vocab_size - 1
    max_vals = x.max(dim=-1).values
    print(f"max value per row = {max_vals.tolist()}   (expected all {V - 1})")

    # Distribution check on a bigger batch — y should roughly cover [0, n)
    _, y_big = gen_argmax_batch(B=10000, n=n, vocab_size=V)
    counts = torch.bincount(y_big, minlength=n)
    print(f"\ny distribution over 10000 samples:")
    print(f"  position counts = {counts.tolist()}    (each ~{10000 // n}, should be roughly even)")

    # ---- Second-max task sanity ------------------------------------------- #
    print()
    print("=" * 60)
    print("gen_second_max_batch sanity")
    print("=" * 60)
    x2, y2 = gen_second_max_batch(B, n, V)
    print(f"x2 =\n{x2}")
    print(f"y2 (= second-occurrence positions) = {y2.tolist()}")

    # Per row, count occurrences of the max token — should be exactly 2.
    max_token = V - 1
    counts_per_row = (x2 == max_token).sum(dim=-1)
    print(f"max-token count per row = {counts_per_row.tolist()}   (expected all 2)")

    # For each row, the SECOND occurrence index should equal y2.
    # nonzero(as_tuple=False) returns (k, 2) of (row, col) of all True positions.
    # For each row we want the column index of the SECOND True.
    occ = (x2 == max_token).nonzero(as_tuple=False)   # (2*B, 2)
    # rows come out sorted; the second occurrence per row is every odd-indexed entry.
    second_occ = occ[1::2, 1]                          # (B,)
    print(f"second occurrence per row = {second_occ.tolist()}")
    print(f"matches y2?               = {torch.equal(second_occ, y2)}")


if __name__ == "__main__":
    main()
