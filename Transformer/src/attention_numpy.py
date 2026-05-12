"""
T2.1 — Scaled dot-product self-attention in pure numpy.

Learning goals
--------------
1. Hand-implement softmax (numerically stable form).
2. Manually verify shapes at every step.
3. Empirically observe softmax saturation when scores are NOT divided by
   sqrt(d_k) — the empirical justification for the "scaled" part.

Run
---
    python attention_numpy.py
"""

import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None, scale=True):
    """
    Compute scaled dot-product attention (single head).

    Parameters
    ----------
    Q : np.ndarray
        Query matrix, shape (n, d_k).
    K : np.ndarray
        Key matrix, shape (n, d_k).
    V : np.ndarray
        Value matrix, shape (n, d_v).
    mask : np.ndarray of bool or None, optional
        Shape (n, n). True  = position allowed to attend.
                      False = position masked out (set to -inf before softmax).
    scale : bool, default True
        If True, divide scores by sqrt(d_k).  Toggle off for the
        micro-ablation in __main__.

    Returns
    -------
    output : np.ndarray
        Shape (n, d_v).
    attn : np.ndarray
        Shape (n, n).  Softmax-normalized attention weights, returned
        for debugging / visualization.
    """
    # ------------------------------------------------------------------ #
    # TODO 1                                                              #
    # Compute raw scores  S = Q @ K.T   ->  shape (n, n).                 #
    # If scale=True, divide by sqrt(d_k) where d_k = Q.shape[-1].         #
    # ------------------------------------------------------------------ #
    S  = Q @ K.T
    if scale:
        d_k = Q.shape[-1]
        S = S / np.sqrt(d_k)
    # ------------------------------------------------------------------ #
    # TODO 2                                                              #
    # If mask is not None, set the masked positions to -inf.              #
    # Hint:  np.where(mask, S, -np.inf)                                   #
    # ------------------------------------------------------------------ #
    if mask is not None:
        S = np.where(mask, S, -np.inf)
    # ------------------------------------------------------------------ #
    # TODO 3                                                              #
    # Row-wise softmax.  MUST be numerically stable:                      #
    #     for each row i, subtract S[i].max() before exp().               #
    #                                                                     #
    #     attn[i, j] =        exp(S[i, j] - max_j' S[i, j'])              #
    #                    ─────────────────────────────────                #
    #                    sum_j  exp(S[i, j ] - max_j' S[i, j'])           #
    #                                                                     #
    # Output shape: (n, n).                                               #
    # ------------------------------------------------------------------ #
    row_maxes = S.max(axis=-1, keepdims=True)  
    exp_S = np.exp(S - row_maxes)
    attn = exp_S / exp_S.sum(axis=-1, keepdims=True)

    # ------------------------------------------------------------------ #
    # TODO 4                                                              #
    # output = attn @ V   ->   shape (n, d_v).                            #
    # Return (output, attn).                                              #
    # ------------------------------------------------------------------ #
    output = attn @ V
    return output, attn


def multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads, mask=None):
    """
    T2.2 — Multi-head self-attention (numpy, no batch dim yet).

    Implements the "one big projection + reshape" pattern instead of h
    independent per-head projections.  The h per-head matrices W_Q^(i)
    of shape (d_model, d_k) are conceptually stacked column-wise into a
    single big W_Q of shape (d_model, d_model).  Reshaping the projected
    output along the last dim recovers the per-head slices.

    Parameters
    ----------
    X : np.ndarray
        Input, shape (n, d_model).
    W_Q, W_K, W_V : np.ndarray
        Each (d_model, d_model).
    W_O : np.ndarray
        Output projection, shape (d_model, d_model).
    num_heads : int
        Number of heads h.  Must divide d_model.
    mask : np.ndarray of bool or None, optional
        Shape (n, n).  Broadcasts across the head dim automatically.

    Returns
    -------
    output : np.ndarray
        Shape (n, d_model).
    attn : np.ndarray
        Shape (h, n, n) — per-head attention weights, kept for debug.
    """
    n, d_model = X.shape
    h = num_heads
    assert d_model % h == 0, "d_model must be divisible by num_heads"
    d_k = d_model // h

    # ------------------------------------------------------------------ #
    # TODO 1 — Project to Q, K, V using the big matrices.                 #
    #                                                                     #
    #   Q = X @ W_Q     ->  (n, d_model)                                  #
    #   K = X @ W_K     ->  (n, d_model)                                  #
    #   V = X @ W_V     ->  (n, d_model)                                  #
    # ------------------------------------------------------------------ #
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    # ------------------------------------------------------------------ #
    # TODO 2 — Reshape Q, K, V from (n, d_model) to (h, n, d_k).          #
    #                                                                     #
    # Two-step pattern:                                                   #
    #   .reshape(n, h, d_k)        ->  (n, h, d_k)                        #
    #   .transpose(1, 0, 2)        ->  (h, n, d_k)                        #
    #                                                                     #
    # Intuition: each token's d_model-vector is sliced into h chunks of   #
    # size d_k (one chunk per head), then we put the head axis first so   #
    # we can run h attentions in parallel via batched matmul.             #
    # ------------------------------------------------------------------ #
    Q = Q.reshape(n, h, d_k).transpose(1, 0, 2)
    K = K.reshape(n, h, d_k).transpose(1, 0, 2)
    V = V.reshape(n, h, d_k).transpose(1, 0, 2)
    # ------------------------------------------------------------------ #
    # TODO 3 — Batched scaled dot-product attention across h heads.       #
    #                                                                     #
    # Scores:                                                             #
    #   S = Q @ K.swapaxes(-1, -2) / sqrt(d_k)        ->  (h, n, n)       #
    #                                                                     #
    #   NOTE: use .swapaxes(-1, -2), NOT .T  — .T reverses ALL axes and   #
    #   would give (d_k, n, h), which is wrong.                           #
    #                                                                     #
    # Mask:                                                               #
    #   if mask is not None:                                              #
    #       S = np.where(mask, S, -np.inf)                                #
    #   (mask shape (n, n) broadcasts over the leading h dim for free.)   #
    #                                                                     #
    # Softmax along the last axis — your existing stable softmax already  #
    # works on (h, n, n) as long as you used axis=-1 with keepdims=True:  #
    #   row_maxes = S.max(axis=-1, keepdims=True)                         #
    #   exp_S = np.exp(S - row_maxes)                                     #
    #   attn = exp_S / exp_S.sum(axis=-1, keepdims=True)   -> (h, n, n)   #
    #                                                                     #
    # Apply to V:                                                         #
    #   head_out = attn @ V                              -> (h, n, d_k)   #
    # ------------------------------------------------------------------ #
    S = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)
    if mask is not None:
        S = np.where(mask, S, -np.inf)
    row_maxes = S.max(axis=-1, keepdims=True)
    exp_S = np.exp(S - row_maxes)
    attn = exp_S / exp_S.sum(axis=-1, keepdims=True)
    head_out = attn @ V
    # ------------------------------------------------------------------ #
    # TODO 4 — Recombine heads and apply output projection W_O.           #
    #                                                                     #
    #   head_out.transpose(1, 0, 2)    ->  (n, h, d_k)                    #
    #   .reshape(n, h * d_k)           ->  (n, d_model)                   #
    #   @ W_O                          ->  (n, d_model)                   #
    #                                                                     #
    # Return (output, attn).                                              #
    # ------------------------------------------------------------------ #
    output = head_out.transpose(1, 0, 2).reshape(n, h * d_k) @ W_O
    return output, attn
    raise NotImplementedError("Fill in TODOs 1 - 4")


# ---------------------------------------------------------------------------- #
# Driver: shape sanity + scaling ablation + masking smoke test                 #
# ---------------------------------------------------------------------------- #


def main():
    rng = np.random.default_rng(seed=0)

    # ---- Sanity run -------------------------------------------------------- #
    print("=" * 60)
    print("Sanity run: single forward pass, print shapes")
    print("=" * 60)

    n, d_k, d_v = 4, 4, 4
    Q = rng.standard_normal((n, d_k))
    K = rng.standard_normal((n, d_k))
    V = rng.standard_normal((n, d_v))
    print(f"Q.shape = {Q.shape}")
    print(f"K.shape = {K.shape}")
    print(f"V.shape = {V.shape}")

    out, attn = scaled_dot_product_attention(Q, K, V)
    print(f"out.shape  = {out.shape}   (expected ({n}, {d_v}))")
    print(f"attn.shape = {attn.shape}  (expected ({n}, {n}))")
    print(f"row sums of attn (should all be 1.0):")
    print(f"  {attn.sum(axis=-1)}")

    # ---- Micro-ablation: scaling vs no scaling ----------------------------- #
    print()
    print("=" * 60)
    print("Micro-ablation: softmax saturation when not dividing by sqrt(d_k)")
    print("=" * 60)
    print(f"{'d_k':>6} | {'with sqrt(d_k)':>16} | {'no sqrt(d_k)':>16}")
    print("-" * 50)
    for d_k_test in (4, 64, 512):
        Q = rng.standard_normal((n, d_k_test))
        K = rng.standard_normal((n, d_k_test))
        V = rng.standard_normal((n, d_v))

        _, attn_scaled  = scaled_dot_product_attention(Q, K, V, scale=True)
        _, attn_noscale = scaled_dot_product_attention(Q, K, V, scale=False)

        # "Sharpness" = mean of per-row max attention weight.
        # Closer to 1.0 => closer to one-hot (saturated softmax).
        sharp_scaled  = attn_scaled.max(axis=-1).mean()
        sharp_noscale = attn_noscale.max(axis=-1).mean()
        print(f"{d_k_test:>6} | {sharp_scaled:>16.3f} | {sharp_noscale:>16.3f}")

    # ---- Masking smoke test ------------------------------------------------ #
    print()
    print("=" * 60)
    print("Masking smoke test: key position 3 (0-indexed) is padding")
    print("=" * 60)
    n, d_k, d_v = 4, 4, 4
    Q = rng.standard_normal((n, d_k))
    K = rng.standard_normal((n, d_k))
    V = rng.standard_normal((n, d_v))

    # Only KEY column 3 is masked.  Query rows are NOT masked
    # (that would create -inf rows -> NaN after softmax).
    mask = np.ones((n, n), dtype=bool)
    mask[:, 3] = False

    _, attn_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
    print(f"attn_masked column 3 (should be all zeros):")
    print(f"  {attn_masked[:, 3]}")
    print(f"row sums (should be 1.0):")
    print(f"  {attn_masked.sum(axis=-1)}")

    # ---- Multi-head sanity: h=1 should match T2.1 single-head -------------- #
    print()
    print("=" * 60)
    print("T2.2 sanity check: h=1 multi-head == T2.1 single-head")
    print("=" * 60)

    n, d_model = 4, 8
    X = rng.standard_normal((n, d_model))
    W_Q = rng.standard_normal((d_model, d_model))
    W_K = rng.standard_normal((d_model, d_model))
    W_V = rng.standard_normal((d_model, d_model))
    W_O = np.eye(d_model)  # identity, so output of MH = output of SH directly

    out_mh, attn_mh = multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads=1)

    Q_sh = X @ W_Q
    K_sh = X @ W_K
    V_sh = X @ W_V
    out_sh, attn_sh = scaled_dot_product_attention(Q_sh, K_sh, V_sh)

    diff_out  = np.abs(out_mh - out_sh).max()
    diff_attn = np.abs(attn_mh.squeeze(0) - attn_sh).max()
    print(f"max |out_mh  - out_sh |  = {diff_out:.2e}   (expect < 1e-12)")
    print(f"max |attn_mh - attn_sh|  = {diff_attn:.2e}   (expect < 1e-12)")

    # ---- Multi-head shape check: h=2 --------------------------------------- #
    print()
    print("=" * 60)
    print("T2.2 shape check: h=2")
    print("=" * 60)

    out, attn = multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads=2)
    print(f"X.shape       = {X.shape}")
    print(f"output.shape  = {out.shape}    (expected ({n}, {d_model}))")
    print(f"attn.shape    = {attn.shape}   (expected (2, {n}, {n}))")
    print(f"per-head row sums (each row should be 1.0):")
    print(f"{attn.sum(axis=-1)}")


if __name__ == "__main__":
    main()
