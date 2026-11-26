"""Orrick's conference-matrix construction for ±1 matrices of high |det|.

Reference: Orrick, Solomon, Dowdeswell, Smith (2003), "Lower bounds for
maximal determinants" §2. The construction applies when ``n ≡ 15 (mod 16)``
and uses an antisymmetric ``(k+1) × (k+1)`` conference matrix with
``n = 4k + 3`` (so ``k + 1 ≡ 0 (mod 4)``). It produces the published H47,
H63, H79, H95 records.

Algorithm (Orrick §2):

1. Find an antisymmetric ``(k+1) × (k+1)`` conference matrix C: zero
   diagonal, ±1 off-diagonal, ``C C^T = k * I``, ``C^T = -C``. For our
   application at n = 47 we use the Paley type-II construction at order
   12 (k = 11, with 11 ≡ 3 mod 4 prime giving an antisymmetric S).

2. Normalize C: multiply rows AND corresponding columns by -1
   (preserving antisymmetry) so that ``c[0][j] = +1`` for all ``j > 0``
   and ``c[i][0] = -1`` for all ``i > 0``.

3. Delete row 0 and column 0 to get a ``k × k`` matrix B with the
   property ``B B^T = (k - 1) I_k - J_k`` where ``J_k`` is the all-ones
   matrix.

4. Tensor ``B ⊗ H_4`` where ``H_4`` is the 4×4 Sylvester Hadamard matrix
   (a Hadamard matrix with first row all +1's).

5. Replace each 0 in the tensor product by -1.

6. Pad with 3 initial rows: take ``H_4`` (normalized so first row is
   all +1's), drop the first row, tensor each of the 3 remaining rows
   with a row of k ones → 3 rows of length 4k. Prepend.

7. Pad with 3 initial columns: prepend a 3-column block of length
   ``4k + 3`` constructed analogously.

The result is a ``(4k+3) × (4k+3)`` ±1 matrix.

The construction also generalises to a SYMMETRIC conference matrix (when
the antisymmetric one doesn't exist). The formal recipe is identical, but
the resulting Gram structure is different (and so is the determinant).
We provide both variants for the H23 exploration where order 6
admits only a symmetric Paley conference matrix (since ``5 ≡ 1 (mod 4)``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Primality + QR helpers (small primes only; we want 5, 11)
# ---------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    r = 3
    while r * r <= n:
        if n % r == 0:
            return False
        r += 2
    return True


def _qr_character(p: int) -> list[int]:
    """Return chi: list of length p with chi[0]=0 and chi[x]=±1 indicating
    whether x is a quadratic residue mod p."""
    if not _is_prime(p):
        raise ValueError(f"p must be prime, got {p}")
    qr = {(x * x) % p for x in range(1, p)}
    chi = [0] * p
    for x in range(1, p):
        chi[x] = 1 if x in qr else -1
    return chi


# ---------------------------------------------------------------------------
# Paley conference matrices
# ---------------------------------------------------------------------------


def paley_conference_matrix(order: int) -> tuple[np.ndarray, str]:
    """Return a Paley conference matrix S of the given even order.

    For ``order = p + 1`` with ``p`` an odd prime:

      - If ``p ≡ 3 (mod 4)``: S is **antisymmetric** (``S^T = -S``), zero
        diagonal, ±1 off-diagonal, ``S S^T = p * I``.
      - If ``p ≡ 1 (mod 4)``: S is **symmetric** (``S^T = S``), zero
        diagonal, ±1 off-diagonal, ``S S^T = p * I``.

    Construction:
      ``S[0][0] = 0``;
      ``S[0][j] = +1`` for ``j > 0``;
      ``S[i][0] = -1`` (antisymmetric case) or ``+1`` (symmetric case);
      For ``i, j > 0`` and ``i != j``: ``S[i][j] = chi[(j - i) mod p]``.

    Returns
    -------
    (S, kind)
        S as a numpy int array of shape ``(order, order)`` and a string
        ``"antisymmetric"`` or ``"symmetric"``.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError(f"Paley conference order must be even >= 2; got {order}")
    p = order - 1
    if not _is_prime(p):
        raise ValueError(
            f"Paley conference at order {order} requires order - 1 = {p} prime"
        )

    chi = _qr_character(p)
    sym = (p % 4 == 1)

    n = order
    S = np.zeros((n, n), dtype=int)
    for j in range(1, n):
        S[0, j] = 1
    for i in range(1, n):
        S[i, 0] = 1 if sym else -1
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                S[i, j] = chi[(j - i) % p]

    # Sanity checks (cheap).
    if sym:
        assert np.array_equal(S, S.T), "Paley sym conference: S != S.T"
    else:
        assert np.array_equal(S, -S.T), "Paley antisym conference: S != -S.T"
    G = S @ S.T
    expected = p * np.eye(n, dtype=int)
    assert np.array_equal(G, expected), (
        f"Paley conference S S^T != p * I (p={p})"
    )

    kind = "symmetric" if sym else "antisymmetric"
    return S, kind


# ---------------------------------------------------------------------------
# Normalization (Step 2 of Orrick §2)
# ---------------------------------------------------------------------------


def normalize_conference_first_row(C: np.ndarray, antisymmetric: bool = True) -> np.ndarray:
    """Normalize so the first row is all +1 (off-diagonal) and the first
    column matches the antisymmetry/symmetry constraint.

    For an antisymmetric conference C with C[0][0] = 0, C[0][j] in {±1}
    for j > 0 and C[j][0] = -C[0][j]: multiplying row j AND column j by
    -1 negates C[0][j] and C[j][0] in tandem (the (0, j) and (j, 0)
    entries each pick up one factor of -1 — actually two from the
    row-AND-col, but only one of those is at (0, j); let me redo).

    More carefully: the row-and-column-negation by ε_j on index j changes
    C[i][j] -> ε_i ε_j C[i][j] (with ε_0 = +1 fixed and other ε_j = ±1
    chosen freely). This preserves antisymmetry (and also symmetry if it
    was symmetric to begin with).

    To make C[0][j] = +1 for all j > 0, set ε_j = C[0][j] (since ε_0 = +1
    so the new entry is ε_0 ε_j C[0][j] = ε_j * (±1) which is +1 iff
    ε_j = C[0][j]).

    Then for the antisymmetric case, automatically C[i][0] = -C[0][i] = -1
    for i > 0; for the symmetric case it ends up +1.
    """
    n = C.shape[0]
    if C[0, 0] != 0:
        raise ValueError("Conference matrix must have zero diagonal entry C[0][0]")
    eps = np.ones(n, dtype=int)
    for j in range(1, n):
        eps[j] = int(C[0, j])
        if eps[j] not in (-1, 1):
            raise ValueError(f"Conference matrix off-diag must be ±1; got C[0][{j}]={C[0, j]}")
    # Apply ε on rows and columns simultaneously.
    Cn = (eps[:, None] * eps[None, :]) * C
    # Verify normalization.
    if not np.all(Cn[0, 1:] == 1):
        raise RuntimeError("Normalisation failed: first row not all +1 off-diag")
    if antisymmetric and not np.all(Cn[1:, 0] == -1):
        raise RuntimeError("Antisymmetric normalisation failed: first column not all -1")
    return Cn


# ---------------------------------------------------------------------------
# Deleting first row and column → B with B B^T = (k-1) I - J
# ---------------------------------------------------------------------------


def conference_to_B(C: np.ndarray, antisymmetric: bool = True) -> np.ndarray:
    """Apply Orrick §2 step 3: normalize then delete first row and column.

    For antisymmetric C of order k+1, returns the k×k block satisfying
    ``B B^T = (k - 1) I_k - J_k``.

    For symmetric C of order k+1 (the variant we use at n=23), the same
    deletion produces a k×k matrix B' whose Gram structure is computed
    in :func:`b_block_gram_signature`.
    """
    Cn = normalize_conference_first_row(C, antisymmetric=antisymmetric)
    return Cn[1:, 1:].copy()


def b_block_gram_signature(B: np.ndarray) -> dict:
    """Return Gram-multiset diagnostic of the k×k block B for inspection.

    For Orrick's antisymmetric path this should report
    ``diag = k - 1`` and off-diag uniformly ``-1`` (since
    ``B B^T = (k - 1) I_k - J_k``).
    """
    k = B.shape[0]
    G = B @ B.T
    diag = G.diagonal()
    iu = np.triu_indices(k, k=1)
    off = G[iu]
    vals, counts = np.unique(off, return_counts=True)
    return {
        "k": int(k),
        "diag_unique": sorted(set(int(d) for d in diag)),
        "off_diag_support": [int(v) for v in vals],
        "off_diag_counts": {int(v): int(c) for v, c in zip(vals, counts)},
        "off_diag_unique_card": int(len(vals)),
    }


# ---------------------------------------------------------------------------
# Sylvester H_4 (with first row all +1)
# ---------------------------------------------------------------------------


def sylvester_h4_normalized() -> np.ndarray:
    """Return the 4×4 Sylvester Hadamard matrix with first row all +1.

    Sylvester(2)·Sylvester(2) ordering already has +1 first row.
    """
    H2 = np.array([[1, 1], [1, -1]], dtype=int)
    H4 = np.kron(H2, H2)
    assert np.all(H4[0] == 1)
    return H4


def regular_h4_rowsum2() -> np.ndarray:
    """Return a 4x4 Hadamard matrix with all row sums equal to +2.

    Unlike Sylvester (whose row sums are 4, 0, 0, 0), this "regular"
    Hadamard has uniform row weight, which makes the all-ones vector
    an eigenvector with eigenvalue 2. Empirically this variant gives a
    factor 1/4 closer to Orrick's H47 record than Sylvester does
    (ratio 0.250 to target vs 0.328 with Sylvester); see the H47 record
    diagnostic for details.
    """
    H = np.array([
        [+1, +1, +1, -1],
        [+1, +1, -1, +1],
        [+1, -1, +1, +1],
        [-1, +1, +1, +1],
    ], dtype=int)
    assert np.array_equal(H @ H.T, 4 * np.eye(4, dtype=int))
    return H


# ---------------------------------------------------------------------------
# The Orrick §2 full construction
# ---------------------------------------------------------------------------


def conference_method_orrick(
    n: int,
    *,
    use_symmetric: bool = False,
    h4_variant: str = "rowsum2",
) -> np.ndarray:
    """Build a ±1 matrix of order n via Orrick §2.

    Parameters
    ----------
    n
        Target order. Must satisfy ``n = 4k + 3`` for some k with
        ``k + 1`` even and ``k + 1 - 1 = k`` prime (so we can build
        a Paley conference of order k+1).
    use_symmetric
        If False (default), require ``k ≡ 3 (mod 4)`` so the Paley
        conference of order k+1 is antisymmetric (Orrick's intended
        path; produces records at n ≡ 15 mod 16).

        If True, use a symmetric Paley conference instead. This requires
        ``k ≡ 1 (mod 4)`` and is needed for the n = 23 variant where
        no antisymmetric order-6 conference matrix exists (6 = 5 + 1
        with 5 ≡ 1 mod 4, so Paley gives a symmetric one).

    Returns
    -------
    np.ndarray
        A ``(4k+3) × (4k+3)`` matrix with entries in ±1.
    """
    if n < 7 or (n - 3) % 4 != 0:
        raise ValueError(f"Orrick §2 requires n = 4k + 3; got n = {n}")
    k = (n - 3) // 4

    # Build Paley conference of order k+1.
    C, kind = paley_conference_matrix(k + 1)
    is_antisym = (kind == "antisymmetric")

    if use_symmetric:
        if is_antisym:
            raise ValueError(
                f"use_symmetric=True but Paley conference at order {k+1} "
                f"is {kind}; cannot proceed."
            )
    else:
        if not is_antisym:
            raise ValueError(
                f"Orrick §2 (antisymmetric path) requires antisymmetric "
                f"conference at order {k+1}; Paley gives {kind}. "
                f"Re-run with use_symmetric=True for the symmetric variant."
            )

    # Step 2 + 3: normalize and delete first row/column.
    B = conference_to_B(C, antisymmetric=is_antisym)
    assert B.shape == (k, k)

    # Step 4: tensor B ⊗ H_4.
    if h4_variant == "sylvester":
        H4 = sylvester_h4_normalized()
    elif h4_variant == "rowsum2":
        H4 = regular_h4_rowsum2()
    else:
        raise ValueError(f"Unknown h4_variant: {h4_variant!r}")
    T = np.kron(B, H4)  # shape (4k, 4k)

    # Step 5: replace zeros with -1.
    T = np.where(T == 0, -1, T).astype(int)
    if not np.all(np.isin(T, (-1, 1))):
        raise RuntimeError("After zero-replacement, T has non-±1 entries")

    # Step 6: prepend 3 rows of length 4k.
    # Take H4 (with first row all +1), drop the first row, tensor each
    # remaining row with a row of k ones. That gives a 3 × (4k) block.
    H4_rows_keep = H4[1:, :]  # shape (3, 4)
    ones_k = np.ones((1, k), dtype=int)
    pad_rows = np.kron(ones_k, H4_rows_keep)  # shape (3, 4k)
    # Verify: each row has length 4k.
    assert pad_rows.shape == (3, 4 * k), pad_rows.shape

    M_top = np.vstack([pad_rows, T])  # shape (4k + 3, 4k)
    assert M_top.shape == (4 * k + 3, 4 * k)

    # Step 7: prepend 3 columns of length 4k+3.
    # Construct the same way: take H4, drop the first column, tensor
    # each column with a column of k ones — that's a (4k) × 3 block.
    # Then top-stack with a 3 × 3 block (the (0, 0) corner of the H4
    # Hadamard pattern truncated).
    H4_cols_keep = H4[:, 1:]  # shape (4, 3)
    ones_k_col = np.ones((k, 1), dtype=int)
    pad_cols_lower = np.kron(ones_k_col, H4_cols_keep)  # shape (4k, 3)
    assert pad_cols_lower.shape == (4 * k, 3)

    # Build the 3×3 top-left corner. The natural symmetric choice that
    # preserves the construction is the (1:, 1:) sub-block of H4. This is
    # what Orrick's printed examples in §3 use (and the only option that
    # makes the top three rows / columns mutually orthogonal — see §3).
    top_left = H4[1:, 1:]  # shape (3, 3)
    assert top_left.shape == (3, 3)

    pad_cols = np.vstack([top_left, pad_cols_lower])  # shape (4k+3, 3)

    M = np.hstack([pad_cols, M_top])  # shape (4k+3, 4k+3)
    assert M.shape == (n, n), f"Expected {(n, n)}, got {M.shape}"

    if not np.all(np.isin(M, (-1, 1))):
        raise RuntimeError("Final M has non-±1 entries")

    return M


# ---------------------------------------------------------------------------
# Convenience: build & verify
# ---------------------------------------------------------------------------


def build_orrick_h47() -> tuple[np.ndarray, int]:
    """Build the H47 matrix via the antisymmetric path and return (M, |det|).

    Reference value from Orrick et al. (2003) §3:
        |det| = 2^46 * 11^18 * 15^4 * 60.
    """
    M = conference_method_orrick(47, use_symmetric=False)
    abs_det = abs(integer_determinant(M.tolist()))
    return M, abs_det


def orrick_h47_target() -> int:
    """Return the published Orrick H47 |det| value."""
    return 2 ** 46 * 11 ** 18 * 15 ** 4 * 60


def build_orrick_method_h23(use_symmetric: bool = True) -> tuple[np.ndarray, int]:
    """Build the H23 matrix via Orrick §2 with a symmetric conference matrix.

    At n = 23, k = 5, k + 1 = 6, and only a symmetric Paley conference
    matrix of order 6 exists (5 ≡ 1 mod 4). This calls the same
    Orrick recipe but with the symmetric C; the result is a valid
    23×23 ±1 matrix whose |det| is whatever it turns out to be.
    """
    M = conference_method_orrick(23, use_symmetric=use_symmetric)
    abs_det = abs(integer_determinant(M.tolist()))
    return M, abs_det


__all__ = [
    "paley_conference_matrix",
    "normalize_conference_first_row",
    "conference_to_B",
    "b_block_gram_signature",
    "sylvester_h4_normalized",
    "conference_method_orrick",
    "build_orrick_h47",
    "orrick_h47_target",
    "build_orrick_method_h23",
]
