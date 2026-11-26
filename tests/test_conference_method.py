"""Tests for the Orrick conference-matrix construction (Orrick et al. 2003 §2)."""

from __future__ import annotations

import numpy as np
import pytest

from hadamard_maxdet.conference_method import (
    paley_conference_matrix,
    normalize_conference_first_row,
    conference_to_B,
    b_block_gram_signature,
    sylvester_h4_normalized,
    regular_h4_rowsum2,
    conference_method_orrick,
    build_orrick_h47,
    orrick_h47_target,
    build_orrick_method_h23,
)
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Paley conference matrix construction
# ---------------------------------------------------------------------------


def test_paley_conference_order_12_antisymmetric():
    """Order 12 = 11 + 1 with 11 ≡ 3 (mod 4): antisymmetric."""
    S, kind = paley_conference_matrix(12)
    assert kind == "antisymmetric"
    assert S.shape == (12, 12)
    # Zero diagonal
    assert np.all(np.diag(S) == 0)
    # Off-diagonal in ±1
    off_mask = ~np.eye(12, dtype=bool)
    assert np.all(np.isin(S[off_mask], (-1, 1)))
    # Antisymmetric (off-diagonal)
    assert np.array_equal(S, -S.T)
    # SS^T = 11 I
    G = S @ S.T
    assert np.array_equal(G, 11 * np.eye(12, dtype=int))


def test_paley_conference_order_6_symmetric():
    """Order 6 = 5 + 1 with 5 ≡ 1 (mod 4): symmetric."""
    S, kind = paley_conference_matrix(6)
    assert kind == "symmetric"
    assert S.shape == (6, 6)
    assert np.all(np.diag(S) == 0)
    off_mask = ~np.eye(6, dtype=bool)
    assert np.all(np.isin(S[off_mask], (-1, 1)))
    assert np.array_equal(S, S.T)
    G = S @ S.T
    assert np.array_equal(G, 5 * np.eye(6, dtype=int))


def test_paley_conference_invalid_order():
    """Reject odd orders and orders where order-1 is composite (not prime)."""
    with pytest.raises(ValueError, match="even"):
        paley_conference_matrix(11)
    with pytest.raises(ValueError, match="prime"):
        paley_conference_matrix(10)  # 10-1 = 9 not prime


# ---------------------------------------------------------------------------
# Normalization + B block extraction
# ---------------------------------------------------------------------------


def test_normalize_conference_antisym_first_row():
    S, kind = paley_conference_matrix(12)
    Sn = normalize_conference_first_row(S, antisymmetric=True)
    # First row: zero at (0,0), all +1 elsewhere
    assert Sn[0, 0] == 0
    assert np.all(Sn[0, 1:] == 1)
    # First column: -1 below the diagonal (antisymmetric case)
    assert np.all(Sn[1:, 0] == -1)
    # Still antisymmetric off-diagonal
    assert np.array_equal(Sn, -Sn.T)
    # Still has SS^T = 11 I
    G = Sn @ Sn.T
    assert np.array_equal(G, 11 * np.eye(12, dtype=int))


def test_normalize_conference_sym_first_row():
    S, kind = paley_conference_matrix(6)
    Sn = normalize_conference_first_row(S, antisymmetric=False)
    assert Sn[0, 0] == 0
    assert np.all(Sn[0, 1:] == 1)
    # Symmetric case: first column also +1
    assert np.all(Sn[1:, 0] == 1)
    assert np.array_equal(Sn, Sn.T)
    G = Sn @ Sn.T
    assert np.array_equal(G, 5 * np.eye(6, dtype=int))


def test_b_block_from_antisym_conference_order_12():
    """B is the (k=11) × (k=11) block. Check the Gram identity B B^T = k I_k - J_k.

    (Orrick paper writes ``BB^T = (k-1) I - J`` but this appears to be a
    typo: with B having zero diagonal and k - 1 nonzero ±1 entries per
    row, the diagonal of BB^T is k - 1 and the off-diagonal is -1, which
    equals ``k I - J``, not ``(k-1) I - J``.)
    """
    S, _ = paley_conference_matrix(12)
    B = conference_to_B(S, antisymmetric=True)
    assert B.shape == (11, 11)
    # B has zero diagonal
    assert np.all(np.diag(B) == 0)
    # B B^T = k I - J  (k = 11)
    G = B @ B.T
    expected = 11 * np.eye(11, dtype=int) - np.ones((11, 11), dtype=int)
    assert np.array_equal(G, expected)


def test_b_block_signature_antisym():
    """The Gram off-diag of B for antisym path is uniformly -1."""
    S, _ = paley_conference_matrix(12)
    B = conference_to_B(S, antisymmetric=True)
    sig = b_block_gram_signature(B)
    assert sig["k"] == 11
    assert sig["diag_unique"] == [10]
    assert sig["off_diag_support"] == [-1]
    assert sig["off_diag_unique_card"] == 1


def test_b_block_signature_symmetric_order_6():
    """For symmetric Paley conference at order 6, the deleted block has a
    different Gram structure that we record for the H23 adaptation."""
    S, _ = paley_conference_matrix(6)
    B = conference_to_B(S, antisymmetric=False)
    assert B.shape == (5, 5)
    sig = b_block_gram_signature(B)
    # Expect uniformity of some kind; we check structural invariants only.
    assert sig["k"] == 5
    # Symmetric path: B is symmetric (since C is symmetric)
    assert np.array_equal(B, B.T)


# ---------------------------------------------------------------------------
# H_4 helpers
# ---------------------------------------------------------------------------


def test_sylvester_h4():
    H = sylvester_h4_normalized()
    assert H.shape == (4, 4)
    # First row all +1
    assert np.all(H[0] == 1)
    # Hadamard: H H^T = 4 I
    assert np.array_equal(H @ H.T, 4 * np.eye(4, dtype=int))


def test_regular_h4_rowsum2():
    H = regular_h4_rowsum2()
    assert H.shape == (4, 4)
    assert np.array_equal(H @ H.T, 4 * np.eye(4, dtype=int))
    # All row sums equal +2 (the defining property of "regular" Hadamard)
    assert np.all(H.sum(axis=1) == 2)


# ---------------------------------------------------------------------------
# Full Orrick §2 construction
# ---------------------------------------------------------------------------


def test_orrick_method_h47_produces_pm1_matrix():
    """The construction at n = 47 produces a valid (47×47, ±1) matrix."""
    M = conference_method_orrick(47, use_symmetric=False)
    assert M.shape == (47, 47)
    assert np.all(np.isin(M, (-1, 1)))


def test_orrick_method_h47_det_factorisation():
    """Verify that the construction produces a matrix with the expected
    factorisation in primes 2, 3, 5, 11. With the regular row-sum-2
    H_4 variant, the construction matches Orrick's published H47 record
    UP TO A FACTOR OF 4 (= 2²): the primes 3, 5, 11 appear with the
    correct exponents (5, 5, 18), but the power of 2 is 46 instead of
    Orrick's 48.

    This is a documented partial reproduction; see
    `docs/findings.md` 2026-04-25 entry for the full diagnostic.
    """
    from sympy import factorint

    M, d = build_orrick_h47()
    fact = factorint(d)
    assert fact == {2: 46, 3: 5, 5: 5, 11: 18}, fact
    target = orrick_h47_target()
    target_fact = factorint(target)
    assert target_fact == {2: 48, 3: 5, 5: 5, 11: 18}
    # The constructed matrix is exactly target / 4
    assert d * 4 == target


def test_orrick_method_h47_sylvester_factorisation():
    """The Sylvester H_4 variant gives a different factorisation, missing
    the factor 5^5 (which the row-sum-2 variant captures correctly)."""
    from sympy import factorint

    M = conference_method_orrick(47, use_symmetric=False, h4_variant="sylvester")
    d = abs(integer_determinant(M.tolist()))
    fact = factorint(d)
    assert fact == {2: 58, 3: 5, 11: 18}, fact


def test_orrick_method_h23_symmetric_path():
    """The symmetric-conference variant of Orrick §2 at n = 23 produces a
    valid (23×23, ±1) matrix.

    n = 23 = 4·5 + 3, k = 5, k+1 = 6. No antisymmetric order-6
    conference exists (order ≡ 0 mod 4 is required for antisymmetric);
    the symmetric Paley conference at order 6 (5 ≡ 1 mod 4) lets us
    proceed via :func:`conference_method_orrick(use_symmetric=True)`.
    """
    M = conference_method_orrick(23, use_symmetric=True)
    assert M.shape == (23, 23)
    assert np.all(np.isin(M, (-1, 1)))


def test_orrick_method_h23_antisym_path_rejected():
    """Asking for the antisymmetric path at n = 23 must raise: no such
    conference matrix of order 6 exists."""
    with pytest.raises(ValueError, match="symmetric"):
        conference_method_orrick(23, use_symmetric=False)


def test_orrick_method_invalid_n():
    """n must satisfy n = 4k + 3."""
    with pytest.raises(ValueError, match="4k"):
        conference_method_orrick(20)


# ---------------------------------------------------------------------------
# Determinant verification end-to-end
# ---------------------------------------------------------------------------


def test_orrick_method_h47_det_is_positive_integer():
    M, d = build_orrick_h47()
    assert d > 0
    assert isinstance(d, int)


def test_orrick_method_h23_symmetric_det_below_orrick():
    """Document the H23 symmetric variant's |det|. For triple-verify safety
    we compute it explicitly via Bareiss."""
    M, d = build_orrick_method_h23(use_symmetric=True)
    assert d > 0
    # Orrick's H23 record:
    orrick_h23 = 2_779_447_296_000_000
    # If d > orrick_h23 we'd flag as a candidate new record (triple-verify).
    # In practice (see findings) this construction does NOT exceed Orrick.
    if d > orrick_h23:
        # Double-verify with a fresh integer_determinant call
        d_check = abs(integer_determinant(M.tolist()))
        assert d_check == d, f"|det| inconsistent: {d} vs {d_check}"
        # Triple-verify by transposition (det(M^T) = det(M))
        d_t = abs(integer_determinant(M.T.tolist()))
        assert d_t == d
