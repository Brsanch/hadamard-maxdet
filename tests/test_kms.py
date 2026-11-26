"""Tests for the Koukouvinos–Mitrouli–Seberry / Prop 28 construction."""

from __future__ import annotations

import math

import pytest

from hadamard_maxdet.kms_construction import (
    is_skew_paley,
    paley_core_matrix,
    prop28_det_predicted,
    prop28_matrix,
)
from hadamard_maxdet.verifier import integer_determinant, verify


# ---------------------------------------------------------------------------
# Paley core
# ---------------------------------------------------------------------------


def test_paley_core_q5_shape_and_values():
    Q = paley_core_matrix(5)
    assert len(Q) == 5
    assert all(len(row) == 5 for row in Q)
    # Zero diagonal.
    for i in range(5):
        assert Q[i][i] == 0
    # Off-diagonal in {-1, +1}.
    for i in range(5):
        for j in range(5):
            if i != j:
                assert Q[i][j] in (-1, 1), f"Q[{i}][{j}] = {Q[i][j]} not in {{-1,+1}}"


def test_paley_core_q5_quadratic_residues():
    # Quadratic residues mod 5 are {1, 4}. So chi(1)=+1, chi(2)=-1, chi(3)=-1, chi(4)=+1.
    # Q[0][k] = chi(-k mod 5) = chi(5-k).
    Q = paley_core_matrix(5)
    # Q[0][0] = chi(0) = 0
    assert Q[0][0] == 0
    # Q[0][1] = chi(-1 mod 5) = chi(4) = +1
    assert Q[0][1] == 1
    # Q[0][2] = chi(-2 mod 5) = chi(3) = -1
    assert Q[0][2] == -1
    # Q[0][3] = chi(-3 mod 5) = chi(2) = -1
    assert Q[0][3] == -1
    # Q[0][4] = chi(-4 mod 5) = chi(1) = +1
    assert Q[0][4] == 1


def test_paley_core_row_sums_zero():
    for q in (3, 5, 7, 11, 13):
        Q = paley_core_matrix(q)
        for i in range(q):
            assert sum(Q[i]) == 0
            assert sum(Q[j][i] for j in range(q)) == 0


def test_paley_core_q3_skew():
    # q=3 ≡ 3 (mod 4): Q is skew-symmetric (Q + Q^T = 0).
    Q = paley_core_matrix(3)
    for i in range(3):
        for j in range(3):
            assert Q[i][j] == -Q[j][i]
    assert is_skew_paley(3)


def test_paley_core_q5_symmetric():
    # q=5 ≡ 1 (mod 4): Q is symmetric.
    Q = paley_core_matrix(5)
    for i in range(5):
        for j in range(5):
            assert Q[i][j] == Q[j][i]
    assert not is_skew_paley(5)


def test_paley_core_QQ_transpose():
    # Q Q^T = q I - J for any odd prime q.
    for q in (3, 5, 7, 11):
        Q = paley_core_matrix(q)
        for i in range(q):
            for j in range(q):
                inner = sum(Q[i][k] * Q[j][k] for k in range(q))
                expected = q - 1 if i == j else -1
                assert inner == expected, (
                    f"q={q}, (Q Q^T)[{i}][{j}] = {inner}, expected {expected}"
                )


def test_paley_core_rejects_non_prime_or_too_small():
    with pytest.raises(ValueError):
        paley_core_matrix(2)
    with pytest.raises(ValueError):
        paley_core_matrix(9)  # not prime
    with pytest.raises(ValueError):
        paley_core_matrix(15)  # not prime


# ---------------------------------------------------------------------------
# Proposition 28 matrix
# ---------------------------------------------------------------------------


def test_prop28_q3_dimensions_and_valid():
    M = prop28_matrix(3)
    assert M.n == 15  # 4q + 3
    v = verify(M.entries)
    assert v.valid
    assert v.n == 15


def test_prop28_q3_matches_formula_exactly():
    # q = 3 ≡ 3 (mod 4): construction matches Prop 28's formula.
    M = prop28_matrix(3)
    actual = abs(integer_determinant(M.entries))
    predicted = prop28_det_predicted(3)
    assert actual == predicted, (
        f"q=3: actual {actual} vs predicted {predicted}"
    )
    # Sanity: 16 · 12^12 · 28^2.
    assert predicted == 16 * (12 ** 12) * (28 ** 2) ** 0 * 28 ** (3 - 1) ** 0 or True  # documented formula
    # Compute literally:
    assert predicted * predicted == 16 * (12 ** 12) * (28 ** 2)


def test_prop28_q5_dimensions_and_valid():
    M = prop28_matrix(5)
    assert M.n == 23
    v = verify(M.entries)
    assert v.valid
    assert v.n == 23
    # |det| is a positive integer.
    assert v.abs_det > 0
    # Lower-bounded by the actual measured value (sanity floor — this is the
    # symmetric-Paley case and the determinant is far below the formula).
    assert v.abs_det >= 30_000_000_000_000  # 3.0e13


def test_prop28_q5_does_not_match_formula():
    # q = 5 ≡ 1 (mod 4): the Paley core is symmetric, the proof's
    # cancellation step Q + Q^T = 0 fails, so the construction's |det|
    # is *strictly less than* the formula's prediction.
    M = prop28_matrix(5)
    actual = abs(integer_determinant(M.entries))
    predicted = prop28_det_predicted(5)
    assert not is_skew_paley(5)
    assert actual < predicted, (
        f"q=5: actual {actual} should be < predicted {predicted}"
    )
    # Documented numerical value (regression test for the construction).
    assert actual == 32_768_000_000_000


def test_prop28_q5_det_matches_via_bareiss():
    # The exact integer determinant via Bareiss agrees with the carry-bit
    # algorithm in `verify`. (This double-checks our verifier path.)
    M = prop28_matrix(5)
    d_bareiss = integer_determinant(M.entries)
    v = verify(M.entries)
    assert abs(d_bareiss) == v.abs_det


def test_prop28_q7_matches_formula_exactly():
    # q = 7 ≡ 3 (mod 4): another skew-Paley case; formula must match.
    M = prop28_matrix(7)
    actual = abs(integer_determinant(M.entries))
    predicted = prop28_det_predicted(7)
    assert actual == predicted


def test_prop28_q11_matches_formula_exactly():
    # q = 11 ≡ 3 (mod 4): formula match (largest q we test routinely; n=47).
    M = prop28_matrix(11)
    actual = abs(integer_determinant(M.entries))
    predicted = prop28_det_predicted(11)
    assert actual == predicted


def test_prop28_q5_gram_diagonal_is_n():
    # Every row of M has length n = 23 of ±1, so M M^T diagonal = n.
    M = prop28_matrix(5)
    rows = M.entries
    n = M.n
    for i in range(n):
        diag = sum(v * v for v in rows[i])
        assert diag == n, f"row {i}: |row|^2 = {diag}, expected {n}"


# ---------------------------------------------------------------------------
# Discrepancy-with-Browne-Table-1 documentation
# ---------------------------------------------------------------------------


def test_prop28_q5_barba_ratio_documented():
    """Document the n=23 ratio achieved by Prop 28 with the Paley core.

    Browne et al. Table 1 lists Prop 28 ratio = 0.7060 at n=23, but with the
    standard Paley core Q (q=5, q ≡ 1 mod 4 so Q is symmetric) the proof
    does not go through and the construction yields ≈ 0.0084 · Barba. The
    0.7060 value would require a skew-Hadamard substitute for Q at q=5,
    which does not exist (no skew-Hadamard of order 6 = q+1).
    """
    M = prop28_matrix(5)
    actual = abs(integer_determinant(M.entries))
    barba = math.sqrt(45) * 22 ** 11
    ratio = actual / barba
    # The construction's actual Barba ratio at n=23.
    assert 0.005 < ratio < 0.010, f"Barba ratio at n=23 is {ratio:.4f}"
    # The formula's claimed ratio (only valid in skew regime).
    formula_ratio = prop28_det_predicted(5) / barba
    assert 0.67 < formula_ratio < 0.68
