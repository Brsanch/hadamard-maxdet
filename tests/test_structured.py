"""Tests for the structured-ansatz module.

Covers:

  (a) Constructors produce valid ±1 matrices that pass :func:`verify`.

  (b) SA on the two-circulant block form reaches |det| = 8^4 = 4096
      at n = 8 (Hadamard tight — verified exhaustively to be reachable).

      We do NOT assert 12^6 at n = 12: exhaustive enumeration over all
      2^12 = 4096 (a, b) pairs shows the block form
      ``[[A, B], [B^T, -A^T]]`` tops out at 2,097,152 at n = 12,
      well below the Hadamard maximum 2,985,984. This is a property of
      this specific block form (different two-circulant block forms
      exist in the literature that do reach 12^6; this module
      implements the canonical ``[[A, B], [B^T, -A^T]]`` form only).

  (c) SA on bordered-circulant at n = 11 reaches |det| = 281,600,
      which is the exhaustive bordered-circulant maximum at n = 11.
      (The true H11 record is 327,680, which is NOT attainable by a
      bordered-circulant matrix in the canonical form — exhaustive
      enumeration over all 2^10 = 1024 core generators confirms 281,600
      is the ansatz ceiling. The original task spec anticipated 327,680
      but that requires a non-circulant-core construction.)

  (d) Skew-circulant validity — constructor rejects bad inputs, and
      sa_skew_circulant on n = 7 reaches the Hadamard-equivalent
      maximum for that ansatz.
"""

import random

import pytest

from hadamard_maxdet import verify
from hadamard_maxdet.structured import (
    bordered_circulant,
    sa_bordered_circulant,
    sa_skew_circulant,
    sa_two_circulant,
    skew_circulant,
    skew_generator_from_free_bits,
    two_circulant,
)
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# (a) Constructors produce valid ±1 matrices.
# ---------------------------------------------------------------------------


def test_two_circulant_is_valid_pm1_at_n8():
    a = [1, -1, 1, -1]
    b = [1, 1, -1, -1]
    M = two_circulant(a, b)
    assert M.n == 8
    v = verify(M.entries)
    assert v.valid, v.error


def test_two_circulant_rejects_length_mismatch():
    with pytest.raises(ValueError):
        two_circulant([1, -1, 1], [1, 1, -1, -1])


def test_two_circulant_rejects_non_pm1():
    with pytest.raises(ValueError):
        two_circulant([1, 0, 1, -1], [1, 1, -1, -1])


def test_bordered_circulant_is_valid_pm1_at_n11():
    core = [1, -1, -1, 1, 1, -1, 1, -1, 1, -1]
    M = bordered_circulant(core)
    assert M.n == 11
    v = verify(M.entries)
    assert v.valid, v.error
    # Canonical border: row 0 is all +1, column 0 below the top is -1.
    assert M.entries[0] == [1] * 11
    for i in range(1, 11):
        assert M.entries[i][0] == -1


def test_bordered_circulant_rejects_non_pm1():
    with pytest.raises(ValueError):
        bordered_circulant([1, 0, 1, -1])


def test_skew_circulant_is_valid_pm1_at_n7():
    # n = 7 odd; free bits length (7-1)/2 = 3.
    gen = skew_generator_from_free_bits(7, [1, -1, 1])
    # a[0]=1; a[1]=1, a[6]=-1; a[2]=-1, a[5]=1; a[3]=1, a[4]=-1
    assert gen[0] == 1
    assert gen[1] == -gen[6]
    assert gen[2] == -gen[5]
    assert gen[3] == -gen[4]
    M = skew_circulant(gen)
    assert M.n == 7
    v = verify(M.entries)
    assert v.valid, v.error
    # Confirm A + A^T = 2 I by direct check.
    n = 7
    for i in range(n):
        for j in range(n):
            s = M.entries[i][j] + M.entries[j][i]
            if i == j:
                assert s == 2
            else:
                assert s == 0


def test_skew_circulant_rejects_even_n():
    with pytest.raises(ValueError):
        skew_circulant([1, -1, 1, -1])  # n = 4, even


def test_skew_circulant_rejects_bad_diagonal():
    # n = 3; need a[0]=+1 and a[1]=-a[2].
    with pytest.raises(ValueError):
        skew_circulant([-1, 1, -1])  # a[0] = -1


def test_skew_circulant_rejects_non_skew():
    with pytest.raises(ValueError):
        skew_circulant([1, 1, 1])  # a[1]=1 but a[2]=1, need a[1]=-a[2]


# ---------------------------------------------------------------------------
# (b) Two-circulant SA reaches the Hadamard bound at n = 8.
# ---------------------------------------------------------------------------


def test_sa_two_circulant_reaches_h8_hadamard_bound():
    # n = 8: exhaustive search over 2^8 = 256 (a, b) pairs shows
    # the two-circulant block form reaches |det| = 4096 = 8^4.
    # SA should find this trivially.
    r = sa_two_circulant(8, iterations=10_000, seed=0)
    assert r.abs_det == 8 ** 4, (
        f"expected |det| = 4096 at n=8 two-circulant, got {r.abs_det}"
    )


# ---------------------------------------------------------------------------
# (c) Bordered-circulant SA reaches the ansatz maximum at n = 11.
# ---------------------------------------------------------------------------


def test_sa_bordered_circulant_reaches_ansatz_max_at_n11():
    # Exhaustive enumeration over 2^10 = 1024 bordered-circulant
    # cores at n = 11 shows the ansatz maximum is 281,600 (NOT the
    # H11 record 327,680, which requires a non-bordered-circulant
    # construction). SA should reach this ansatz ceiling with ease.
    r = sa_bordered_circulant(11, iterations=30_000, seed=0)
    assert r.abs_det == 281_600, (
        f"expected bordered-circulant n=11 max = 281600, got {r.abs_det}"
    )


def test_sa_bordered_circulant_reaches_h7_maximum():
    # At n = 7 the bordered-circulant maximum equals the true H7 max 576.
    r = sa_bordered_circulant(7, iterations=5_000, seed=0)
    assert r.abs_det == 576


# ---------------------------------------------------------------------------
# (d) Skew-circulant SA reaches its ansatz maximum at n = 7.
# ---------------------------------------------------------------------------


def test_sa_skew_circulant_n7_positive_det():
    # n = 7 skew-circulants: 3 free bits = 8 configurations; exhaustive.
    import itertools
    best = 0
    for bits in itertools.product([-1, 1], repeat=3):
        gen = skew_generator_from_free_bits(7, list(bits))
        from hadamard_maxdet.matrix import Matrix
        d = abs(integer_determinant(Matrix.circulant(gen).entries))
        if d > best:
            best = d
    r = sa_skew_circulant(7, iterations=2_000, seed=0)
    assert r.abs_det == best


# ---------------------------------------------------------------------------
# Sanity: SA-returned matrices are in-family (still ±1 and verify).
# ---------------------------------------------------------------------------


def test_sa_two_circulant_output_verifies():
    r = sa_two_circulant(8, iterations=1_000, seed=7)
    v = verify(r.matrix.entries)
    assert v.valid
    assert v.abs_det == r.abs_det


def test_sa_bordered_circulant_output_verifies():
    r = sa_bordered_circulant(11, iterations=1_000, seed=7)
    v = verify(r.matrix.entries)
    assert v.valid
    assert v.abs_det == r.abs_det


def test_sa_skew_circulant_output_verifies():
    r = sa_skew_circulant(9, iterations=1_000, seed=7)
    v = verify(r.matrix.entries)
    assert v.valid
    assert v.abs_det == r.abs_det
