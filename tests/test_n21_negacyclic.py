"""Sanity tests for n=21 negacyclic ansatz."""

from __future__ import annotations

from hadamard_maxdet.n21_negacyclic import (
    bordered_negacyclic_two,
    negacirculant,
    sa_bordered_negacyclic_two,
)
from hadamard_maxdet.verifier import integer_determinant


def test_negacirculant_2x2():
    """Negacirculant a = [1, -1] should be:
        N[0][0] = a[0] = 1
        N[0][1] = a[1] = -1   (no wrap)
        N[1][0] = a[-1 mod 2] = a[1], but -1 wraps -> sign flip -> -a[1] = 1
        N[1][1] = a[0] = 1
    """
    N = negacirculant([1, -1])
    assert N == [[1, -1], [1, 1]]


def test_negacirculant_pm1():
    N = negacirculant([1, 1, -1, 1, -1])
    assert len(N) == 5
    for row in N:
        for v in row:
            assert v in (-1, 1)


def test_bordered_negacyclic_two_shape_n21():
    M = bordered_negacyclic_two([1, -1, 1, 1, -1, -1, 1, -1, 1, -1], [1] * 10)
    assert M.n == 21
    for row in M.entries:
        assert len(row) == 21
        for v in row:
            assert v in (-1, 1)


def test_sa_bordered_negacyclic_two_runs():
    best, tele, M = sa_bordered_negacyclic_two(n=21, iterations=200, seed=0)
    assert best > 0
    assert M.n == 21
    assert abs(integer_determinant(M.entries)) == best
