"""Sanity tests for n=21 GS-array module."""

from __future__ import annotations

from hadamard_maxdet.n21_gs_array import bordered_gs, gs_array, sa_bordered_gs
from hadamard_maxdet.verifier import integer_determinant


def test_gs_array_shape():
    M = gs_array([1, 1, 1, 1, 1], [1] * 5, [1] * 5, [1] * 5)
    assert M.n == 20
    for row in M.entries:
        assert len(row) == 20
        for v in row:
            assert v in (-1, 1)


def test_gs_array_at_m1():
    """m=1 GS array is a 4x4 matrix; check its structure for trivial generators."""
    M = gs_array([1], [1], [1], [1])
    assert M.n == 4
    # All blocks are 1×1; the layout matches the GS pattern
    expected = [
        [1, 1, 1, 1],
        [-1, 1, 1, -1],
        [-1, -1, 1, 1],
        [-1, 1, -1, 1],
    ]
    assert M.entries == expected


def test_bordered_gs_shape():
    M = bordered_gs([1, -1, 1, 1, -1], [1, 1, -1, 1, 1], [1, 1, 1, -1, 1], [-1, 1, 1, 1, 1])
    assert M.n == 21
    for row in M.entries:
        assert len(row) == 21
        for v in row:
            assert v in (-1, 1)


def test_sa_bordered_gs_runs():
    best, tele, M = sa_bordered_gs(m=5, iterations=200, seed=0)
    assert best > 0
    assert M.n == 21
    # Re-verify
    assert abs(integer_determinant(M.entries)) == best
