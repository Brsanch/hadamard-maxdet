"""Sanity tests for n=21 block-cyclic ansatz."""

from __future__ import annotations

from hadamard_maxdet.n21_block_cyclic import (
    block_cyclic_circulant,
    block_cyclic_from_blocks,
    sa_block_cyclic_free,
)
from hadamard_maxdet.verifier import integer_determinant


def test_block_cyclic_shape_and_pm1():
    M = block_cyclic_circulant([1, -1, 1, 1, -1, -1, 1], [1] * 7, [-1] * 7)
    assert M.n == 21
    assert len(M.entries) == 21
    for r in M.entries:
        assert len(r) == 21
        for v in r:
            assert v in (-1, 1)


def test_block_layout():
    """Verify [[A,B,C],[C,A,B],[B,C,A]] layout."""
    A = [[1] * 7 for _ in range(7)]
    B = [[-1] * 7 for _ in range(7)]
    C = [[1 if (i + j) % 2 == 0 else -1 for j in range(7)] for i in range(7)]
    M = block_cyclic_from_blocks(A, B, C)
    # Row 0 starts with A's first row.
    assert M.entries[0][:7] == A[0]
    # Row 0 cols 7..13 are B's first row.
    assert M.entries[0][7:14] == B[0]
    # Row 0 cols 14..20 are C's first row.
    assert M.entries[0][14:21] == C[0]
    # Row 7 (start of second block-row) starts with C, then A, then B.
    assert M.entries[7][:7] == C[0]
    assert M.entries[7][7:14] == A[0]
    assert M.entries[7][14:21] == B[0]
    # Row 14 starts with B, then C, then A.
    assert M.entries[14][:7] == B[0]
    assert M.entries[14][7:14] == C[0]
    assert M.entries[14][14:21] == A[0]


def test_sa_block_cyclic_free_runs():
    best, blocks, tele = sa_block_cyclic_free(iterations=200, seed=0)
    assert best > 0
    assert tele["iterations"] == 200
    A, B, C = blocks
    M = block_cyclic_from_blocks(A, B, C)
    # Re-verify
    assert abs(integer_determinant(M.entries)) == best
