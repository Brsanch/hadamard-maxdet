"""Sanity tests for crossover module."""
from __future__ import annotations

import random

from hadamard_maxdet.crossover import (
    align_rows_to_match,
    crossover_children,
    row_recombine,
)
from hadamard_maxdet.verifier import integer_determinant


def _rand(n: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.choice([-1, 1]) for _ in range(n)] for _ in range(n)]


def test_align_to_self_is_identity_up_to_sign():
    """Aligning P to P (same matrix) reproduces P (perm = identity, signs = +1)."""
    M = _rand(7, 1)
    aligned, perm, signs = align_rows_to_match(M, M)
    assert aligned == M
    assert perm == list(range(7))
    assert signs == [1] * 7


def test_align_to_negated_self_signs():
    """Aligning P to (-P) should give back P with all signs = -1."""
    M = _rand(6, 2)
    Mn = [[-v for v in row] for row in M]
    aligned, perm, signs = align_rows_to_match(M, Mn)
    assert aligned == M
    assert signs == [-1] * 6


def test_recombine_with_full_mask_a():
    """mask all True returns parent_a."""
    A = _rand(7, 3)
    B = _rand(7, 4)
    child = row_recombine(A, B, mask=[True] * 7, align=False)
    assert child == A


def test_recombine_with_empty_mask_returns_b_aligned_or_b():
    """mask all False returns b (or b_aligned)."""
    A = _rand(7, 5)
    B = _rand(7, 6)
    child_no_align = row_recombine(A, B, mask=[False] * 7, align=False)
    assert child_no_align == [list(r) for r in B]


def test_crossover_children_returns_valid_pm1():
    """Children entries are ±1; |det| matches Bareiss."""
    A = _rand(8, 7)
    B = _rand(8, 8)
    out = crossover_children(A, B, n_children=8, seed=42)
    for ch, d in out:
        assert d == abs(integer_determinant(ch))
        for row in ch:
            for v in row:
                assert v in (-1, 1)
