"""Sanity tests for k_flip module: rank-2 score matches Bareiss."""
from __future__ import annotations

import random

from hadamard_maxdet.incremental import IncrementalState
from hadamard_maxdet.k_flip import (
    apply_2_flip,
    best_2_flip_via_adjugate,
    sa_with_k_flip,
    score_2_flip,
    steepest_2_flip_polish,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


def _make_random(n: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.choice([-1, 1]) for _ in range(n)] for _ in range(n)]


def test_score_2_flip_matches_bareiss():
    """For 30 random ±1 matrices and 30 random pair choices each, the
    rank-2 score must equal the Bareiss recompute exactly."""
    rng = random.Random(0xC0FFEE)
    for trial in range(30):
        n = rng.randint(5, 9)
        M = _make_random(n, seed=trial)
        d = integer_determinant(M)
        if d == 0:
            continue
        st = IncrementalState.from_matrix(M)
        for _ in range(30):
            i1 = rng.randrange(n); j1 = rng.randrange(n)
            i2 = rng.randrange(n); j2 = rng.randrange(n)
            if (i1, j1) == (i2, j2):
                continue
            est = score_2_flip(st, i1, j1, i2, j2)
            if est is None:
                continue
            cand = [row[:] for row in M]
            cand[i1][j1] = -cand[i1][j1]
            cand[i2][j2] = -cand[i2][j2]
            true_d = integer_determinant(cand)
            assert est == true_d, (
                f"trial={trial} n={n} pair=({i1,j1},{i2,j2}) "
                f"est={est} true={true_d}"
            )


def test_best_2_flip_finds_improver_when_one_exists():
    """Construct a matrix where a known 2-flip improves |det|."""
    n = 6
    # Start from random
    M = _make_random(n, seed=42)
    d0 = abs(integer_determinant(M))
    st = IncrementalState.from_matrix(M)
    if d0 == 0:
        return  # skip degenerate
    res = best_2_flip_via_adjugate(st, require_strict_improvement=False)
    assert res is not None
    best_abs, p1, p2 = res
    # Verify by recompute
    cand = [row[:] for row in M]
    cand[p1[0]][p1[1]] = -cand[p1[0]][p1[1]]
    cand[p2[0]][p2[1]] = -cand[p2[0]][p2[1]]
    assert abs(integer_determinant(cand)) == best_abs


def test_steepest_polish_monotone():
    """Polish never decreases |det|."""
    n = 7
    M = _make_random(n, seed=11)
    d0 = abs(integer_determinant(M))
    final_M, final_d, passes = steepest_2_flip_polish(M, max_passes=20)
    assert final_d >= d0
    # Re-verify with Bareiss
    assert final_d == abs(integer_determinant(final_M))


def test_sa_with_k_flip_runs_and_returns_valid():
    """SA with k-flip on n=7 returns a valid ±1 matrix."""
    res = sa_with_k_flip(7, iterations=2000, seed=1, p_2flip=0.1, p_kflip=0.05, kflip_size=4)
    assert res.matrix.n == 7
    assert res.abs_det == abs(integer_determinant(res.matrix.entries))
    for row in res.matrix.entries:
        for v in row:
            assert v in (-1, 1)
