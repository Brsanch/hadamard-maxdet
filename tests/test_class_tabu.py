"""Tests for class-tabu (basin-exclusion) SA."""
from __future__ import annotations

import random

import numpy as np

from hadamard_maxdet.class_tabu import (
    bridge_from_seed,
    class_signature,
    iterative_basin_discovery,
    sa_with_class_tabu,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


def _random_matrix(n: int, seed: int) -> Matrix:
    rng = random.Random(seed)
    return Matrix.random(n, rng)


def test_class_signature_format():
    """Signature is a sorted tuple of (value, count) pairs."""
    M = [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
    sig = class_signature(M)
    assert isinstance(sig, tuple)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in sig)
    # Sorted by value.
    vals = [v for v, _ in sig]
    assert vals == sorted(vals)
    # Counts sum to n*(n-1)/2.
    n = len(M)
    assert sum(c for _, c in sig) == n * (n - 1) // 2


def test_class_signature_permutation_invariant():
    """Row and column permutations preserve the class signature."""
    n = 8
    rng = random.Random(42)
    M = [[1 if rng.random() < 0.5 else -1 for _ in range(n)] for _ in range(n)]
    sig0 = class_signature(M)

    # Permute rows.
    perm = list(range(n))
    rng.shuffle(perm)
    M_rp = [M[p] for p in perm]
    assert class_signature(M_rp) == sig0

    # Permute columns.
    cperm = list(range(n))
    rng.shuffle(cperm)
    M_cp = [[row[c] for c in cperm] for row in M]
    assert class_signature(M_cp) == sig0

    # Both permutations.
    M_both = [[M[p][c] for c in cperm] for p in perm]
    assert class_signature(M_both) == sig0


def test_class_signature_invariant_under_double_negation():
    """Negating row i AND column i preserves the Gram off-diagonal multiset
    exactly. (M -> D M D' with D, D' = ±I diagonal: G -> D G D, which only
    changes signs on row/column i — and the multiset of the strict upper
    triangle is preserved iff the signs balance, which they do under the
    double-negation transform M -> -M, G -> G.)
    """
    n = 6
    rng = random.Random(7)
    M = [[1 if rng.random() < 0.5 else -1 for _ in range(n)] for _ in range(n)]
    sig0 = class_signature(M)
    # Negate the entire matrix: G unchanged.
    M_neg = [[-x for x in row] for row in M]
    assert class_signature(M_neg) == sig0


def test_sa_with_class_tabu_rejects_locked_class():
    """If we lock the starting matrix's signature, the SA reports
    rejected_by_class_tabu > 0 (every move that returns to the start's class
    is rejected) and the final signature differs from the locked one (SA
    must end up somewhere else, including possibly a different class).
    """
    M = _random_matrix(7, seed=11)
    sig_start = class_signature(M.entries)

    res = sa_with_class_tabu(
        M,
        iterations=2000,
        locked_signatures=[sig_start],
        t_start=1.0,
        t_end=0.01,
        seed=3,
    )
    # The final class is not the locked one.
    assert res.final_signature != sig_start
    # At least some proposals were rejected by class-tabu.
    assert res.rejected_by_class_tabu > 0
    # |det| matches Bareiss recompute.
    assert res.abs_det == abs(integer_determinant(res.matrix.entries))


def test_sa_with_class_tabu_runs_without_lockout():
    """With an empty lockout set, SA should behave like standard SA and
    produce a valid result."""
    M = _random_matrix(7, seed=21)
    res = sa_with_class_tabu(
        M,
        iterations=2000,
        locked_signatures=[],
        seed=4,
    )
    assert res.rejected_by_class_tabu == 0
    assert res.abs_det == abs(integer_determinant(res.matrix.entries))
    assert res.matrix.n == 7


def test_iterative_basin_discovery_finds_multiple_basins_at_n21():
    """At n=21, the BTC basin (4-valued Gram support {-3,-1,1,3} at 51.2T)
    is dominant. With class-tabu, repeated SA should find at least 2
    distinct basins within a reasonable phase budget.
    """
    n = 21
    result = iterative_basin_discovery(
        n,
        max_phases=8,
        iters_per_phase=20_000,
        base_seed=2026,
        patience=8,
    )
    assert len(result.basins) >= 2, (
        f"expected >= 2 basins at n={n}, got {len(result.basins)}: "
        f"{[b.signature for b in result.basins]}"
    )
    # Every basin's |det| must match the Bareiss recompute.
    for b in result.basins:
        assert b.abs_det == abs(integer_determinant(b.matrix.entries))
    # Distinct signatures.
    sigs = {b.signature for b in result.basins}
    assert len(sigs) == len(result.basins)


def test_bridge_from_seed_returns_valid_matrix():
    """bridge_from_seed runs unlocked SA and returns an exact-Bareiss
    verified matrix."""
    M = _random_matrix(9, seed=5)
    res = bridge_from_seed(M, iterations=2000, seed=6)
    assert res.final_abs_det == abs(integer_determinant(res.final_matrix.entries))
    # Either drifted or didn't; both are legitimate outcomes for a short run.
    assert isinstance(res.drifted, bool)
