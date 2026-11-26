"""Tests for `gram_2val`."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from hadamard_maxdet.gram_2val import (
    Gram2ValState,
    count_offdiag_value,
    count_outside_support,
    gram_offdiag_counter,
    sa_2valued,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


REPO_ROOT = Path(__file__).resolve().parents[1]
ORRICK_PATH = REPO_ROOT / "results" / "H23_orrick2003_record.json"


def _load_orrick() -> list[list[int]]:
    with open(ORRICK_PATH) as f:
        d = json.load(f)
    return [list(r) for r in d["entries"]]


# ---------------------------------------------------------------------------
# Recompute helpers
# ---------------------------------------------------------------------------


def test_count_outside_support_orrick_is_zero():
    M = _load_orrick()
    assert count_outside_support(M, {-1, 3}) == 0


def test_count_offdiag_value_orrick_pos3_is_45():
    M = _load_orrick()
    assert count_offdiag_value(M, 3) == 45


def test_count_offdiag_value_orrick_neg1_is_208():
    M = _load_orrick()
    assert count_offdiag_value(M, -1) == 253 - 45


def test_count_outside_support_random_matrix():
    rng = random.Random(0)
    n = 8
    M = Matrix.random(n, rng).entries
    # All Gram off-diagonals are even (n even). Span all even values in [-n, n].
    even_support = set(range(-n, n + 1, 2))
    out = count_outside_support(M, even_support)
    assert out == 0


def test_count_offdiag_value_handles_no_match():
    # Identity-like: every off-diag is 0 if rows are orthogonal; here they
    # are not, but we just verify the function runs and returns 0 for an
    # absent value.
    rng = random.Random(1)
    M = Matrix.random(5, rng).entries
    # 99 cannot appear in a 5x5 ±1 Gram off-diagonal (max |G_ij| = 5).
    assert count_offdiag_value(M, 99) == 0


def test_gram_offdiag_counter_sums_to_total_pairs():
    rng = random.Random(2)
    n = 7
    M = Matrix.random(n, rng).entries
    cnt = gram_offdiag_counter(M)
    assert sum(cnt.values()) == n * (n - 1) // 2


# ---------------------------------------------------------------------------
# Gram2ValState bootstrap
# ---------------------------------------------------------------------------


def test_gram2valstate_orrick_counts_match_recompute():
    M = _load_orrick()
    s = Gram2ValState.from_matrix(M, pos_value=3, neg_value=-1)
    assert s.count_pos == 45
    assert s.count_neg == 253 - 45
    assert s.count_out == 0


def test_gram2valstate_random_invariants():
    rng = random.Random(3)
    n = 9
    M = Matrix.random(n, rng).entries
    s = Gram2ValState.from_matrix(M, pos_value=3, neg_value=-1)
    total = n * (n - 1) // 2
    assert s.count_pos + s.count_neg + s.count_out == total
    # Cross-check against the recompute helpers.
    assert s.count_pos == count_offdiag_value(M, 3)
    assert s.count_neg == count_offdiag_value(M, -1)
    assert s.count_out == count_outside_support(M, {-1, 3})


# ---------------------------------------------------------------------------
# Incremental update vs full recompute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_incremental_update_matches_full_recompute_small_n(seed):
    rng = random.Random(seed)
    n = 7
    M = Matrix.random(n, rng).entries
    s = Gram2ValState.from_matrix(M, pos_value=3, neg_value=-1)

    for step in range(40):
        i = rng.randrange(n)
        c = rng.randrange(n)
        # Apply flip on M, then update state, then compare to recompute.
        M[i][c] = -M[i][c]
        d_pos, d_neg, d_out = s.deltas_for_flip(M, i, c)
        s.apply_after_flip(M, i, c)
        # Cross-check against full recompute.
        assert s.count_pos == count_offdiag_value(M, 3), f"step={step} pos"
        assert s.count_neg == count_offdiag_value(M, -1), f"step={step} neg"
        assert s.count_out == count_outside_support(M, {-1, 3}), f"step={step} out"
        # Cross-check Gram.
        Gfull = (np.asarray(M, dtype=int) @ np.asarray(M, dtype=int).T)
        assert np.array_equal(s.G, Gfull), f"step={step} G drift"
        # Sanity on deltas.
        assert d_pos + d_neg + d_out == 0


@pytest.mark.parametrize("n", [5, 8, 11])
def test_incremental_update_matches_full_recompute_other_sizes(n):
    rng = random.Random(100 + n)
    M = Matrix.random(n, rng).entries
    s = Gram2ValState.from_matrix(M, pos_value=3, neg_value=-1)
    for _ in range(20):
        i = rng.randrange(n)
        c = rng.randrange(n)
        M[i][c] = -M[i][c]
        s.apply_after_flip(M, i, c)
    assert s.count_pos == count_offdiag_value(M, 3)
    assert s.count_neg == count_offdiag_value(M, -1)
    assert s.count_out == count_outside_support(M, {-1, 3})


# ---------------------------------------------------------------------------
# SA from Orrick seed at k=45 should stay feasible (energy doesn't push it out)
# ---------------------------------------------------------------------------


def test_sa_from_orrick_at_k45_stays_feasible_short():
    """Run a tiny SA from the Orrick seed at the matching k=45.

    With strong support penalty and the count penalty zero at the seed, SA
    should remain feasible-ever and the best feasible |det| should be at
    least the Orrick |det|.
    """
    M = _load_orrick()
    initial = Matrix(n=23, entries=[row[:] for row in M])
    res = sa_2valued(
        n=23,
        target_k_pos=45,
        iterations=2_000,           # short for test speed
        t_start=0.05, t_end=0.005,  # cool fast: don't wander far
        mu_support=10.0,
        lambda_count=0.5,
        seed=12345,
        initial=initial,
        refresh_every=500,
    )
    orrick_det = abs(integer_determinant(M))
    assert res.feasible_ever
    # The Orrick seed was itself feasible at start, so best_feasible >= seed.
    assert res.best_feasible_abs_det >= orrick_det


def test_sa_random_seed_short_run_returns_valid_matrix():
    """Smoke test: a random-seed short run returns a verified ±1 matrix."""
    res = sa_2valued(
        n=11,
        target_k_pos=10,
        iterations=2_000,
        t_start=0.5, t_end=0.01,
        mu_support=2.0, lambda_count=0.1,
        seed=7,
        refresh_every=500,
    )
    assert res.matrix.n == 11
    for row in res.matrix.entries:
        assert all(v in (-1, 1) for v in row)
    assert res.abs_det == abs(integer_determinant(res.matrix.entries))


# ---------------------------------------------------------------------------
# Edge-case sanity
# ---------------------------------------------------------------------------


def test_count_offdiag_value_n_equals_2():
    M = [[1, 1], [1, -1]]
    # Gram = [[2, 0], [0, 2]]; one unordered off-diag = 0.
    assert count_offdiag_value(M, 0) == 1
    assert count_offdiag_value(M, 2) == 0


def test_count_outside_support_empty_support():
    rng = random.Random(0)
    M = Matrix.random(5, rng).entries
    assert count_outside_support(M, set()) == 5 * 4 // 2  # all pairs are "outside"
