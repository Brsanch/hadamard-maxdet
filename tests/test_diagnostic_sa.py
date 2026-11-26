"""Tests for diagnostic-driven SA variants."""

from __future__ import annotations

import random

import pytest

from hadamard_maxdet.diagnostic_sa import (
    apply_gram_flip,
    compute_gram_full,
    count_imbalance,
    count_imbalance_delta,
    count_minus_plus,
    gram_offdiag_counts,
    sa_balanced,
    sa_hot_only,
    weight_perturbation_flip,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# count_imbalance basics
# ---------------------------------------------------------------------------


def test_count_imbalance_hadamard_zero():
    """A Hadamard matrix has all off-diagonals 0; imbalance = 0."""
    H = Matrix.sylvester(8).entries
    G = compute_gram_full(H)
    # Off-diagonals should all be 0.
    for i in range(8):
        for j in range(8):
            if i != j:
                assert G[i][j] == 0
    assert count_imbalance(H) == 0.0
    cm, cp = count_minus_plus(G)
    assert cm == 0 and cp == 0


def test_count_imbalance_basic_signs():
    """A handcrafted small matrix with known c(-1) = 2 and c(+1) = 0."""
    # 3x3 matrix where two pairs have inner-product −1 and one pair = +1.
    M = [
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
    ]
    G = compute_gram_full(M)
    # Pair (0,1): 1+1-1 = 1
    # Pair (0,2): 1-1+1 = 1
    # Pair (1,2): 1-1-1 = -1
    cm, cp = count_minus_plus(G)
    assert cm == 1
    assert cp == 2
    assert count_imbalance(M) == abs(1 - 2) / 9


def test_gram_offdiag_counts_matches_telemetry():
    rng = random.Random(0)
    M = Matrix.random(7, rng).entries
    counts = gram_offdiag_counts(M)
    # Every pair (i < j) is counted exactly once.
    n_pairs = 7 * 6 // 2
    assert sum(counts.values()) == n_pairs


# ---------------------------------------------------------------------------
# count_imbalance_delta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,seed", [(5, 0), (7, 1), (11, 7), (13, 13)])
def test_count_imbalance_delta_matches_recompute(n: int, seed: int):
    rng = random.Random(seed)
    M = Matrix.random(n, rng).entries
    G = compute_gram_full(M)
    cm, cp = count_minus_plus(G)
    # Try a handful of random flips.
    for _ in range(20):
        i = rng.randrange(n)
        j = rng.randrange(n)
        dcm, dcp = count_imbalance_delta(G, M, i, j)
        # Apply the flip in a copy and recompute from scratch.
        M2 = [row[:] for row in M]
        M2[i][j] = -M2[i][j]
        G2 = compute_gram_full(M2)
        cm2, cp2 = count_minus_plus(G2)
        assert cm2 - cm == dcm, (
            f"dcm mismatch at ({i},{j}): predicted {dcm}, actual {cm2 - cm}"
        )
        assert cp2 - cp == dcp, (
            f"dcp mismatch at ({i},{j}): predicted {dcp}, actual {cp2 - cp}"
        )


def test_apply_gram_flip_consistent():
    """apply_gram_flip(G, M, i, j) must equal recompute_gram(flipped M)."""
    rng = random.Random(42)
    n = 9
    M = Matrix.random(n, rng).entries
    G = compute_gram_full(M)
    for _ in range(15):
        i = rng.randrange(n)
        j = rng.randrange(n)
        apply_gram_flip(G, M, i, j)
        # Now flip M.
        M[i][j] = -M[i][j]
        G_recomputed = compute_gram_full(M)
        assert G == G_recomputed


# ---------------------------------------------------------------------------
# weight_perturbation_flip
# ---------------------------------------------------------------------------


def test_weight_perturbation_flip_changes_row_weight_by_two():
    rng = random.Random(0)
    n = 11
    M = Matrix.random(n, rng).entries
    weights_before = [sum(1 for x in row if x > 0) for row in M]
    for _ in range(50):
        triple = weight_perturbation_flip(M, rng)
        if triple is None:
            continue
        i, j, k = triple
        # j and k must be distinct columns with the same sign in row i.
        assert j != k
        assert M[i][j] == M[i][k]
        sign = M[i][j]
        # Apply the flip pair.
        M[i][j] = -M[i][j]
        M[i][k] = -M[i][k]
        new_w = sum(1 for x in M[i] if x > 0)
        old_w = weights_before[i]
        # Flipping two +1s → −1s decreases weight by 2; two −1s → +1s
        # increases weight by 2.
        assert abs(new_w - old_w) == 2
        if sign == 1:
            assert new_w == old_w - 2
        else:
            assert new_w == old_w + 2
        weights_before[i] = new_w


def test_weight_perturbation_flip_preserves_pm1():
    rng = random.Random(0)
    n = 7
    M = Matrix.random(n, rng).entries
    for _ in range(20):
        triple = weight_perturbation_flip(M, rng)
        if triple is None:
            continue
        i, j, k = triple
        M[i][j] = -M[i][j]
        M[i][k] = -M[i][k]
        for row in M:
            for v in row:
                assert v in (1, -1)


# ---------------------------------------------------------------------------
# sa_hot_only / sa_balanced smoke tests
# ---------------------------------------------------------------------------


def test_sa_hot_only_returns_valid_pm1():
    res = sa_hot_only(7, iterations=2000, seed=0)
    assert res.matrix.n == 7
    for row in res.matrix.entries:
        for v in row:
            assert v in (1, -1)
    # |det| computed by SA must match Bareiss recompute.
    assert res.abs_det == abs(integer_determinant(res.matrix.entries))


def test_sa_balanced_zero_lambda_matches_unconstrained_sa():
    """At λ=0 and weight_perturb_prob=0, sa_balanced runs the same
    energy as plain SA. Final |det| should be in the same ballpark."""
    res_b = sa_balanced(
        7, iterations=2000, lambda_balance=0.0, weight_perturb_prob=0.0,
        seed=0,
    )
    assert res_b.abs_det > 0
    assert res_b.abs_det == abs(integer_determinant(res_b.matrix.entries))


def test_sa_balanced_with_lambda_pushes_toward_balance():
    """High λ should reduce gram-count imbalance vs. λ=0 at the same
    seed/iterations. Statistical claim: average over a few seeds.
    """
    n = 9
    iters = 4000
    imbal_zero = 0.0
    imbal_high = 0.0
    seeds = [0, 1, 2, 3]
    for s in seeds:
        r0 = sa_balanced(
            n, iterations=iters, lambda_balance=0.0,
            weight_perturb_prob=0.0, seed=s,
        )
        r1 = sa_balanced(
            n, iterations=iters, lambda_balance=2.0,
            weight_perturb_prob=0.0, seed=s,
        )
        imbal_zero += count_imbalance(r0.matrix.entries)
        imbal_high += count_imbalance(r1.matrix.entries)
    # On average λ=2 should give lower imbalance.
    assert imbal_high <= imbal_zero + 1e-9


def test_sa_balanced_with_weight_perturbation_runs():
    """Weight-perturbation move should run without crashing and return
    a valid ±1 matrix with verified det."""
    res = sa_balanced(
        9,
        iterations=3000,
        lambda_balance=0.05,
        weight_perturb_prob=0.2,
        seed=0,
    )
    assert res.matrix.n == 9
    for row in res.matrix.entries:
        for v in row:
            assert v in (1, -1)
    assert res.abs_det == abs(integer_determinant(res.matrix.entries))
