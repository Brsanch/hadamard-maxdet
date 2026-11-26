"""Tests for ecology-domain Competitive Exclusion Niche SA."""

from __future__ import annotations

import copy
import random

import pytest

from hadamard_maxdet.niche_sa import (
    ORRICK_ABS_DET_H23,
    NicheSAResult,
    gram_fingerprint,
    niche_sa_search,
    triple_verify,
)
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_pm1(M: list[list[int]]) -> bool:
    return all(v in (-1, 1) for row in M for v in row)


def _is_square(M: list[list[int]], n: int) -> bool:
    return len(M) == n and all(len(row) == n for row in M)


def _random_pm1(n: int, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[1 if rng.random() < 0.5 else -1 for _ in range(n)] for _ in range(n)]


# ---------------------------------------------------------------------------
# gram_fingerprint
# ---------------------------------------------------------------------------


def test_gram_fingerprint_length_n5():
    M = _random_pm1(5, seed=1)
    fp = gram_fingerprint(M)
    # C(5,2) = 10 row pairs
    assert len(fp) == 10


def test_gram_fingerprint_length_n23():
    M = _random_pm1(23, seed=2)
    fp = gram_fingerprint(M)
    # C(23,2) = 253 row pairs
    assert len(fp) == 253


def test_gram_fingerprint_is_sorted():
    M = _random_pm1(7, seed=3)
    fp = gram_fingerprint(M)
    assert fp == tuple(sorted(fp))


def test_gram_fingerprint_values_in_range_n5():
    M = _random_pm1(5, seed=4)
    fp = gram_fingerprint(M)
    # For n=5 ±1, inner products are odd in [-5, 5]
    for v in fp:
        assert -5 <= v <= 5
        assert v % 2 != 0


def test_gram_fingerprint_deterministic():
    M = _random_pm1(7, seed=5)
    fp1 = gram_fingerprint(M)
    fp2 = gram_fingerprint(M)
    assert fp1 == fp2


def test_gram_fingerprint_row_permutation_changes_fingerprint():
    # Sorting the fingerprint makes it invariant to row permutation — verify
    M = _random_pm1(5, seed=6)
    M2 = [M[2], M[0], M[1], M[3], M[4]]  # row permutation
    fp1 = gram_fingerprint(M)
    fp2 = gram_fingerprint(M2)
    assert fp1 == fp2  # sorted fingerprint is invariant


def test_gram_fingerprint_different_matrices_different_or_same():
    M1 = _random_pm1(5, seed=7)
    M2 = _random_pm1(5, seed=8)
    fp1 = gram_fingerprint(M1)
    fp2 = gram_fingerprint(M2)
    # Can be equal by chance but usually are not; just verify they're tuples
    assert isinstance(fp1, tuple)
    assert isinstance(fp2, tuple)


# ---------------------------------------------------------------------------
# triple_verify
# ---------------------------------------------------------------------------


def test_triple_verify_consistency():
    M = _random_pm1(5, seed=10)
    tv = triple_verify(M)
    assert tv["consistent"]
    assert abs(tv["det"]) == abs(tv["det_transpose"]) == abs(tv["det_copy"])


def test_triple_verify_matches_integer_determinant():
    M = _random_pm1(7, seed=11)
    tv = triple_verify(M)
    d = integer_determinant(M)
    assert abs(tv["det"]) == abs(d)


def test_triple_verify_does_not_modify_input():
    M = _random_pm1(5, seed=12)
    M_copy = copy.deepcopy(M)
    triple_verify(M)
    assert M == M_copy


# ---------------------------------------------------------------------------
# niche_sa_search — small n smoke tests
# ---------------------------------------------------------------------------


def test_niche_sa_search_n5_returns_result():
    result = niche_sa_search(n=5, budget_seconds=3.0, sa_iters=2000, greedy_iters=0, seed=1)
    assert isinstance(result, NicheSAResult)


def test_niche_sa_search_n5_pm1_output():
    result = niche_sa_search(n=5, budget_seconds=3.0, sa_iters=2000, greedy_iters=0, seed=2)
    assert _all_pm1(result.best_matrix)


def test_niche_sa_search_n5_square_output():
    result = niche_sa_search(n=5, budget_seconds=3.0, sa_iters=2000, greedy_iters=0, seed=3)
    assert _is_square(result.best_matrix, 5)


def test_niche_sa_search_n5_positive_det():
    result = niche_sa_search(n=5, budget_seconds=3.0, sa_iters=2000, greedy_iters=0, seed=4)
    assert result.best_abs_det > 0


def test_niche_sa_search_n5_det_consistent():
    result = niche_sa_search(n=5, budget_seconds=3.0, sa_iters=2000, greedy_iters=0, seed=5)
    d = abs(integer_determinant(result.best_matrix))
    assert d == result.best_abs_det


def test_niche_sa_search_n5_community_nonempty():
    result = niche_sa_search(n=5, budget_seconds=3.0, sa_iters=2000, greedy_iters=0, seed=6)
    assert result.community_size >= 1


def test_niche_sa_search_n5_chain_counts():
    result = niche_sa_search(n=5, budget_seconds=3.0, sa_iters=2000, greedy_iters=0, seed=7)
    assert result.total_chains == result.exploration_chains + result.exploitation_chains
    assert result.total_chains >= 1


def test_niche_sa_search_n5_niche_collision_sum():
    result = niche_sa_search(n=5, budget_seconds=3.0, sa_iters=2000, greedy_iters=0, seed=8)
    assert result.new_niches + result.collisions == result.total_chains


def test_niche_sa_search_n5_above_orrick_false():
    # n=5: Orrick H5 bound is much smaller than H23; this test checks the flag
    result = niche_sa_search(n=5, budget_seconds=3.0, sa_iters=2000, greedy_iters=0, seed=9)
    # above_orrick compares against H23 Orrick constant — for n=5 always False
    assert result.above_orrick is False


def test_niche_sa_search_n5_ratio_type():
    result = niche_sa_search(n=5, budget_seconds=3.0, sa_iters=2000, greedy_iters=0, seed=10)
    assert isinstance(result.ratio_to_orrick, float)


def test_niche_sa_search_n5_elapsed_positive():
    result = niche_sa_search(n=5, budget_seconds=3.0, sa_iters=2000, greedy_iters=0, seed=11)
    assert result.elapsed_seconds > 0


def test_niche_sa_search_n5_histogram_sums_to_community():
    result = niche_sa_search(n=5, budget_seconds=3.0, sa_iters=2000, greedy_iters=0, seed=12)
    assert sum(result.niche_det_histogram) == result.community_size


def test_niche_sa_search_n11_smoke():
    result = niche_sa_search(n=11, budget_seconds=5.0, sa_iters=5000, greedy_iters=0, seed=42)
    assert isinstance(result, NicheSAResult)
    assert _all_pm1(result.best_matrix)
    assert _is_square(result.best_matrix, 11)
    assert result.best_abs_det > 0


def test_niche_sa_search_n11_det_consistent():
    result = niche_sa_search(n=11, budget_seconds=5.0, sa_iters=5000, greedy_iters=0, seed=43)
    d = abs(integer_determinant(result.best_matrix))
    assert d == result.best_abs_det


def test_niche_sa_search_exploit_only():
    # With exploit_prob=0.0 all chains are exploration; community grows
    result = niche_sa_search(
        n=5, budget_seconds=3.0, sa_iters=2000, greedy_iters=0, exploit_prob=0.0, seed=13
    )
    assert result.exploitation_chains == 0
    assert result.exploration_chains == result.total_chains


def test_niche_sa_search_exploit_all():
    # With exploit_prob=1.0 exploitation dominates once community non-empty
    result = niche_sa_search(
        n=5, budget_seconds=4.0, sa_iters=2000, greedy_iters=0, exploit_prob=1.0, seed=14
    )
    # First chain must be exploration (community empty); rest are exploitation
    assert result.exploration_chains >= 1
    assert result.total_chains == result.exploration_chains + result.exploitation_chains
