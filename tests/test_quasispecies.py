"""Tests for quasispecies error-threshold search (evolutionary_biology domain)."""

from __future__ import annotations

import copy
import random

import pytest

from hadamard_maxdet.quasispecies import (
    ORRICK_ABS_DET_H23,
    QSResult,
    _mutate,
    _random_pm1_matrix,
    _weighted_sample_index,
    quasispecies_search,
    triple_verify,
)
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_pm1(matrix: list[list[int]]) -> bool:
    return all(v in (-1, 1) for row in matrix for v in row)


def _is_square(matrix: list[list[int]], n: int) -> bool:
    return len(matrix) == n and all(len(row) == n for row in matrix)


# ---------------------------------------------------------------------------
# Random matrix tests
# ---------------------------------------------------------------------------


def test_random_pm1_matrix_shape():
    rng = random.Random(1)
    m = _random_pm1_matrix(5, rng)
    assert _is_square(m, 5)


def test_random_pm1_matrix_entries():
    rng = random.Random(2)
    m = _random_pm1_matrix(7, rng)
    assert _all_pm1(m)


# ---------------------------------------------------------------------------
# Mutation tests
# ---------------------------------------------------------------------------


def test_mutate_preserves_pm1():
    rng = random.Random(3)
    m = _random_pm1_matrix(7, rng)
    m2 = _mutate(m, 0.3, rng)
    assert _all_pm1(m2)
    assert _is_square(m2, 7)


def test_mutate_zero_rate_no_change():
    rng = random.Random(4)
    m = _random_pm1_matrix(5, rng)
    m2 = _mutate(m, 0.0, rng)
    assert m2 == m


def test_mutate_full_rate_all_flipped():
    rng = random.Random(5)
    m = [[1, -1], [-1, 1]]
    m2 = _mutate(m, 1.0, rng)
    assert m2 == [[-1, 1], [1, -1]]


def test_mutate_does_not_modify_original():
    rng = random.Random(6)
    m = [[1, 1], [1, 1]]
    original = copy.deepcopy(m)
    _mutate(m, 0.5, rng)
    assert m == original


# ---------------------------------------------------------------------------
# Weighted sampling tests
# ---------------------------------------------------------------------------


def test_weighted_sample_deterministic_single():
    """With one non-zero weight, must always return that index."""
    rng = random.Random(7)
    weights = [0.0, 5.0, 0.0]
    for _ in range(20):
        idx = _weighted_sample_index(weights, 5.0, rng)
        assert idx == 1


def test_weighted_sample_range():
    rng = random.Random(8)
    weights = [1.0, 2.0, 3.0]
    total = sum(weights)
    counts = [0, 0, 0]
    for _ in range(600):
        idx = _weighted_sample_index(weights, total, rng)
        assert 0 <= idx < 3
        counts[idx] += 1
    # Rough distribution check: index 2 should be most sampled
    assert counts[2] > counts[0]


# ---------------------------------------------------------------------------
# Triple verify tests
# ---------------------------------------------------------------------------


def test_triple_verify_consistent():
    rng = random.Random(9)
    m = _random_pm1_matrix(5, rng)
    d1, d2, d3 = triple_verify(m)
    assert d1 == d2 == d3


def test_triple_verify_matches_integer_determinant():
    rng = random.Random(10)
    m = _random_pm1_matrix(5, rng)
    d1, d2, d3 = triple_verify(m)
    expected = abs(integer_determinant(m))
    assert d1 == expected


def test_triple_verify_nonsingular_nonzero():
    m = [[1, 1], [1, -1]]
    d1, d2, d3 = triple_verify(m)
    assert d1 == d2 == d3 == 2


# ---------------------------------------------------------------------------
# Full run tests (small n)
# ---------------------------------------------------------------------------


def test_quasispecies_n5_runs():
    result = quasispecies_search(
        n=5,
        pop_size=5,
        mu_0=0.3,
        mu_final=0.01,
        alpha=2.0,
        restart_every=20,
        wall_time=2.0,
        orrick=48,
        seed=11,
        verbose=False,
    )
    assert result.best_abs_det > 0
    assert result.n_generations > 0
    assert _all_pm1(result.best_matrix)
    assert _is_square(result.best_matrix, 5)


def test_quasispecies_n5_pm1_output():
    result = quasispecies_search(
        n=5,
        pop_size=4,
        mu_0=0.2,
        mu_final=0.01,
        alpha=2.0,
        restart_every=15,
        wall_time=1.0,
        orrick=48,
        seed=12,
        verbose=False,
    )
    assert _all_pm1(result.best_matrix)


def test_quasispecies_n11_runs():
    result = quasispecies_search(
        n=11,
        pop_size=8,
        mu_0=0.2,
        mu_final=0.002,
        alpha=2.0,
        restart_every=30,
        wall_time=3.0,
        orrick=131072,
        seed=13,
        verbose=False,
    )
    assert result.best_abs_det > 0
    assert result.n_generations > 0
    assert _all_pm1(result.best_matrix)
    assert _is_square(result.best_matrix, 11)


def test_quasispecies_det_consistency():
    result = quasispecies_search(
        n=5,
        pop_size=5,
        mu_0=0.25,
        mu_final=0.01,
        alpha=2.0,
        restart_every=15,
        wall_time=1.0,
        orrick=48,
        seed=14,
        verbose=False,
    )
    expected = abs(integer_determinant(result.best_matrix))
    assert result.best_abs_det == expected


def test_quasispecies_ratio_type():
    result = quasispecies_search(
        n=5,
        pop_size=4,
        mu_0=0.2,
        mu_final=0.01,
        alpha=2.0,
        restart_every=10,
        wall_time=1.0,
        orrick=48,
        seed=15,
        verbose=False,
    )
    assert isinstance(result.best_ratio_to_orrick, float)


def test_quasispecies_above_orrick_flag():
    # Use a tiny Orrick threshold so we trigger the flag
    result = quasispecies_search(
        n=5,
        pop_size=5,
        mu_0=0.2,
        mu_final=0.01,
        alpha=2.0,
        restart_every=20,
        wall_time=1.0,
        orrick=1,  # trivially beaten
        seed=16,
        verbose=False,
    )
    assert result.above_orrick is True


def test_quasispecies_above_orrick_false():
    # Use H5 max det = 48; random matrices typically don't hit 48
    # Just check flag is a bool
    result = quasispecies_search(
        n=3,
        pop_size=4,
        mu_0=0.2,
        mu_final=0.01,
        alpha=2.0,
        restart_every=10,
        wall_time=0.5,
        orrick=2,  # H3 max = 4; may or may not exceed 2
        seed=17,
        verbose=False,
    )
    assert isinstance(result.above_orrick, bool)


def test_quasispecies_restarts_counted():
    result = quasispecies_search(
        n=5,
        pop_size=4,
        mu_0=0.2,
        mu_final=0.01,
        alpha=2.0,
        restart_every=5,   # very short — forces multiple restarts
        wall_time=2.0,
        orrick=48,
        seed=18,
        verbose=False,
    )
    assert result.n_restarts >= 1


def test_quasispecies_n5_triple_verify():
    result = quasispecies_search(
        n=5,
        pop_size=4,
        mu_0=0.2,
        mu_final=0.01,
        alpha=2.0,
        restart_every=15,
        wall_time=1.0,
        orrick=48,
        seed=19,
        verbose=False,
    )
    d1, d2, d3 = triple_verify(result.best_matrix)
    assert d1 == d2 == d3 == result.best_abs_det
