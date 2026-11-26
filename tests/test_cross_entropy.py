"""Tests for the Cross-Entropy Method module (cross_entropy.py)."""

from __future__ import annotations

import math
import random

import pytest

from hadamard_maxdet.cross_entropy import (
    ORRICK_DET_H23,
    CemResult,
    cem_sample,
    dist_entropy,
    run_cem,
    update_probs,
)


# ---------------------------------------------------------------------------
# cem_sample
# ---------------------------------------------------------------------------


def test_cem_sample_shape():
    p = [[0.5] * 5 for _ in range(5)]
    rng = random.Random(0)
    m = cem_sample(p, rng)
    assert len(m) == 5
    assert all(len(row) == 5 for row in m)


def test_cem_sample_entries_pm1():
    p = [[0.5] * 4 for _ in range(4)]
    rng = random.Random(1)
    for _ in range(20):
        m = cem_sample(p, rng)
        for row in m:
            for v in row:
                assert v in (-1, 1)


def test_cem_sample_biased_toward_plus1():
    n = 10
    p = [[0.9] * n for _ in range(n)]
    rng = random.Random(42)
    plus_count = 0
    total = 0
    for _ in range(50):
        m = cem_sample(p, rng)
        for row in m:
            for v in row:
                if v == 1:
                    plus_count += 1
                total += 1
    # Expect roughly 90% +1; 3-sigma window ≈ 0.90 ± 0.03 × sqrt(50*100) ≈ wide
    assert plus_count / total > 0.80, f"expected >80% +1, got {plus_count}/{total}"


def test_cem_sample_biased_toward_minus1():
    n = 8
    p = [[0.05] * n for _ in range(n)]
    rng = random.Random(7)
    plus_count = sum(
        v for _ in range(30) for row in cem_sample(p, rng) for v in row if v == 1
    )
    total = 30 * n * n
    assert plus_count / total < 0.25


# ---------------------------------------------------------------------------
# update_probs
# ---------------------------------------------------------------------------


def test_update_probs_all_plus1_elite():
    p = [[0.5] * 3 for _ in range(3)]
    elite = [[[1] * 3 for _ in range(3)]] * 4  # every entry is +1
    update_probs(p, elite, alpha=1.0, p_min=0.0)
    for row in p:
        for v in row:
            assert v == 1.0


def test_update_probs_all_minus1_elite():
    p = [[0.5] * 3 for _ in range(3)]
    elite = [[[-1] * 3 for _ in range(3)]] * 4
    update_probs(p, elite, alpha=1.0, p_min=0.0)
    for row in p:
        for v in row:
            assert v == 0.0


def test_update_probs_clamps_to_p_min():
    p = [[0.5] * 2 for _ in range(2)]
    elite = [[[-1, -1], [-1, -1]]]  # all -1
    update_probs(p, elite, alpha=1.0, p_min=0.15)
    for row in p:
        for v in row:
            assert v >= 0.15
            assert v <= 1.0 - 0.15


def test_update_probs_smoothing():
    p = [[0.5] * 2 for _ in range(2)]
    p_before = [row[:] for row in p]
    elite = [[[1, 1], [1, 1]]]  # all +1
    update_probs(p, elite, alpha=0.5, p_min=0.0)
    # Expected: (1-0.5)*0.5 + 0.5*1.0 = 0.75
    for row in p:
        for v in row:
            assert abs(v - 0.75) < 1e-9


def test_update_probs_empty_elite_no_change():
    p = [[0.5] * 3 for _ in range(3)]
    p_before = [row[:] for row in p]
    update_probs(p, [], alpha=0.7, p_min=0.1)
    for i, row in enumerate(p):
        for j, v in enumerate(row):
            assert v == p_before[i][j]


# ---------------------------------------------------------------------------
# dist_entropy
# ---------------------------------------------------------------------------


def test_dist_entropy_uniform_is_ln2():
    n = 5
    p = [[0.5] * n for _ in range(n)]
    h = dist_entropy(p)
    assert abs(h - math.log(2)) < 1e-9


def test_dist_entropy_fully_pinned_is_zero():
    n = 4
    p = [[1.0] * n for _ in range(n)]
    h = dist_entropy(p)
    assert h == 0.0


def test_dist_entropy_decreases_after_update():
    n = 5
    p = [[0.5] * n for _ in range(n)]
    h_before = dist_entropy(p)
    # Elite: half +1, half -1, but skewed
    elite = [[[1, -1, 1, -1, 1]] * n, [[1, 1, 1, 1, 1]] * n]
    update_probs(p, elite, alpha=0.9, p_min=0.0)
    h_after = dist_entropy(p)
    assert h_after < h_before


# ---------------------------------------------------------------------------
# ORRICK_DET_H23 constant
# ---------------------------------------------------------------------------


def test_orrick_det_constant():
    assert ORRICK_DET_H23 == 2_779_447_296_000_000


# ---------------------------------------------------------------------------
# run_cem integration (tiny n, very short run)
# ---------------------------------------------------------------------------


def test_run_cem_small_n():
    result = run_cem(
        n=5,
        n_init=2,
        n_samples=2,
        n_elite=1,
        n_generations=2,
        sa_iterations=500,
        alpha=0.7,
        p_min=0.1,
        wall_cap=30.0,
        rng_seed=0,
        verbose=False,
    )
    assert isinstance(result, CemResult)
    assert result.best_abs_det > 0
    assert result.n_sa_total >= 1
    assert result.walltime_seconds > 0
    assert 0.0 < result.final_entropy <= math.log(2) + 1e-9


def test_run_cem_entropy_non_increasing_overall():
    result = run_cem(
        n=5,
        n_init=3,
        n_samples=3,
        n_elite=2,
        n_generations=3,
        sa_iterations=200,
        alpha=0.8,
        p_min=0.05,
        wall_cap=30.0,
        rng_seed=99,
        verbose=False,
    )
    assert result.final_entropy <= result.initial_entropy + 1e-9


def test_run_cem_returns_pm1_matrix():
    result = run_cem(
        n=4,
        n_init=1,
        n_samples=1,
        n_elite=1,
        n_generations=1,
        sa_iterations=100,
        wall_cap=10.0,
        rng_seed=7,
        verbose=False,
    )
    M = result.best_matrix
    assert len(M) == 4
    for row in M:
        assert len(row) == 4
        for v in row:
            assert v in (-1, 1)


def test_run_cem_generation_log_populated():
    result = run_cem(
        n=4,
        n_init=2,
        n_samples=2,
        n_elite=1,
        n_generations=2,
        sa_iterations=100,
        wall_cap=30.0,
        rng_seed=3,
        verbose=False,
    )
    assert len(result.generation_log) >= 1
    for entry in result.generation_log:
        assert "generation" in entry
        assert "entropy" in entry
        assert "best_abs_det" in entry
