"""Tests for hippocampal_replay.py — neuroscience domain analogy."""
from __future__ import annotations

import random
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest

from hadamard_maxdet.hippocampal_replay import (
    ORRICK_ABS_DET_H23,
    ExperienceBuffer,
    HRResult,
    column_crossover,
    hippocampal_replay,
    row_crossover,
    triple_verify,
)
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# ExperienceBuffer tests
# ---------------------------------------------------------------------------


def _make_rng():
    return random.Random(42)


def _rand_pm1_matrix(n, rng):
    return [[rng.choice([-1, 1]) for _ in range(n)] for _ in range(n)]


def test_buffer_empty_initially():
    buf = ExperienceBuffer(capacity=5)
    assert len(buf) == 0


def test_buffer_rejects_below_threshold():
    buf = ExperienceBuffer(capacity=5, threshold_ratio=0.5)
    low_det = int(0.3 * ORRICK_ABS_DET_H23)
    rng = _make_rng()
    M = _rand_pm1_matrix(5, rng)
    added = buf.try_add(low_det, M)
    assert not added
    assert len(buf) == 0


def test_buffer_accepts_above_threshold():
    buf = ExperienceBuffer(capacity=5, threshold_ratio=0.5)
    high_det = int(0.8 * ORRICK_ABS_DET_H23)
    rng = _make_rng()
    M = _rand_pm1_matrix(5, rng)
    added = buf.try_add(high_det, M)
    assert added
    assert len(buf) == 1


def test_buffer_capacity_eviction():
    buf = ExperienceBuffer(capacity=3, threshold_ratio=0.0)
    rng = _make_rng()
    dets = [1000, 2000, 3000, 4000]
    for d in dets:
        M = _rand_pm1_matrix(5, rng)
        buf.try_add(d, M)
    # Buffer is capped at 3; should have the top-3
    assert len(buf) == 3
    items = buf.all_matrices()
    stored_dets = sorted([x[0] for x in items], reverse=True)
    assert stored_dets == [4000, 3000, 2000]


def test_buffer_eviction_rejects_worse():
    buf = ExperienceBuffer(capacity=2, threshold_ratio=0.0)
    rng = _make_rng()
    buf.try_add(5000, _rand_pm1_matrix(5, rng))
    buf.try_add(4000, _rand_pm1_matrix(5, rng))
    added = buf.try_add(3000, _rand_pm1_matrix(5, rng))
    assert not added
    assert len(buf) == 2


def test_buffer_sorted_descending():
    buf = ExperienceBuffer(capacity=5, threshold_ratio=0.0)
    rng = _make_rng()
    for d in [100, 500, 200, 400, 300]:
        buf.try_add(d, _rand_pm1_matrix(4, rng))
    items = buf.all_matrices()
    dets = [x[0] for x in items]
    assert dets == sorted(dets, reverse=True)


def test_buffer_weighted_sample_pair_distinct():
    buf = ExperienceBuffer(capacity=10, threshold_ratio=0.0)
    rng = _make_rng()
    matrices = [_rand_pm1_matrix(5, rng) for _ in range(5)]
    for i, M in enumerate(matrices):
        buf.try_add((i + 1) * 1000, M)
    rng2 = random.Random(99)
    for _ in range(50):
        A, B = buf.weighted_sample_pair(rng2)
        assert A is not B  # distinct objects


def test_buffer_sample_pair_needs_at_least_2():
    buf = ExperienceBuffer(capacity=5, threshold_ratio=0.0)
    rng = _make_rng()
    buf.try_add(1000, _rand_pm1_matrix(5, rng))
    with pytest.raises(ValueError):
        buf.weighted_sample_pair(_make_rng())


# ---------------------------------------------------------------------------
# Row and column crossover tests
# ---------------------------------------------------------------------------


def test_row_crossover_valid_pm1():
    rng = _make_rng()
    n = 7
    A = _rand_pm1_matrix(n, rng)
    B = _rand_pm1_matrix(n, rng)
    H = row_crossover(A, B, rng)
    assert len(H) == n
    for row in H:
        assert len(row) == n
        for v in row:
            assert v in (-1, 1)


def test_row_crossover_rows_from_parents():
    rng = random.Random(7)
    n = 5
    A = _rand_pm1_matrix(n, rng)
    B = _rand_pm1_matrix(n, rng)
    H = row_crossover(A, B, rng)
    for i in range(n):
        assert H[i] == A[i] or H[i] == B[i], f"Row {i} of hybrid not from A or B"


def test_column_crossover_valid_pm1():
    rng = _make_rng()
    n = 6
    A = _rand_pm1_matrix(n, rng)
    B = _rand_pm1_matrix(n, rng)
    H = column_crossover(A, B, rng)
    assert len(H) == n
    for row in H:
        assert len(row) == n
        for v in row:
            assert v in (-1, 1)


def test_column_crossover_columns_from_parents():
    rng = random.Random(13)
    n = 5
    A = _rand_pm1_matrix(n, rng)
    B = _rand_pm1_matrix(n, rng)
    H = column_crossover(A, B, rng)
    for j in range(n):
        col_H = [H[i][j] for i in range(n)]
        col_A = [A[i][j] for i in range(n)]
        col_B = [B[i][j] for i in range(n)]
        assert col_H == col_A or col_H == col_B, f"Col {j} of hybrid not from A or B"


def test_row_crossover_bias_a_1_always_a():
    rng = random.Random(0)
    n = 5
    A = _rand_pm1_matrix(n, rng)
    B = _rand_pm1_matrix(n, rng)
    H = row_crossover(A, B, rng, bias_a=1.0)
    assert H == A


def test_row_crossover_bias_a_0_always_b():
    rng = random.Random(0)
    n = 5
    A = _rand_pm1_matrix(n, rng)
    B = _rand_pm1_matrix(n, rng)
    H = row_crossover(A, B, rng, bias_a=0.0)
    assert H == B


# ---------------------------------------------------------------------------
# Triple verify
# ---------------------------------------------------------------------------


def test_triple_verify_consistent_small():
    n = 5
    M = [[1, 1, 1, 1, 1],
         [1, -1, 1, -1, 1],
         [1, 1, -1, -1, 1],
         [1, -1, -1, 1, -1],
         [-1, 1, 1, -1, -1]]
    d1, d2, d3 = triple_verify(M)
    assert abs(d1) == abs(d2) == abs(d3)
    assert abs(d1) > 0


def test_triple_verify_matches_integer_determinant():
    rng = random.Random(2026)
    n = 7
    M = _rand_pm1_matrix(n, rng)
    d1, d2, d3 = triple_verify(M)
    expected = integer_determinant(M)
    assert d1 == expected


# ---------------------------------------------------------------------------
# Full run tests (small n to keep fast)
# ---------------------------------------------------------------------------


def test_hippocampal_replay_n5_runs():
    result = hippocampal_replay(
        n=5,
        wall_budget_seconds=2.0,
        buffer_capacity=5,
        buffer_threshold_ratio=0.0,
        explore_iters=500,
        consolidate_iters=500,
        seed=42,
    )
    assert isinstance(result, HRResult)
    assert result.total_sa_steps > 0
    assert result.n_exploration_chains > 0


def test_hippocampal_replay_n5_valid_pm1_output():
    result = hippocampal_replay(
        n=5,
        wall_budget_seconds=2.0,
        buffer_capacity=5,
        buffer_threshold_ratio=0.0,
        explore_iters=500,
        consolidate_iters=500,
        seed=123,
    )
    M = result.best_matrix
    assert len(M) == 5
    for row in M:
        assert len(row) == 5
        for v in row:
            assert v in (-1, 1)


def test_hippocampal_replay_n5_det_consistent():
    result = hippocampal_replay(
        n=5,
        wall_budget_seconds=2.0,
        buffer_capacity=5,
        buffer_threshold_ratio=0.0,
        explore_iters=500,
        consolidate_iters=500,
        seed=99,
    )
    d1, d2, d3 = triple_verify(result.best_matrix)
    assert abs(d1) == abs(d2) == abs(d3)
    assert abs(d1) == result.best_abs_det


def test_hippocampal_replay_n11_reaches_reasonable_det():
    """At n=11 the Hadamard bound is 32768; good SA should get within 10x."""
    result = hippocampal_replay(
        n=11,
        wall_budget_seconds=5.0,
        buffer_capacity=10,
        buffer_threshold_ratio=0.0,
        explore_iters=5000,
        consolidate_iters=3000,
        seed=2026,
    )
    # Max det for n=11 is 32768; we should easily exceed 100
    assert result.best_abs_det > 100


def test_hippocampal_replay_above_orrick_flag_correct():
    result = hippocampal_replay(
        n=5,
        wall_budget_seconds=1.0,
        buffer_capacity=5,
        buffer_threshold_ratio=0.0,
        explore_iters=300,
        consolidate_iters=300,
        seed=0,
    )
    # For n=5, best det is much less than H23 Orrick; flag must be False
    assert not result.above_orrick


def test_hippocampal_replay_ratio_to_orrick_type():
    result = hippocampal_replay(
        n=5,
        wall_budget_seconds=1.0,
        buffer_capacity=5,
        buffer_threshold_ratio=0.0,
        explore_iters=300,
        consolidate_iters=300,
        seed=77,
    )
    assert isinstance(result.best_ratio_to_orrick, float)
    assert 0.0 <= result.best_ratio_to_orrick < 1.0
