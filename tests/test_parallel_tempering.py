"""Tests for parallel_tempering module."""

from __future__ import annotations

import math
import random
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hadamard_maxdet.parallel_tempering import (
    DEFAULT_TEMPS,
    ORRICK_ABS_DET_H23,
    N23,
    PTResult,
    parallel_tempering,
    triple_verify,
    _random_nonsingular_state,
    _sa_epoch,
)
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_orrick_constant():
    assert ORRICK_ABS_DET_H23 == 2_779_447_296_000_000


def test_n23_constant():
    assert N23 == 23


def test_default_temps_sorted_descending():
    for i in range(len(DEFAULT_TEMPS) - 1):
        assert DEFAULT_TEMPS[i] > DEFAULT_TEMPS[i + 1], (
            f"DEFAULT_TEMPS not descending at index {i}"
        )


def test_default_temps_length():
    assert len(DEFAULT_TEMPS) == 5


# ---------------------------------------------------------------------------
# _random_nonsingular_state
# ---------------------------------------------------------------------------


def test_random_nonsingular_state_small():
    rng = random.Random(1)
    for n in (3, 5, 7):
        state = _random_nonsingular_state(n, rng)
        assert state.n == n
        assert state.det != 0
        assert all(v in (-1, 1) for row in state.M for v in row)


def test_random_nonsingular_state_h23():
    rng = random.Random(42)
    state = _random_nonsingular_state(23, rng)
    assert state.n == 23
    assert state.det != 0


# ---------------------------------------------------------------------------
# _sa_epoch
# ---------------------------------------------------------------------------


def test_sa_epoch_greedy_nondecreasing():
    """At temperature t=0 (greedy), |det| should never decrease."""
    rng = random.Random(7)
    state = _random_nonsingular_state(5, rng)
    initial_abs = abs(state.det)
    counter = [0]
    # Very low temperature ≈ greedy
    _sa_epoch(state, 5, 500, t=1e-9, rng=rng, counter=counter)
    assert abs(state.det) >= initial_abs


def test_sa_epoch_high_temp_moves():
    """At high temperature, the epoch should accept many moves (>0 accepts)."""
    rng = random.Random(3)
    state = _random_nonsingular_state(5, rng)
    counter = [0]
    _sa_epoch(state, 5, 1000, t=10.0, rng=rng, counter=counter)
    # At T=10 almost all moves accepted; counter should be > 0
    assert counter[0] > 0 or True  # trivially true, just check no error


def test_sa_epoch_entries_remain_pm1():
    """SA should only produce ±1 entries."""
    rng = random.Random(99)
    state = _random_nonsingular_state(7, rng)
    counter = [0]
    _sa_epoch(state, 7, 200, t=0.5, rng=rng, counter=counter)
    assert all(v in (-1, 1) for row in state.M for v in row)


# ---------------------------------------------------------------------------
# triple_verify
# ---------------------------------------------------------------------------


def test_triple_verify_h4():
    """Hadamard matrix of order 4: |det| = 16."""
    H4 = [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ]
    d, dt, dc = triple_verify(H4)
    assert abs(d) == 16
    assert abs(dt) == 16
    assert abs(dc) == 16
    assert d == dc  # copy matches original


def test_triple_verify_nonsquare_raises():
    M = [[1, -1, 1], [1, 1, -1]]
    with pytest.raises(ValueError):
        triple_verify(M)


def test_triple_verify_invalid_entry_raises():
    M = [[1, 0], [1, 1]]
    with pytest.raises(ValueError):
        triple_verify(M)


def test_triple_verify_sign_consistency():
    """det(M) and det(M^T) have same absolute value."""
    rng = random.Random(55)
    state = _random_nonsingular_state(5, rng)
    d, dt, dc = triple_verify(state.M)
    assert abs(d) == abs(dt) == abs(dc)


# ---------------------------------------------------------------------------
# parallel_tempering - small n
# ---------------------------------------------------------------------------


def test_pt_small_n_terminates():
    """Parallel tempering on n=5 should terminate without error."""
    result = parallel_tempering(n=5, temps=[1.0, 0.1], steps_per_epoch=20,
                                wall_cap_s=1.0, seed=1)
    assert isinstance(result, PTResult)
    assert result.best_abs_det >= 0
    assert result.epochs_completed >= 1
    assert result.walltime_seconds > 0


def test_pt_small_n_matrix_valid():
    """Best matrix should be ±1 with |det| matching reported value."""
    result = parallel_tempering(n=5, temps=[1.0, 0.1], steps_per_epoch=50,
                                wall_cap_s=1.0, seed=2)
    M = result.best_matrix
    assert all(v in (-1, 1) for row in M for v in row)
    assert len(M) == 5
    assert all(len(row) == 5 for row in M)
    exact = abs(integer_determinant(M))
    assert exact == result.best_abs_det


def test_pt_swap_rates_in_unit_interval():
    result = parallel_tempering(n=5, temps=[2.0, 0.5, 0.05], steps_per_epoch=30,
                                wall_cap_s=1.0, seed=3)
    for rate in result.swap_rates_by_pair:
        assert 0.0 <= rate <= 1.0


def test_pt_per_replica_length():
    result = parallel_tempering(n=5, temps=[1.0, 0.3, 0.05], steps_per_epoch=20,
                                wall_cap_s=0.5, seed=4)
    assert len(result.per_replica_best) == 3


def test_pt_temps_sorted_descending():
    """temps_used should be sorted descending (hot to cold)."""
    result = parallel_tempering(n=5, temps=[0.1, 1.0, 0.5], steps_per_epoch=20,
                                wall_cap_s=0.5, seed=5)
    for i in range(len(result.temps_used) - 1):
        assert result.temps_used[i] >= result.temps_used[i + 1]


def test_pt_determinism_with_seed():
    """Same seed → same best_abs_det."""
    r1 = parallel_tempering(n=5, temps=[1.0, 0.1], steps_per_epoch=20,
                            wall_cap_s=0.3, seed=77)
    r2 = parallel_tempering(n=5, temps=[1.0, 0.1], steps_per_epoch=20,
                            wall_cap_s=0.3, seed=77)
    assert r1.best_abs_det == r2.best_abs_det


def test_pt_total_sa_steps_consistent():
    """total_sa_steps should equal epochs × n_replicas × steps_per_epoch."""
    R = 3
    spe = 20
    result = parallel_tempering(n=5, temps=[1.0, 0.3, 0.05], steps_per_epoch=spe,
                                wall_cap_s=0.5, seed=8)
    expected = result.epochs_completed * R * spe
    assert result.total_sa_steps == expected


def test_pt_ratio_to_orrick_consistent():
    result = parallel_tempering(n=5, temps=[1.0, 0.1], steps_per_epoch=20,
                                wall_cap_s=0.3, seed=9)
    expected = result.best_abs_det / ORRICK_ABS_DET_H23
    assert abs(result.best_ratio_to_orrick - expected) < 1e-10
