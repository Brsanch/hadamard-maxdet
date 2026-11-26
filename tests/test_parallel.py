"""Tests for replica exchange and population annealing.

These tests verify that the parallel samplers reach the proven maximal
|det| at orders where both the exact value and a Hadamard-style
construction are known.
"""

from __future__ import annotations

import pytest

from hadamard_maxdet.parallel import (
    population_annealing,
    replica_exchange,
)


def test_replica_exchange_h4_reaches_16():
    """n=4: proven max 16 (Sylvester). Any decent RE run finds it fast."""
    r = replica_exchange(
        4,
        n_replicas=8,
        iterations=300,
        exchange_every=50,
        seed=0,
    )
    assert r.abs_det == 16


def test_replica_exchange_h5_reaches_48():
    """n=5: Ehlich–Wojtas tight at 48."""
    r = replica_exchange(
        5,
        n_replicas=10,
        iterations=600,
        exchange_every=100,
        seed=1,
    )
    assert r.abs_det == 48


def test_replica_exchange_h8_reaches_sylvester():
    """n=8: Hadamard bound 4096 (Sylvester)."""
    r = replica_exchange(
        8,
        n_replicas=16,
        iterations=1500,
        exchange_every=200,
        seed=2,
    )
    assert r.abs_det == 4096


@pytest.mark.slow
def test_replica_exchange_h11_reaches_327680():
    """n=11: proven max 327680. 30 replicas x 10000 steps as specified."""
    r = replica_exchange(
        11,
        n_replicas=30,
        iterations=10_000,
        exchange_every=500,
        seed=7,
    )
    assert r.abs_det == 327680


def test_replica_exchange_cold_pair_acceptance_nonzero():
    """Diagnostic: on the default ladder the coldest-pair swap acceptance
    should be meaningfully positive. Tests that the ladder is not so
    spread out that cold replicas never exchange."""
    r = replica_exchange(
        6,
        n_replicas=20,
        iterations=800,
        exchange_every=50,
        seed=3,
    )
    rates = r.swap_rates()
    # Cold pair = index 0 (between temps[0] and temps[1]).
    assert rates[0] > 0.05, (
        f"cold-pair swap rate too low: {rates[0]:.3f}; full rates={rates}"
    )


def test_replica_exchange_reports_swap_stats():
    """Swap-rate bookkeeping matches n_replicas - 1."""
    r = replica_exchange(
        4,
        n_replicas=6,
        iterations=200,
        exchange_every=50,
        seed=11,
    )
    assert len(r.swap_attempts) == 5
    assert len(r.swap_accepts) == 5
    assert len(r.temperatures) == 6
    # Monotone ladder (cold to hot).
    for a, b in zip(r.temperatures, r.temperatures[1:]):
        assert b > a


def test_population_annealing_h4_reaches_16():
    r = population_annealing(
        4,
        population_size=16,
        n_temperatures=10,
        iterations_per_step=50,
        seed=0,
    )
    assert r.abs_det == 16


def test_population_annealing_h5_reaches_48():
    r = population_annealing(
        5,
        population_size=24,
        n_temperatures=15,
        iterations_per_step=80,
        seed=4,
    )
    assert r.abs_det == 48


def test_population_annealing_ess_shrinks_or_stable():
    """ESS is bounded above by the population size."""
    r = population_annealing(
        5,
        population_size=16,
        n_temperatures=8,
        iterations_per_step=50,
        seed=5,
    )
    for ess in r.effective_sample_sizes:
        assert 0.0 < ess <= 16.0 + 1e-9
