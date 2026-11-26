"""Sanity tests for the five diagnostic deep-dive routines."""

from __future__ import annotations

import json

from hadamard_maxdet.diagnostics import (
    basin_map,
    cross_n_validation,
    energy_profile,
    move_profile,
    orbit_size_estimate,
)
from hadamard_maxdet.matrix import Matrix
import random


def test_basin_map_smoke_n5():
    res = basin_map(5, n_seeds=20, iters_per_seed=2_000, base_seed=0)
    assert res.n == 5
    assert res.n_seeds == 20
    # We must end up with at least one cluster covering all seeds.
    assert sum(c.cluster_size for c in res.clusters) == 20
    assert len(res.clusters) >= 1
    j = res.to_json()
    assert "clusters" in j
    assert "largest_cluster_fraction" in j
    # Each cluster has positive size.
    for c in res.clusters:
        assert c.cluster_size >= 1
        assert c.abs_det >= 0


def test_energy_profile_d1_n7():
    rng = random.Random(42)
    M = Matrix.random(7, rng).entries
    res = energy_profile(M, distances=(1, 2), n_samples_per_distance=50, seed=0)
    assert res.n == 7
    bins = {b.distance: b for b in res.bins}
    assert 1 in bins and 2 in bins
    # d=1 covers all 7*7 = 49 single-flip neighbours.
    assert bins[1].n_samples == 49
    assert bins[2].n_samples <= 50
    # max abs_det >= min abs_det
    assert bins[1].max_abs_det >= bins[1].min_abs_det
    j = res.to_json()
    json.dumps(j)  # JSON-serialisable


def test_orbit_size_estimate_minimum_lower_bound():
    rng = random.Random(0)
    M = Matrix.random(5, rng).entries
    res = orbit_size_estimate(M, n_samples=200, seed=0)
    assert res.n_unique >= 1
    # Lower bound is at least 1 (orbit can't be empty).
    assert res.lower_bound_95 >= 1
    j = res.to_json()
    json.dumps(j)


def test_move_profile_smoke_n5():
    res = move_profile(5, iterations=500, seed=0)
    assert res.n == 5
    assert res.iterations == 500
    assert len(res.decile_stats) == 10
    assert res.best_abs_det >= 1
    j = res.to_json()
    json.dumps(j)


def test_cross_n_validation_small_orders():
    rows = cross_n_validation(
        orders_to_test=(5, 7),
        n_starts=2,
        iterations=2_000,
        use_structured=False,
        base_seed=0,
    )
    assert len(rows) == 2
    for r in rows:
        assert r.sa_best >= 1
        assert r.overall_best == max(r.sa_best, r.structured_best or 0)
        assert r.proven_max is not None  # both are proven cases
    j = [r.to_json() for r in rows]
    json.dumps(j)
