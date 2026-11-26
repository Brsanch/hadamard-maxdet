"""Tests for the M1-anchored SA module."""
from __future__ import annotations

import random

import numpy as np

from hadamard_maxdet.incremental import sa_incremental
from hadamard_maxdet.m1_anchor import (
    TSupportState,
    sa_m1_anchored,
    t_support_cardinality,
    t_support_counter,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.structural_test import triple_cyclic_product_stats
from hadamard_maxdet.verifier import integer_determinant


def _random_pm1(n: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        [1 if rng.random() < 0.5 else -1 for _ in range(n)]
        for _ in range(n)
    ]


def test_t_support_cardinality_matches_structural_test() -> None:
    """t_support_cardinality(M) agrees with triple_cyclic_product_stats."""
    for seed in (0, 1, 2, 7, 42):
        for n in (5, 8, 11):
            M = _random_pm1(n, seed)
            card_fast = t_support_cardinality(M)
            stats = triple_cyclic_product_stats(M)
            assert card_fast == stats["cardinality"], (
                f"n={n} seed={seed}: fast={card_fast} stats={stats['cardinality']}"
            )


def test_t_support_counter_matches_full_recompute() -> None:
    """The Counter built incrementally agrees with one from scratch on every flip."""
    n = 8
    rng = random.Random(2026)
    M = _random_pm1(n, seed=2026)
    state = TSupportState.from_matrix(M)
    # Reference: full recompute.
    ref_cnt = t_support_counter(M)
    # Drop zero-count keys for comparison.
    state_keys = {k: v for k, v in state.counter.items() if v > 0}
    ref_keys = {k: v for k, v in ref_cnt.items() if v > 0}
    assert state_keys == ref_keys
    assert state.cardinality == len(ref_keys)

    # Apply 100 random flips, comparing after each.
    for step in range(100):
        i = rng.randrange(n)
        c = rng.randrange(n)
        # Flip M, then call update_after_flip with the NEW value already in place.
        M[i][c] = -M[i][c]
        state.update_after_flip(M, i, c)

        ref_cnt = t_support_counter(M)
        state_keys = {k: v for k, v in state.counter.items() if v > 0}
        ref_keys = {k: v for k, v in ref_cnt.items() if v > 0}
        assert state_keys == ref_keys, (
            f"step {step}, flip ({i}, {c}): "
            f"incremental != reference"
        )
        assert state.cardinality == len(ref_keys)
        # Gram check too.
        A = np.asarray(M, dtype=int)
        G_ref = (A @ A.T).astype(int)
        assert np.array_equal(state.G, G_ref), (
            f"Gram drift at step {step}"
        )


def test_t_support_state_invariant_under_row_negation() -> None:
    """T(i, j, k) is invariant under negating any single row, so the multiset
    counter and cardinality are too. (Sanity check on the M1 invariance
    property.)"""
    n = 7
    M = _random_pm1(n, seed=7)
    cnt_orig = t_support_counter(M)
    # Negate row 3.
    M2 = [row[:] for row in M]
    M2[3] = [-x for x in M2[3]]
    cnt_neg = t_support_counter(M2)
    assert cnt_orig == cnt_neg


def test_sa_m1_anchored_lambda_zero_matches_sa_incremental_quality() -> None:
    """At lambda=0, sa_m1_anchored should produce the same |det| trajectory
    as sa_incremental for the same seed (both are vanilla SA on -log|det|).

    They will not be byte-identical because the random number stream is
    consumed slightly differently (sa_m1_anchored has the extra T-support
    branch), but with identical seeds and short runs the best |det| should
    be in the same ballpark.
    """
    n = 8
    res_anchor = sa_m1_anchored(
        n,
        iterations=5_000,
        seed=1234,
        lambda_m1=0.0,
        refresh_every=0,
    )
    res_plain = sa_incremental(
        n,
        iterations=5_000,
        seed=1234,
        refresh_every=0,
    )
    # Both should reach a nontrivial det.
    assert res_anchor.abs_det > 0
    assert res_plain.abs_det > 0
    # And both should beat a random baseline (random ±1 at n=8 averages
    # |det| ~ a few hundred; SA at n=8 should hit the proven max 4096).
    # Just sanity-check that the anchored version is at least within 0.5x
    # of the plain run.
    assert res_anchor.abs_det >= 0.5 * res_plain.abs_det, (
        f"anchored {res_anchor.abs_det} vs plain {res_plain.abs_det}"
    )


def test_sa_m1_anchored_high_lambda_drives_card_low() -> None:
    """At high lambda, SA should land in low-T-support basins."""
    n = 8
    rng = random.Random(99)
    M0 = Matrix.random(n, rng)
    res_low = sa_m1_anchored(
        n,
        iterations=5_000,
        seed=2025,
        lambda_m1=0.0,
        initial=M0,
        refresh_every=0,
    )
    res_high = sa_m1_anchored(
        n,
        iterations=5_000,
        seed=2025,
        lambda_m1=2.0,
        initial=M0,
        refresh_every=0,
    )
    card_low = t_support_cardinality(res_low.matrix.entries)
    card_high = t_support_cardinality(res_high.matrix.entries)
    # High-lambda run should not have higher card than low-lambda run.
    assert card_high <= card_low + 1, (
        f"high-lambda card {card_high} > low-lambda card {card_low} (+1 slack)"
    )


def test_sa_m1_anchored_returns_verified_det() -> None:
    """The returned ``abs_det`` is the exact Bareiss determinant of the matrix."""
    n = 7
    res = sa_m1_anchored(n, iterations=2_000, seed=11, lambda_m1=0.1)
    assert res.abs_det == abs(integer_determinant(res.matrix.entries))


def test_sa_m1_anchored_uses_initial_seed() -> None:
    """When given an `initial=M_seed`, sa_m1_anchored starts at M_seed.

    With zero iterations the result should be exactly M_seed (after
    Bareiss verification).  With a very cold schedule (t_start = t_end ≈ 0)
    and lambda=0, even hundreds of iterations should rarely move far from
    M_seed's |det|, since uphill moves are nearly forbidden.
    """
    n = 9
    rng = random.Random(123)
    M_seed = Matrix.random(n, rng)
    seed_det = abs(integer_determinant(M_seed.entries))
    assert seed_det > 0, "test fixture must be nonsingular"

    res = sa_m1_anchored(
        n,
        iterations=0,
        seed=0,
        lambda_m1=0.0,
        initial=M_seed,
        refresh_every=0,
    )
    # Zero iterations: best is the seed itself.
    assert res.matrix.entries == M_seed.entries
    assert res.abs_det == seed_det

    # Very cold schedule: should not collapse to zero or a wildly worse |det|.
    res_cold = sa_m1_anchored(
        n,
        iterations=500,
        seed=7,
        lambda_m1=0.0,
        initial=M_seed,
        t_start=0.001,
        t_end=0.0001,
        refresh_every=0,
    )
    # At very cold T, only downhill moves accept; |det| should not decrease.
    assert res_cold.abs_det >= seed_det, (
        f"cold-from-seed dropped: {res_cold.abs_det} < {seed_det}"
    )
