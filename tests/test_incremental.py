"""Tests for rank-1 incremental determinant updates."""

from __future__ import annotations

import os
import random
import time

import pytest

from hadamard_maxdet.incremental import (
    IncrementalState,
    apply_col_negate,
    apply_entry_flip,
    apply_row_negate,
    propose_entry_flip,
    refresh,
    residual,
    sa_incremental,
)
from hadamard_maxdet.search import simulated_annealing
from hadamard_maxdet.verifier import integer_determinant


def _nonsingular_random_matrix(n: int, rng: random.Random) -> list[list[int]]:
    """Return a random ±1 matrix with nonzero determinant."""
    for _ in range(256):
        M = [[rng.choice([-1, 1]) for _ in range(n)] for _ in range(n)]
        if integer_determinant(M) != 0:
            return M
    raise RuntimeError(f"could not sample non-singular ±1 matrix of order {n}")


# ----------------------------------------------------------------------------
# Algebraic correctness: rank-1 flip predicts and preserves det exactly.
# ----------------------------------------------------------------------------


def test_propose_matches_exact_recompute_small():
    rng = random.Random(0)
    for n in (3, 4, 5, 6):
        M = _nonsingular_random_matrix(n, rng)
        st = IncrementalState.from_matrix(M)
        for i in range(n):
            for j in range(n):
                predicted = propose_entry_flip(st, i, j)
                st.M[i][j] = -st.M[i][j]
                actual = integer_determinant(st.M)
                st.M[i][j] = -st.M[i][j]
                assert predicted == actual, (
                    f"n={n} (i,j)=({i},{j}): predicted {predicted} ≠ actual {actual}"
                )


def test_apply_keeps_invariant_long_sequence():
    """After 10000 random flips interleaved with refresh, carried det matches exact."""
    rng = random.Random(42)
    n = 12
    M = _nonsingular_random_matrix(n, rng)
    st = IncrementalState.from_matrix(M)

    applied = 0
    for step in range(10_000):
        i, j = rng.randrange(n), rng.randrange(n)
        new_det = propose_entry_flip(st, i, j)
        if new_det == 0:
            continue  # singular proposal; would break adj update
        apply_entry_flip(st, i, j)
        applied += 1

        if step % 1000 == 999:
            exact = integer_determinant(st.M)
            assert st.det == exact, (
                f"step {step}: carried {st.det} ≠ exact {exact} (applied={applied})"
            )
            assert residual(st) == 0, f"step {step}: residual nonzero"
            # Also check refresh is a no-op at this point.
            refresh(st)
            assert st.det == exact

    # Final check
    exact = integer_determinant(st.M)
    assert st.det == exact
    assert residual(st) == 0


def test_apply_keeps_invariant_n19():
    """Same as above at n=19 (the open H19 order)."""
    rng = random.Random(2026)
    n = 19
    M = _nonsingular_random_matrix(n, rng)
    st = IncrementalState.from_matrix(M)

    for step in range(3000):
        i, j = rng.randrange(n), rng.randrange(n)
        pred = propose_entry_flip(st, i, j)
        if pred == 0:
            continue
        apply_entry_flip(st, i, j)

    exact = integer_determinant(st.M)
    assert st.det == exact, f"n=19 carried {st.det} ≠ exact {exact}"
    assert residual(st) == 0


def test_row_negate_preserves_invariant():
    rng = random.Random(3)
    n = 6
    M = _nonsingular_random_matrix(n, rng)
    st = IncrementalState.from_matrix(M)
    for i in range(n):
        apply_row_negate(st, i)
        exact = integer_determinant(st.M)
        assert st.det == exact
        assert residual(st) == 0


def test_col_negate_preserves_invariant():
    rng = random.Random(4)
    n = 6
    M = _nonsingular_random_matrix(n, rng)
    st = IncrementalState.from_matrix(M)
    for j in range(n):
        apply_col_negate(st, j)
        exact = integer_determinant(st.M)
        assert st.det == exact
        assert residual(st) == 0


# ----------------------------------------------------------------------------
# SA correctness: same endpoint as the baseline on tiny orders.
# ----------------------------------------------------------------------------


def test_sa_incremental_reaches_h4():
    r = sa_incremental(4, iterations=5_000, seed=0)
    assert r.abs_det == 16


def test_sa_incremental_reaches_h5():
    r = sa_incremental(5, iterations=20_000, seed=0)
    assert r.abs_det == 48


def test_sa_incremental_refresh_catches_no_drift():
    """After a long SA, no refresh should ever have corrected the carried det
    (integer adjugate → zero drift). residual_triggered must be 0."""
    r = sa_incremental(11, iterations=20_000, seed=0, refresh_every=500)
    assert r.residual_triggered == 0, (
        f"unexpected adjugate corrections: {r.residual_triggered}"
    )


def test_sa_incremental_final_is_exact():
    """The reported abs_det must come from an exact recomputation, not a
    carried float-y det. Compare to integer_determinant on the reported matrix."""
    r = sa_incremental(7, iterations=5_000, seed=0)
    recomputed = abs(integer_determinant(r.matrix.entries))
    assert r.abs_det == recomputed


# ----------------------------------------------------------------------------
# Timing: incremental is ≥5× faster at n=15. Gated by env var so it does not
# slow CI down; set HADAMARD_RUN_TIMING=1 to run.
# ----------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("HADAMARD_RUN_TIMING"),
    reason="timing test; set HADAMARD_RUN_TIMING=1 to run",
)
def test_sa_incremental_faster_than_baseline_n15():
    n = 15
    # Calibrate iters such that each takes ~1s.
    t0 = time.monotonic()
    baseline = simulated_annealing(n, iterations=3000, seed=0)
    base_rate = baseline.iterations / (time.monotonic() - t0)

    t0 = time.monotonic()
    incr = sa_incremental(n, iterations=50_000, seed=0)
    inc_rate = incr.iterations / (time.monotonic() - t0)

    speedup = inc_rate / base_rate
    assert speedup >= 5.0, (
        f"expected ≥5× speedup at n=15, got {speedup:.2f}× "
        f"(baseline {base_rate:.0f} iter/s, incremental {inc_rate:.0f} iter/s)"
    )
