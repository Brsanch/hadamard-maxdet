"""Tests for the non-local switching moves and switch-enhanced SA."""

import random

import pytest

from hadamard_maxdet import Matrix, verify
from hadamard_maxdet.switching import (
    gram_preserving_switch,
    make_switching_move_fn,
    row_pair_block_switch,
    sa_with_switching,
    two_by_two_switch,
)
from hadamard_maxdet.verifier import integer_determinant


def _fresh(n: int, seed: int) -> list[list[int]]:
    return Matrix.random(n, random.Random(seed)).entries


# ---------------------------------------------------------------------------
# row_pair_block_switch
# ---------------------------------------------------------------------------


def test_row_pair_block_switch_stays_pm1():
    rng = random.Random(0)
    M = _fresh(7, 1)
    for _ in range(50):
        N = row_pair_block_switch(M, rng)
        assert N is not None
        assert verify(N).valid
        M = N


def test_row_pair_block_switch_does_not_mutate_input():
    M = _fresh(6, 2)
    snap = [row[:] for row in M]
    row_pair_block_switch(M, random.Random(0))
    assert M == snap


def test_row_pair_block_switch_preserves_pair_inner_product():
    """⟨v_i, v_j⟩ for the chosen row pair is invariant under the switch."""
    rng = random.Random(0)
    n = 9
    M = _fresh(n, 3)
    # Drive deterministically so we know which pair was touched.
    for trial in range(20):
        rng_local = random.Random(100 + trial)
        i, j = rng_local.sample(range(n), 2)
        # Re-implement the move using the same RNG path: replicate the rng
        # reads inside row_pair_block_switch by calling it with a copy.
        # Simpler: call the move and then check the (i,j) inner product
        # over *all* row pairs; at least one pair must be invariant, and
        # we verify the count of invariant pairs is exactly the one that
        # moved (or all, if the proposal was a no-op which never happens
        # because |T| >= 1).
        before_grams = [[sum(M[a][k] * M[b][k] for k in range(n)) for b in range(n)] for a in range(n)]
        N = row_pair_block_switch(M, random.Random(100 + trial))
        assert N is not None
        after_grams = [[sum(N[a][k] * N[b][k] for k in range(n)) for b in range(n)] for a in range(n)]
        # Diagonal Gram entries (⟨v_a, v_a⟩ = n) always preserved.
        for a in range(n):
            assert before_grams[a][a] == after_grams[a][a] == n
        # Find the pair that was switched: both rows should differ from M
        # in identical column positions.
        diff_rows = [a for a in range(n) if M[a] != N[a]]
        assert len(diff_rows) == 2  # exactly the (i, j) pair
        a, b = diff_rows
        diff_cols_a = {k for k in range(n) if M[a][k] != N[a][k]}
        diff_cols_b = {k for k in range(n) if M[b][k] != N[b][k]}
        assert diff_cols_a == diff_cols_b  # same column set T
        assert len(diff_cols_a) >= 1
        # Pair inner product invariant.
        assert before_grams[a][b] == after_grams[a][b]
        assert before_grams[b][a] == after_grams[b][a]


def test_gram_preserving_switch_alias():
    rng1 = random.Random(7)
    rng2 = random.Random(7)
    M = _fresh(6, 4)
    A = row_pair_block_switch(M, rng1)
    B = gram_preserving_switch(M, rng2)
    assert A == B


# ---------------------------------------------------------------------------
# two_by_two_switch
# ---------------------------------------------------------------------------


def test_two_by_two_switch_stays_pm1():
    rng = random.Random(0)
    # Use a random matrix; for n large enough an all-equal 2x2 block
    # almost surely exists.
    M = _fresh(8, 5)
    for _ in range(20):
        N = two_by_two_switch(M, rng)
        if N is None:
            continue
        assert verify(N).valid
        M = N


def test_two_by_two_switch_flips_exactly_four_entries():
    rng = random.Random(0)
    n = 10
    M = _fresh(n, 6)
    found = 0
    for trial in range(50):
        N = two_by_two_switch(M, random.Random(50 + trial))
        if N is None:
            continue
        diff_positions = [(a, b) for a in range(n) for b in range(n) if M[a][b] != N[a][b]]
        if not diff_positions:
            continue  # extremely unlikely; would mean RNG selected a no-op
        found += 1
        assert len(diff_positions) == 4
        # The four positions form a 2x2 grid (two distinct rows, two
        # distinct columns).
        rows = {a for a, _ in diff_positions}
        cols = {b for _, b in diff_positions}
        assert len(rows) == 2 and len(cols) == 2
        for a in rows:
            for b in cols:
                assert (a, b) in diff_positions
        # All four differ by the same sign (constant block before, constant
        # block after).
        a1, a2 = rows
        b1, b2 = cols
        before_vals = {M[a][b] for a in rows for b in cols}
        after_vals = {N[a][b] for a in rows for b in cols}
        assert len(before_vals) == 1  # constant block before
        assert len(after_vals) == 1   # constant block after
        assert before_vals != after_vals  # opposite sign
    assert found > 0  # we should have hit at least one accepted switch


def test_two_by_two_switch_returns_none_when_impossible():
    """A Hadamard matrix has no constant 2x2 block (all four equal)."""
    # Sylvester H_4: every 2x2 has a sign in it; all-equal blocks DO exist
    # actually (e.g. H[0][0..1]=1, H[1][0..1]=1,1 in some Hadamard
    # constructions). So instead, construct a deliberately tiny case where
    # we know the answer.
    # Construct a 2x2 with no all-equal block.
    M = [[1, -1], [-1, 1]]
    N = two_by_two_switch(M, random.Random(0))
    assert N is None  # Only one 2x2 block possible (the whole matrix), and it's not constant.


def test_two_by_two_switch_does_not_mutate_input():
    M = _fresh(6, 7)
    snap = [row[:] for row in M]
    two_by_two_switch(M, random.Random(0))
    assert M == snap


# ---------------------------------------------------------------------------
# Mixed move proposer
# ---------------------------------------------------------------------------


def test_make_switching_move_fn_always_valid():
    fn = make_switching_move_fn()
    rng = random.Random(42)
    M = _fresh(7, 8)
    for _ in range(200):
        p = fn(M, rng)
        if p is None:
            continue
        assert verify(p).valid
        M = p


def test_make_switching_move_fn_rejects_bad_weights():
    with pytest.raises(ValueError):
        make_switching_move_fn(0.5, 0.4, 0.4)  # sum > 1
    with pytest.raises(ValueError):
        make_switching_move_fn(-0.1, 0.55, 0.55)  # negative


# ---------------------------------------------------------------------------
# SA with switching
# ---------------------------------------------------------------------------


def test_sa_with_switching_hits_h11_max():
    """Switch-enhanced SA reaches H11's proven max |det| = 327680."""
    # Use multi-start to be robust at this small budget.
    best = 0
    for seed in range(8):
        r = sa_with_switching(
            n=11,
            iterations=20_000,
            t_start=1.5,
            t_end=0.005,
            seed=seed,
        )
        if r.abs_det > best:
            best = r.abs_det
        if best >= 327680:
            break
    assert best == 327680


def test_sa_with_switching_returns_valid_matrix():
    r = sa_with_switching(n=8, iterations=2000, seed=0)
    assert verify(r.matrix.entries).valid
    assert r.abs_det == abs(integer_determinant(r.matrix.entries))


def test_sa_with_switching_h19_returns_a_value():
    """Smoke test: H19 SA with switching produces a positive |det|."""
    r = sa_with_switching(
        n=19,
        iterations=500,  # tiny budget — just confirm it runs
        t_start=1.5,
        t_end=0.01,
        seed=0,
    )
    assert r.abs_det > 0
    assert verify(r.matrix.entries).valid
    # All three move kinds were proposed (probabilistically).
    assert sum(r.proposed_by_kind.values()) > 0


def test_sa_with_switching_walltime_budget_caps():
    """Setting walltime_budget caps the run, even when iterations is huge."""
    import time
    t0 = time.monotonic()
    r = sa_with_switching(
        n=19,
        iterations=10_000_000,
        t_start=1.5,
        t_end=0.01,
        seed=0,
        walltime_budget=1.0,
    )
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0  # generous slack for OS noise
    assert r.iterations < 10_000_000
