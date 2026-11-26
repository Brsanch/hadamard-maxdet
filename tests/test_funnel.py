"""Tests for the funnel search architecture."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from hadamard_maxdet.funnel import (
    FunnelDiagnostics,
    funnel_search,
    gram_conserving_pair_flip,
    gram_loudness_filter,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------
# gram_loudness_filter
# ----------------------------------------------------------------------------


def test_gram_loudness_filter_accepts_btc_h23_seed():
    """The BTC seed at n=23 has Gram support {-5, -1, +3} (size 3) and
    strongly negative off-diagonal sum; the filter must accept it."""
    path = REPO_ROOT / "results" / "H23_rich_ansatz_best.json"
    data = json.loads(path.read_text())
    entries = data["entries"]
    assert len(entries) == 23
    assert gram_loudness_filter(
        entries, max_support_size=8, require_negative_bias=True
    ), "BTC H23 seed should pass the loudness filter"


def test_gram_loudness_filter_rejects_random_h23():
    """A uniform random ±1 matrix at n=23 should typically have large
    Gram support and near-zero sum, failing the filter."""
    rng = random.Random(20260424)
    n_rejected = 0
    for _ in range(10):
        M = Matrix.random(23, rng)
        if not gram_loudness_filter(
            M.entries, max_support_size=8, require_negative_bias=True
        ):
            n_rejected += 1
    # In expectation ≥ 8/10 random matrices fail at n=23. With a fixed seed
    # the actual count is deterministic; we assert most are rejected.
    assert n_rejected >= 7, (
        f"expected ≥ 7/10 random ±1 matrices at n=23 to be rejected; "
        f"got {n_rejected}"
    )


def test_gram_loudness_filter_support_only_mode():
    """When require_negative_bias is False, only support size matters."""
    # Single row swap of identity-style pattern: design a matrix with
    # narrow support but non-negative sum.
    rng = random.Random(1)
    M = Matrix.random(11, rng).entries
    # Weak assertion: when bias is required, fewer matrices pass than
    # when only support size is required.
    n_pass_strict = 0
    n_pass_loose = 0
    for _ in range(50):
        Mr = Matrix.random(11, rng).entries
        if gram_loudness_filter(Mr, max_support_size=20, require_negative_bias=True):
            n_pass_strict += 1
        if gram_loudness_filter(Mr, max_support_size=20, require_negative_bias=False):
            n_pass_loose += 1
    assert n_pass_loose >= n_pass_strict


# ----------------------------------------------------------------------------
# gram_conserving_pair_flip
# ----------------------------------------------------------------------------


def _inner_product(M: list[list[int]], i: int, j: int) -> int:
    return sum(M[i][k] * M[j][k] for k in range(len(M)))


def test_gram_conserving_pair_flip_preserves_target_inner_product():
    """The load-bearing claim: flipping M[i][k] AND M[j][k] simultaneously
    leaves ⟨v_i, v_j⟩ exactly unchanged. Asserted on multiple sizes and seeds."""
    for n in (5, 7, 11, 23):
        for seed in range(5):
            rng = random.Random(seed * 31 + n)
            M = Matrix.random(n, rng).entries
            for _ in range(10):
                i, j = sorted(rng.sample(range(n), 2))
                before = _inner_product(M, i, j)
                M_new = gram_conserving_pair_flip(M, rng, target_pair=(i, j))
                after = _inner_product(M_new, i, j)
                assert before == after, (
                    f"Gram-conserving move violated invariant at n={n}, "
                    f"pair=({i},{j}): {before} → {after}"
                )


def test_gram_conserving_pair_flip_produces_valid_pm1_matrix():
    """The result must still be a valid square ±1 matrix of the same size."""
    rng = random.Random(42)
    for n in (3, 7, 12, 23):
        M = Matrix.random(n, rng).entries
        M_new = gram_conserving_pair_flip(M, rng)
        assert len(M_new) == n
        for row in M_new:
            assert len(row) == n
            for v in row:
                assert v in (-1, 1), f"non-±1 entry: {v!r}"


def test_gram_conserving_pair_flip_does_not_mutate_input():
    rng = random.Random(0)
    M = Matrix.random(7, rng).entries
    M_snapshot = [row[:] for row in M]
    gram_conserving_pair_flip(M, rng)
    assert M == M_snapshot, "input matrix was mutated"


def test_gram_conserving_pair_flip_default_picks_worst_pair():
    """Without a target pair, the move picks the (i,j) with maximum
    inner product. Verified by checking it preserves *that* pair's
    inner product, which is the claim of the API contract."""
    rng = random.Random(7)
    n = 11
    M = Matrix.random(n, rng).entries
    # Find the worst pair manually
    best_val = None
    best_pair = (0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            v = _inner_product(M, i, j)
            if best_val is None or v > best_val:
                best_val = v
                best_pair = (i, j)
    M_new = gram_conserving_pair_flip(M, rng, target_pair=None)
    # The default target should preserve the worst pair's inner product.
    bi, bj = best_pair
    assert _inner_product(M, bi, bj) == _inner_product(M_new, bi, bj)


def test_gram_conserving_pair_flip_rejects_self_pair():
    rng = random.Random(0)
    M = Matrix.random(5, rng).entries
    with pytest.raises(ValueError):
        gram_conserving_pair_flip(M, rng, target_pair=(2, 2))


# ----------------------------------------------------------------------------
# funnel_search
# ----------------------------------------------------------------------------


def test_funnel_search_reaches_proven_max_at_n5():
    """At n=5 the proven |det|_max is 48 (Ehlich-Wojtas; 2n-1=9, perfect
    square). The funnel should reach it within a small budget."""
    best, diag = funnel_search(
        n=5,
        total_budget_seconds=10.0,
        seed=20260424,
        max_support_size=20,
        require_negative_bias=False,
        refine_iterations=3000,
        phase_a_min_seconds=0.5,
        no_new_window=20,
        phase_b_no_improve_iters=30_000,
    )
    det = abs(integer_determinant(best.entries))
    assert det == 48, f"funnel_search on n=5 returned |det|={det}, expected 48"
    assert isinstance(diag, FunnelDiagnostics)
    assert diag.final_best_det == 48
    assert diag.candidates_tried > 0


def test_funnel_search_reaches_proven_max_at_n7():
    """At n=7 the proven |det|_max is 576 (Barba; n≡3 mod 4)."""
    best, diag = funnel_search(
        n=7,
        total_budget_seconds=15.0,
        seed=20260424,
        max_support_size=20,
        require_negative_bias=False,
        refine_iterations=4000,
        phase_a_min_seconds=0.5,
        no_new_window=20,
        phase_b_no_improve_iters=50_000,
    )
    det = abs(integer_determinant(best.entries))
    assert det == 576, f"funnel_search on n=7 returned |det|={det}, expected 576"


def test_funnel_search_returns_valid_matrix():
    """Sanity: returned matrix is square, ±1, of the right order, and the
    diagnostics match."""
    best, diag = funnel_search(
        n=5,
        total_budget_seconds=3.0,
        seed=0,
        max_support_size=20,
        require_negative_bias=False,
        refine_iterations=1000,
        phase_a_min_seconds=0.2,
        no_new_window=10,
        phase_b_no_improve_iters=10_000,
    )
    assert best.n == 5
    assert len(best.entries) == 5
    for row in best.entries:
        assert len(row) == 5
        for v in row:
            assert v in (-1, 1)
    assert diag.final_best_det == abs(integer_determinant(best.entries))


def test_funnel_search_with_initial_seed_never_regresses():
    """If we hand the funnel a known matrix as initial_best, the returned
    |det| must be ≥ that matrix's |det|."""
    rng = random.Random(0)
    init = Matrix.random(5, rng)
    init_det = abs(integer_determinant(init.entries))
    best, _diag = funnel_search(
        n=5,
        total_budget_seconds=3.0,
        seed=1,
        initial_best=init,
        max_support_size=20,
        require_negative_bias=False,
        refine_iterations=1000,
        phase_a_min_seconds=0.2,
        no_new_window=10,
        phase_b_no_improve_iters=10_000,
    )
    final_det = abs(integer_determinant(best.entries))
    assert final_det >= init_det
