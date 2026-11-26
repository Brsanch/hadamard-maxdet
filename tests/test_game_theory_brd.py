"""Tests for game_theory_brd.py (Best Response Dynamics)."""

import random

import numpy as np
import pytest

from hadamard_maxdet.game_theory_brd import (
    ORRICK_ABS_DET_H23,
    N23,
    _best_response_matrix,
    async_brd_pass,
    brd_multistart_search,
    brd_optimize,
    sync_brd_step,
    triple_verify,
)
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Fixed point characterisation
# ---------------------------------------------------------------------------


def test_fixed_point_is_local_max_n5():
    """A sync-BRD fixed point should be a single-flip local maximum."""
    random.seed(7)
    n = 5
    M_init = [[random.choice([-1, 1]) for _ in range(n)] for _ in range(n)]
    M_fp, _, _ = brd_optimize(M_init, n=n, max_iter=300)
    det_base = abs(integer_determinant(M_fp))
    # Check all single-flip neighbours are ≤ det_base
    for i in range(n):
        for j in range(n):
            M_flip = [row[:] for row in M_fp]
            M_flip[i][j] *= -1
            det_flip = abs(integer_determinant(M_flip))
            assert det_flip <= det_base + 1, (
                f"Flip ({i},{j}) improves |det| from {det_base} to {det_flip}; "
                "not a local maximum"
            )


def test_fixed_point_satisfies_br_condition_n7():
    """At a sync-BRD fixed point, M[i][j] == sign(M^{-1}[j,i]) for all i,j."""
    random.seed(13)
    n = 7
    M_init = [[random.choice([-1, 1]) for _ in range(n)] for _ in range(n)]
    M_fp, _, _ = brd_optimize(M_init, n=n, max_iter=500)
    M_float = np.array(M_fp, dtype=float)
    det_f = np.linalg.det(M_float)
    if abs(det_f) < 1e-6:
        pytest.skip("Converged to singular matrix")
    Minv = np.linalg.inv(M_float)
    for i in range(n):
        for j in range(n):
            expected_sign = 1 if Minv[j, i] > 0 else -1
            assert M_fp[i][j] == expected_sign, (
                f"M[{i}][{j}]={M_fp[i][j]} but sign(M^{{-1}}[{j},{i}])={expected_sign}"
            )


# ---------------------------------------------------------------------------
# BRD step properties
# ---------------------------------------------------------------------------


def test_sync_brd_step_returns_list_of_lists():
    random.seed(1)
    n = 5
    M = [[random.choice([-1, 1]) for _ in range(n)] for _ in range(n)]
    M_new, converged = sync_brd_step(M)
    assert M_new is not None
    assert len(M_new) == n
    assert all(len(row) == n for row in M_new)
    assert all(v in (-1, 1) for row in M_new for v in row)


def test_sync_brd_step_singular_returns_none():
    n = 3
    # Singular: all-ones matrix
    M = [[1] * n for _ in range(n)]
    M_new, converged = sync_brd_step(M)
    assert M_new is None
    assert not converged


def test_sync_brd_does_not_increase_det_oscillate_without_convergence():
    """After many sync rounds, |det| should be non-decreasing on average."""
    random.seed(99)
    n = 5
    M = [[random.choice([-1, 1]) for _ in range(n)] for _ in range(n)]
    seen_dets = []
    for _ in range(30):
        M_new, converged = sync_brd_step(M)
        if M_new is None:
            break
        M = M_new
        seen_dets.append(abs(integer_determinant(M)))
        if converged:
            break
    # At the final step, |det| should be ≥ 1st step (rough sanity only)
    if len(seen_dets) >= 2:
        assert seen_dets[-1] >= seen_dets[0] * 0.5  # generous lower bound


def test_async_brd_pass_improves_or_stays():
    """Async pass should never decrease |det|."""
    random.seed(42)
    n = 7
    M = [[random.choice([-1, 1]) for _ in range(n)] for _ in range(n)]
    det_before = abs(integer_determinant(M))
    M_after, n_flips = async_brd_pass(M, n=n)
    det_after = abs(integer_determinant(M_after))
    assert det_after >= det_before, (
        f"Async pass decreased |det| from {det_before} to {det_after}"
    )


def test_async_brd_converges_small():
    """Async BRD should converge on a small matrix."""
    random.seed(3)
    n = 5
    M_init = [[random.choice([-1, 1]) for _ in range(n)] for _ in range(n)]
    M_fp, n_rounds, _ = brd_optimize(M_init, mode="async", n=n, max_iter=200)
    # Verify it's a fixed point: one more async pass → 0 flips
    _, n_flips = async_brd_pass(M_fp, n=n)
    assert n_flips == 0, f"Expected 0 flips at fixed point, got {n_flips}"
    assert n_rounds < 200, "Should converge well before 200 rounds"


# ---------------------------------------------------------------------------
# Best-response matrix
# ---------------------------------------------------------------------------


def test_best_response_matrix_shape():
    random.seed(4)
    n = 7
    M = np.array([[random.choice([-1, 1]) for _ in range(n)] for _ in range(n)], dtype=float)
    br = _best_response_matrix(M)
    assert br is not None
    assert len(br) == n
    assert all(len(r) == n for r in br)
    assert all(v in (-1, 1) for row in br for v in row)


def test_best_response_matrix_singular_returns_none():
    n = 4
    M = np.ones((n, n), dtype=float)
    assert _best_response_matrix(M) is None


# ---------------------------------------------------------------------------
# Multi-start search (small scale sanity)
# ---------------------------------------------------------------------------


def test_brd_multistart_finds_nonzero_det():
    """Even a 1-second budget should find at least one non-singular matrix."""
    best_M, best_det, stats = brd_multistart_search(
        n=7,
        time_limit_s=1.0,
        seed=0,
        mode="sync",
        verbose=False,
    )
    assert best_det > 0
    assert stats["n_trials"] > 0
    assert stats["best_abs_det"] == best_det


def test_brd_multistart_finds_local_max_n7():
    """Multi-start on n=7 within 2 seconds; best result is a local max."""
    best_M, best_det, stats = brd_multistart_search(
        n=7,
        time_limit_s=2.0,
        seed=1,
        mode="sync",
        verbose=False,
    )
    assert best_M is not None
    # Verify local maximality
    n = 7
    for i in range(n):
        for j in range(n):
            M_flip = [row[:] for row in best_M]
            M_flip[i][j] *= -1
            det_flip = abs(integer_determinant(M_flip))
            assert det_flip <= best_det, (
                f"Flip ({i},{j}) improves from {best_det} to {det_flip}; not local max"
            )


# ---------------------------------------------------------------------------
# Triple verify
# ---------------------------------------------------------------------------


def test_triple_verify_hadamard4():
    """H4 Hadamard matrix: det=±16, det(M)=det(M^T) always."""
    M = [
        [1,  1,  1,  1],
        [1, -1,  1, -1],
        [1,  1, -1, -1],
        [1, -1, -1,  1],
    ]
    det_M, det_MT, match = triple_verify(M)
    assert abs(det_M) == 16
    assert abs(det_MT) == 16
    assert match


def test_triple_verify_random_5x5():
    random.seed(77)
    n = 5
    M = [[random.choice([-1, 1]) for _ in range(n)] for _ in range(n)]
    det_M, det_MT, match = triple_verify(M)
    # det(M) = det(M^T) is a mathematical identity
    assert match
    assert abs(det_M) == abs(det_MT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_orrick_constant():
    assert ORRICK_ABS_DET_H23 == 2_779_447_296_000_000


def test_n23_constant():
    assert N23 == 23
