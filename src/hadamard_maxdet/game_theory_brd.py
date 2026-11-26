"""
Best Response Dynamics (BRD) search for H23 maxdet.

Models the n×n ±1 matrix as a cooperative coordination game: each of the
n² entries is a player choosing ±1.  All players share the same payoff
|det(M)|.  The individual best response for player (i,j) is

    M[i,j]* = sign(M^{-1}[j, i])

because this maximises |det| holding all other entries fixed.  Proof via the
matrix determinant lemma: flipping M[i,j] multiplies det by the factor
(1 + δ·(M^{-1})[j,i]) where δ = new − old ∈ {−2, +2}.  The absolute value
of this factor exceeds 1 iff sign(M[i,j]) ≠ sign(M^{-1}[j,i]), so the
best-response rule is exactly: M[i,j]* = sign(M^{-1}[j,i]).

Fixed points of BRD (every entry at its best response) are exactly the local
maxima of |det| under single-entry flips — pure Nash equilibria.

Implementation uses *asynchronous* BRD: entries are visited in random order
and each is updated immediately (before the next is checked).  This guarantees
monotone non-decrease of |det| and convergence to a local maximum in finitely
many steps.  The inverse is maintained via rank-1 Sherman-Morrison updates,
making each flip O(n²) instead of O(n³).

A periodic full recomputation guards against floating-point accumulation.
"""

from __future__ import annotations

import random
import time
from typing import Optional

import numpy as np

from hadamard_maxdet.verifier import integer_determinant

N23 = 23
ORRICK_ABS_DET_H23: int = 2_779_447_296_000_000


# ---------------------------------------------------------------------------
# Core BRD primitive: greedy async with Sherman-Morrison inverse updates
# ---------------------------------------------------------------------------


def _best_response_matrix(M_float: np.ndarray) -> Optional[list[list[int]]]:
    """
    Given a float ±1 matrix, compute the best-response assignment for all entries.

    Returns None if M is (near-)singular; otherwise returns a new list-of-lists
    where entry [i][j] = sign(M^{-1}[j, i]).
    """
    det_f = np.linalg.det(M_float)
    if abs(det_f) < 1e-6:
        return None
    Minv = np.linalg.inv(M_float)
    opt_arr = np.sign(Minv.T)
    opt_arr[opt_arr == 0] = 1.0
    return opt_arr.astype(int).tolist()


def greedy_brd_sm(
    M_init: list[list[int]],
    n: int = N23,
    max_passes: int = 100,
    refresh_every: int = 10,
) -> tuple[list[list[int]], int]:
    """
    Async BRD with Sherman-Morrison inverse maintenance.

    At each step the entry (i,j) is flipped iff sign(M[i,j]) ≠ sign(M^{-1}[j,i]).
    Each accepted flip updates M^{-1} via the rank-1 formula

        M_new^{-1} = M^{-1} − (δ / (1 + δ·(M^{-1})[j,i])) · col_i ⊗ row_j

    where δ = new_val − old_val = ∓2 and col_i, row_j are the i-th column and
    j-th row of M^{-1}.  Since only flips that strictly increase |det| are
    accepted (δ·(M^{-1})[j,i] < 0 → |1 + δ·x| > 1 → |det| increases), the
    denominator is always in (0, 1] and the update is numerically stable.

    Returns (M_final, n_passes_taken).  M_final is a local maximum of |det|
    (or as close as float precision permits at convergence).
    """
    M = [row[:] for row in M_init]
    A = np.array(M, dtype=float)
    if abs(np.linalg.det(A)) < 1e-6:
        return M, 0

    Minv = np.linalg.inv(A)
    order = list(range(n * n))

    for pass_num in range(max_passes):
        n_flips = 0
        random.shuffle(order)

        for idx in order:
            i, j = divmod(idx, n)
            v = M[i][j]
            minv_ji = float(Minv[j, i])

            # Best response: sign(minv_ji).  Flip iff current sign is wrong.
            if v * minv_ji >= 0:
                continue

            # δ = new − old = −v − v = −2v  (since new = −v for a ±1 flip)
            delta = -2.0 * v
            denom = 1.0 + delta * minv_ji
            if abs(denom) < 1e-10:
                continue  # near-singular denominator; skip

            col_i = Minv[:, i].copy()
            row_j = Minv[j, :].copy()
            Minv -= (delta / denom) * np.outer(col_i, row_j)
            M[i][j] = -v
            n_flips += 1

        if n_flips == 0:
            return M, pass_num + 1

        # Guard against float drift
        if pass_num % refresh_every == refresh_every - 1:
            Minv = np.linalg.inv(np.array(M, dtype=float))

    return M, max_passes


# ---------------------------------------------------------------------------
# Legacy sync helpers (kept for tests; not used in production search)
# ---------------------------------------------------------------------------


def sync_brd_step(
    M: list[list[int]],
) -> tuple[Optional[list[list[int]]], bool]:
    """
    One synchronous BRD step (all entries update simultaneously).

    Returns (M_new, converged).  May cycle on non-trivial matrices; prefer
    greedy_brd_sm for production use.
    """
    M_float = np.array(M, dtype=float)
    M_new = _best_response_matrix(M_float)
    if M_new is None:
        return None, False
    converged = M_new == M
    return M_new, converged


def async_brd_pass(
    M: list[list[int]],
    n: int = N23,
) -> tuple[list[list[int]], int]:
    """
    One full asynchronous BRD pass (random entry order, full-inversion variant).

    Returns (M_updated, n_flips).  Recomputes M^{-1} from scratch for each
    entry visited — slower than greedy_brd_sm but simpler reference impl.
    """
    M = [row[:] for row in M]
    n_flips = 0

    indices = list(range(n * n))
    random.shuffle(indices)

    for idx in indices:
        i, j = divmod(idx, n)
        M_float = np.array(M, dtype=float)
        det_f = np.linalg.det(M_float)
        if abs(det_f) < 1e-6:
            return M, n_flips
        Minv = np.linalg.inv(M_float)
        optimal = 1 if Minv[j, i] > 0 else -1
        if optimal != M[i][j]:
            M[i][j] = optimal
            n_flips += 1

    return M, n_flips


def brd_optimize(
    M_init: list[list[int]],
    *,
    mode: str = "greedy_sm",
    max_iter: int = 150,
    n: int = N23,
    perturb_k: int = 3,
    cycle_window: int = 8,
) -> tuple[list[list[int]], int, int]:
    """
    Unified BRD driver.

    mode="greedy_sm"  — recommended, uses greedy_brd_sm.
    mode="sync"       — synchronous BRD with cycle detection + perturbation.
    mode="async"      — full-inversion async (reference, slow).
    """
    if mode == "greedy_sm":
        M_final, n_rounds = greedy_brd_sm(M_init, n=n, max_passes=max_iter)
        return M_final, n_rounds, 0

    # --- sync mode ---
    M = [row[:] for row in M_init]
    n_rounds = 0
    n_perturbations = 0
    recent: list[str] = []

    for _ in range(max_iter):
        if mode == "sync":
            M_new, converged = sync_brd_step(M)
            if M_new is None:
                idxs = random.sample(range(n * n), perturb_k)
                for idx in idxs:
                    ii, jj = divmod(idx, n)
                    M[ii][jj] *= -1
                n_perturbations += 1
                recent.clear()
                n_rounds += 1
                continue
            M = M_new
            n_rounds += 1
            if converged:
                break
            key = _matrix_key(M)
            if key in recent:
                idxs = random.sample(range(n * n), perturb_k)
                for idx in idxs:
                    ii, jj = divmod(idx, n)
                    M[ii][jj] *= -1
                n_perturbations += 1
                recent.clear()
            else:
                recent.append(key)
                if len(recent) > cycle_window:
                    recent.pop(0)
        else:  # async reference
            M, n_flips = async_brd_pass(M, n)
            n_rounds += 1
            if n_flips == 0:
                break

    return M, n_rounds, n_perturbations


def _matrix_key(M: list[list[int]]) -> str:
    """Cheap hashable key for cycle detection (first 50 entries)."""
    flat = [v for row in M for v in row]
    return "".join("1" if v > 0 else "0" for v in flat[:50])


# ---------------------------------------------------------------------------
# Multi-start search
# ---------------------------------------------------------------------------


def brd_multistart_search(
    n: int = N23,
    time_limit_s: float = 780.0,
    seed: int = 42,
    mode: str = "greedy_sm",
    verbose: bool = True,
) -> tuple[Optional[list[list[int]]], int, dict]:
    """
    Run BRD from many random ±1 starts within the time budget.

    Returns (best_matrix, best_abs_det, stats).
    """
    random.seed(seed)
    np.random.seed(seed)

    best_det = 0
    best_M: Optional[list[list[int]]] = None
    n_trials = 0
    det_histogram: dict[int, int] = {}
    t_start = time.time()

    while time.time() - t_start < time_limit_s:
        M_init = [[random.choice([-1, 1]) for _ in range(n)] for _ in range(n)]

        M_final, n_rounds, _ = brd_optimize(M_init, mode=mode, n=n)

        det_val = integer_determinant(M_final)
        abs_det = abs(det_val)

        bucket = int(abs_det / ORRICK_ABS_DET_H23 * 10)
        det_histogram[bucket] = det_histogram.get(bucket, 0) + 1

        if abs_det > best_det:
            best_det = abs_det
            best_M = [row[:] for row in M_final]
            if verbose:
                ratio = best_det / ORRICK_ABS_DET_H23
                elapsed = time.time() - t_start
                print(
                    f"  t={elapsed:.1f}s trial={n_trials}: "
                    f"|det|={best_det} ({ratio:.4f}×Orrick) "
                    f"passes={n_rounds}"
                )

        n_trials += 1
        if verbose and n_trials % 10000 == 0:
            elapsed = time.time() - t_start
            rate = n_trials / elapsed
            print(
                f"  t={elapsed:.1f}s: {n_trials} trials @ {rate:.0f}/s, "
                f"best={best_det/ORRICK_ABS_DET_H23:.4f}×Orrick"
            )

    elapsed = time.time() - t_start
    stats = {
        "n_trials": n_trials,
        "elapsed_s": elapsed,
        "trials_per_second": n_trials / elapsed,
        "best_abs_det": best_det,
        "best_ratio_to_orrick": best_det / ORRICK_ABS_DET_H23,
        "det_histogram": det_histogram,
        "mode": mode,
        "seed": seed,
        "n": n,
    }
    return best_M, best_det, stats


# ---------------------------------------------------------------------------
# Triple-verify helper
# ---------------------------------------------------------------------------


def triple_verify(M: list[list[int]]) -> tuple[int, int, bool]:
    """
    Return (det_M, det_MT, match) using integer_determinant on both M and M^T.
    """
    det_M = integer_determinant(M)
    MT = [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]
    det_MT = integer_determinant(MT)
    return det_M, det_MT, abs(det_M) == abs(det_MT)
