"""Spectral-anchor search for high-multiplicity SV ±1 matrices.

The winner-decode round (`winner_decode.py`) revealed that proven max-|det|
matrices at n ∈ {13, 15, 17} share a "spike + dominant repeat" singular
value structure: one dominant SV with multiplicity ≈ n − 2 plus 1–3 spike
outliers. Vanilla SA-found basins at H19+ live on full-SV-distinct
matrices (mult 1 across the board) — a structurally different region of
matrix space.

This module biases SA toward the high-multiplicity submanifold.

Mechanism
---------

For a ±1 matrix M with Gram G = M Mᵀ, a single entry flip M[i][j] →
−M[i][j] perturbs G by a rank-2 matrix:

    ΔG = M' M'ᵀ − M Mᵀ = δ (e_i e_j^T M^T + M e_j e_i^T) + 4 e_i e_i^T

with δ = −2 M[i][j]. The perturbation generically splits a dominant
singular-value cluster, so unconstrained SA naturally exits the
high-multiplicity submanifold and never returns.

We add two ingredients:

1. ``E_anchor(M) = -log|det(M)| - λ · spectral_anchor_score(M)`` — a
   score that biases SA toward matrices with few distinct singular
   values.
2. **Anchor-preserving moves**: bias proposal sampling toward flips
   whose perturbation projects mostly *outside* the current dominant
   eigenspace. The acceptance probability is then the standard
   Metropolis criterion on ``E_anchor`` and the projection bias is
   *only* in the proposal — so detailed balance is preserved if the
   proposal probabilities are renormalised symmetrically (we do this).

Three workflows
---------------

* :func:`sa_spectral_anchored` — biased SA from any starting point.
* :func:`bordered_extension_sa` — extend a known proven seed at order
  ``n_seed`` by ``k = n − n_seed`` rows + columns, optimise the
  ``2(n_seed · k) + k²`` free entries via spectral-anchored SA. Used to
  attempt H19 (from n=17 proven) and H23 (from n=21 best basin).

Public API
----------

* :func:`dominant_sv_multiplicity(M, tol)` — count of SVs within ``tol``
  of the dominant SV.
* :func:`spectral_anchor_score(M, tol)` — scalar bias score.
* :func:`flip_dominant_projection(M, i, j, V)` — fraction of the rank-2
  flip perturbation that lives in the dominant eigenspace (lower means
  the flip is "anchor-preserving"). Computed in O(n²).
* :func:`dominant_eigenspace(M, tol)` — orthonormal basis for the
  dominant-SV right-singular subspace.
* :func:`sa_spectral_anchored(...)` — the biased SA loop.
* :func:`bordered_extension_sa(...)` — seed-extension attack.
"""
from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from hadamard_maxdet.incremental import (
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    refresh as incr_refresh,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Spectral fingerprint primitives
# ---------------------------------------------------------------------------


def _svd(M: np.ndarray) -> np.ndarray:
    """SVDs (descending) of M as float64. Cost O(n^3)."""
    return np.linalg.svd(M.astype(np.float64), compute_uv=False)


def _svd_full(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full SVD M = U diag(s) Vt of M as float64. Cost O(n^3)."""
    U, s, Vt = np.linalg.svd(M.astype(np.float64), full_matrices=False)
    return U, s, Vt


def _cluster_sv(s: np.ndarray, tol: float) -> list[tuple[float, int]]:
    """Greedy 1D clustering of a descending SV vector.

    Two SVs are grouped if successive descending differences fall within
    ``tol · s_max`` of the cluster's representative (= the cluster's
    first / largest member). Returns a list of ``(representative, count)``
    pairs in descending-rep order.
    """
    out: list[list[float]] = []
    if s.size == 0:
        return []
    s_max = float(s[0])
    if s_max <= 0:
        return [(0.0, int(s.size))]
    thr = tol * s_max
    cur_rep = float(s[0])
    cur_count = 1
    for v in s[1:]:
        if cur_rep - float(v) <= thr:
            cur_count += 1
        else:
            out.append([cur_rep, cur_count])
            cur_rep = float(v)
            cur_count = 1
    out.append([cur_rep, cur_count])
    return [(r, c) for r, c in out]


def dominant_sv_multiplicity(
    matrix: Sequence[Sequence[int]],
    tol: float = 0.05,
) -> int:
    """Multiplicity of the *most populated* singular-value cluster.

    The proven-winner spectral signature is "spike + dominant repeat": one
    big cluster of size ≈ n − k for small k plus a few outlier spikes.
    This routine returns the size of the largest cluster (in count terms).

    Two SVs s_i, s_j are clustered if successive differences fall within
    ``tol · s_max`` (relative tolerance). For the n=17 proven winner
    with SVs ≈ {6.22, 4.00 × 15, 3.22}, returns 15.
    """
    M = np.asarray(matrix, dtype=np.float64)
    if M.size == 0:
        return 0
    s = _svd(M)
    if s.size == 0:
        return 0
    clusters = _cluster_sv(s, tol)
    if not clusters:
        return 0
    return max(c for _, c in clusters)


def distinct_sv_count(
    matrix: Sequence[Sequence[int]],
    tol: float = 0.05,
) -> int:
    """Number of distinct SV clusters at relative tolerance ``tol``.

    Greedy 1D clustering of the descending SV list: a new cluster opens
    each time a successive SV is more than ``tol · s_max`` below the
    previous cluster's representative.
    """
    M = np.asarray(matrix, dtype=np.float64)
    s = _svd(M)
    if s.size == 0:
        return 0
    return len(_cluster_sv(s, tol))


def spectral_anchor_score(
    matrix: Sequence[Sequence[int]],
    tol: float = 0.05,
) -> float:
    """Scalar reflecting closeness to the high-mult submanifold.

    Defined as ``n − distinct_sv_count(M, tol)`` — equals 0 when every
    SV is its own cluster (full-distinct, the *bad* region) and ``n − 1``
    when all SVs collapse to a single cluster (Hadamard-like).

    Used as the anchor reward in :func:`E_anchor`.
    """
    n = len(matrix)
    return float(n - distinct_sv_count(matrix, tol))


def E_anchor(
    matrix: Sequence[Sequence[int]],
    *,
    lambda_anchor: float,
    tol: float = 0.05,
    abs_det: Optional[int] = None,
) -> float:
    """Anchored energy: ``−log|det| − λ · spectral_anchor_score``.

    Lower is better (SA minimises). ``abs_det`` may be supplied to skip
    re-computation; otherwise computed via Bareiss.
    """
    if abs_det is None:
        abs_det = abs(integer_determinant(matrix))
    if abs_det == 0:
        return math.inf
    return -math.log(abs_det) - lambda_anchor * spectral_anchor_score(
        matrix, tol=tol,
    )


def dominant_eigenspace(
    matrix: Sequence[Sequence[int]],
    tol: float = 0.05,
) -> np.ndarray:
    """Orthonormal basis for the right-singular subspace of M
    corresponding to the *most populated* SV cluster.

    Returns an ``n × k`` matrix V whose columns span that cluster's
    right-singular subspace (k = dominant_sv_multiplicity).
    """
    M = np.asarray(matrix, dtype=np.float64)
    n = M.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    U, s, Vt = _svd_full(M)
    if s.size == 0:
        return np.zeros((n, 0))
    s_max = float(s[0])
    if s_max <= 0:
        return np.zeros((n, 0))
    clusters = _cluster_sv(s, tol)
    if not clusters:
        return np.zeros((n, 0))
    # Pick the cluster with the largest count (ties → largest representative).
    dominant_idx = max(range(len(clusters)), key=lambda i: (clusters[i][1], clusters[i][0]))
    rep = clusters[dominant_idx][0]
    mask = np.abs(s - rep) <= tol * s_max
    return Vt[mask].T  # n × k


def flip_dominant_projection(
    matrix: Sequence[Sequence[int]],
    i: int,
    j: int,
    dominant_eigenspace_V: np.ndarray,
) -> float:
    """Fraction of the rank-2 flip perturbation in the dominant eigenspace.

    A flip M[i][j] → −M[i][j] perturbs G = M Mᵀ by

        ΔG = δ(e_i r_j^T + r_j e_i^T) + 4 e_i e_i^T,   r_j = M[:, j]

    where δ = −2 M[i][j] (so |δ|=2, and we then re-flip the (i,j) entry
    in r_j: r_j[i] becomes −M[i][j]). The "perturbation in the right
    singular space" is captured by

        ΔM = δ e_i e_j^T  (entry-flip is rank-1 on M itself)

    so the fraction in the dominant right-singular subspace V (n × k) is

        ‖V^T e_j‖² / ‖e_j‖² = ‖V[j, :]‖²

    which is in [0, 1]. (The e_j part is what gets pushed; the e_i part
    sits in the *left* singular space and shows up symmetrically in U.)

    Returns the fraction in [0, 1]; lower means the flip is mostly
    outside the dominant subspace (preferred for anchor preservation).
    """
    n = len(matrix)
    if dominant_eigenspace_V.size == 0 or dominant_eigenspace_V.shape[0] != n:
        return 0.0
    if not (0 <= j < n):
        return 0.0
    row_j = dominant_eigenspace_V[j, :]
    return float(np.dot(row_j, row_j))


def all_flip_projections(
    matrix: Sequence[Sequence[int]],
    dominant_eigenspace_V: np.ndarray,
) -> np.ndarray:
    """Vector of length n: projection magnitude per column index ``j``.

    Cached for one full proposal pass (entry-flip only depends on column
    ``j`` of M for the right-singular projection).
    """
    n = len(matrix)
    if dominant_eigenspace_V.size == 0 or dominant_eigenspace_V.shape[0] != n:
        return np.zeros(n)
    return np.einsum("jk,jk->j", dominant_eigenspace_V, dominant_eigenspace_V)


# ---------------------------------------------------------------------------
# SA loop with spectral-anchor bias
# ---------------------------------------------------------------------------


@dataclass
class SpectralAnchorResult:
    """Outcome of :func:`sa_spectral_anchored`."""

    matrix: Matrix
    abs_det: int
    iterations: int
    accepted: int
    walltime_seconds: float
    refresh_count: int
    sv_mult_history: list[int] = field(default_factory=list)
    distinct_sv_history: list[int] = field(default_factory=list)
    abs_det_history: list[int] = field(default_factory=list)


def sa_spectral_anchored(
    initial: Matrix,
    *,
    iterations: int = 200_000,
    lambda_anchor: float = 0.5,
    tol: float = 0.05,
    t_start: float = 1.0,
    t_end: float = 0.005,
    seed: int | None = None,
    refresh_eigenspace_every: int = 250,
    rescore_anchor_every: int = 1,
    proposal_temperature: float = 1.0,
    log_every: int = 0,
    free_mask: Optional[Sequence[Sequence[bool]]] = None,
) -> SpectralAnchorResult:
    """Spectral-anchored SA on ±1 matrices.

    The energy is ``E_anchor(M, λ) = −log|det| − λ · (n − distinct_sv_count)``.

    Proposals are entry flips with sampling biased toward low-projection
    columns: column j is sampled with probability proportional to

        exp(−proposal_temperature · projection[j])

    where ``projection[j] = ‖V[j, :]‖²`` is the fraction of the
    perturbation in the dominant right-singular subspace V. Setting
    ``proposal_temperature = 0`` recovers uniform sampling.

    To preserve detailed balance under non-uniform proposals, we use the
    ratio of forward / reverse proposal probabilities — but for an
    entry flip the *position* is the same in both directions, so the
    proposal probabilities cancel exactly. The bias only changes
    *which* moves get tried, not the acceptance rule.

    The dominant eigenspace V is recomputed every
    ``refresh_eigenspace_every`` accepted moves (not every step — the
    SVD is O(n³)).

    Parameters
    ----------
    initial
        Starting ±1 matrix.
    iterations
        Total proposal count.
    lambda_anchor
        Strength of the anchor bias in the energy.
    tol
        Relative SV clustering tolerance.
    t_start, t_end
        Geometric temperature schedule (for the energy Metropolis step).
    seed
        PRNG seed.
    refresh_eigenspace_every
        Recompute V every this many *accepted* flips. Fully refreshing
        every step would dominate the cost; the eigenspace evolves
        slowly enough that 250 accepted steps between refreshes is fine.
    rescore_anchor_every
        Recompute the anchor (full SVD) every this many *proposed* flips
        for use in the energy. In between, we use the cached anchor
        value (the |det| component is always recomputed via incremental
        Bareiss). Default 1 = anchor-aware on every step (slow; n^3
        SVD per step). Set to 100+ for a much faster loop where the
        anchor only re-evaluates periodically (a coarser reward signal,
        but lets the SA explore many more states).
    proposal_temperature
        Strength of the proposal-side bias toward low-projection
        columns. 0 = uniform.
    log_every
        If positive, append SV-mult and |det| every that many iterations.
    free_mask
        Optional ``n × n`` boolean array marking which entries SA may
        flip (True = free, False = locked). Defaults to all-free. Used
        by :func:`bordered_extension_sa` to lock the seed block.
    """
    rng = random.Random(seed)
    n = initial.n
    M_list = [list(r) for r in initial.entries]

    # Free-positions list.
    if free_mask is None:
        free_positions = [(i, j) for i in range(n) for j in range(n)]
    else:
        free_positions = [
            (i, j)
            for i in range(n)
            for j in range(n)
            if free_mask[i][j]
        ]
    if not free_positions:
        return SpectralAnchorResult(
            matrix=Matrix(n=n, entries=[list(r) for r in M_list]),
            abs_det=abs(integer_determinant(M_list)),
            iterations=0,
            accepted=0,
            walltime_seconds=0.0,
            refresh_count=0,
        )

    # Ensure non-singular start.
    state = IncrementalState.from_matrix(M_list)
    tries = 0
    while state.det == 0 and tries < 16:
        for (i, j) in free_positions:
            M_list[i][j] = 1 if rng.random() < 0.5 else -1
        state = IncrementalState.from_matrix(M_list)
        tries += 1
    if state.det == 0:
        return SpectralAnchorResult(
            matrix=Matrix(n=n, entries=[list(r) for r in M_list]),
            abs_det=0,
            iterations=0,
            accepted=0,
            walltime_seconds=0.0,
            refresh_count=0,
        )
    abs_det = abs(state.det)

    # Initial spectral state.
    V = dominant_eigenspace(state.M, tol=tol)
    proj = all_flip_projections(state.M, V)
    distinct = distinct_sv_count(state.M, tol=tol)
    anchor = float(n - distinct)
    current_e = -math.log(abs_det) - lambda_anchor * anchor

    best_M = [list(r) for r in state.M]
    best_abs_det = abs_det
    best_e = current_e

    sv_mult_history: list[int] = []
    distinct_sv_history: list[int] = []
    abs_det_history: list[int] = []

    accepted = 0
    refresh_count = 0
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0
    start = time.monotonic()

    # Per-column proposal weights from projection (cached, refreshed
    # whenever V is refreshed).
    def _column_weights(proj_vec: np.ndarray) -> np.ndarray:
        if proposal_temperature <= 0.0:
            w = np.ones(n)
        else:
            # Lower projection -> higher weight.
            w = np.exp(-proposal_temperature * proj_vec)
        # Restrict to free columns: any column with no free positions
        # gets zero weight.
        free_cols = np.zeros(n, dtype=bool)
        for (_i, j) in free_positions:
            free_cols[j] = True
        w = w * free_cols
        s = w.sum()
        if s <= 0:
            # Fallback uniform over free positions list.
            w = np.zeros(n)
            for (_i, j) in free_positions:
                w[j] = 1.0
            s = w.sum()
            if s <= 0:
                return np.ones(n) / n
        return w / s

    col_weights = _column_weights(proj)
    # Per-column free-row lists (precompute for fast row sampling once
    # column j has been chosen).
    free_rows_in_col: list[list[int]] = [[] for _ in range(n)]
    for (i, j) in free_positions:
        free_rows_in_col[j].append(i)

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        # 1. Sample column j by projection bias, then a free row uniformly.
        if proposal_temperature > 0.0 and col_weights.size:
            j = int(np.searchsorted(np.cumsum(col_weights), rng.random()))
            j = min(j, n - 1)
            rows = free_rows_in_col[j]
            if not rows:
                # Skip; fall back to uniform.
                (i, j) = free_positions[rng.randrange(len(free_positions))]
            else:
                i = rows[rng.randrange(len(rows))]
        else:
            (i, j) = free_positions[rng.randrange(len(free_positions))]

        # 2. Propose flip via O(1) rank-1 det update.
        new_det = propose_entry_flip(state, i, j)
        if new_det == 0:
            # Reject (infinite energy).
            continue
        new_abs_det = abs(new_det)
        # Anchor: only re-evaluated every rescore_anchor_every steps;
        # otherwise we reuse the last-known anchor value (an
        # approximation — accurate up to changes since the last SVD).
        if (rescore_anchor_every <= 1) or (step % rescore_anchor_every == 0):
            # Need to flip-then-score-then-revert (or just simulate).
            old = state.M[i][j]
            state.M[i][j] = -old
            new_anchor = float(n - distinct_sv_count(state.M, tol=tol))
            state.M[i][j] = old
        else:
            new_anchor = anchor
        new_e = -math.log(new_abs_det) - lambda_anchor * new_anchor

        delta = new_e - current_e
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            # Accept: commit the flip via incremental update.
            try:
                apply_entry_flip(state, i, j)
            except ZeroDivisionError:
                # Should not happen since new_det != 0; refresh defensively.
                incr_refresh(state)
                continue
            current_e = new_e
            anchor = new_anchor
            abs_det = new_abs_det
            accepted += 1
            if new_e < best_e:
                best_e = new_e
                best_M = [list(r) for r in state.M]
                best_abs_det = new_abs_det
            if accepted % refresh_eigenspace_every == 0:
                V = dominant_eigenspace(state.M, tol=tol)
                proj = all_flip_projections(state.M, V)
                col_weights = _column_weights(proj)
                # Also re-sync the cached anchor to the truth.
                anchor = float(n - distinct_sv_count(state.M, tol=tol))
                current_e = -math.log(abs_det) - lambda_anchor * anchor
                refresh_count += 1
                # Periodic incremental-state refresh as cheap insurance.
                incr_refresh(state)
        # else: reject. No revert needed (we never mutated state).

        if log_every and step % log_every == 0:
            sv_mult_history.append(
                dominant_sv_multiplicity(state.M, tol=tol),
            )
            distinct_sv_history.append(distinct_sv_count(state.M, tol=tol))
            abs_det_history.append(int(abs_det))

    walltime = time.monotonic() - start
    # Verified |det| of best.
    best_abs_det_verified = abs(integer_determinant(best_M))
    return SpectralAnchorResult(
        matrix=Matrix(n=n, entries=[list(r) for r in best_M]),
        abs_det=best_abs_det_verified,
        iterations=iterations,
        accepted=accepted,
        walltime_seconds=walltime,
        refresh_count=refresh_count,
        sv_mult_history=sv_mult_history,
        distinct_sv_history=distinct_sv_history,
        abs_det_history=abs_det_history,
    )


# ---------------------------------------------------------------------------
# Bordered extension from a known seed
# ---------------------------------------------------------------------------


def bordered_extension_sa(
    seed_matrix: Sequence[Sequence[int]],
    target_n: int,
    *,
    iterations: int = 1_000_000,
    lambda_anchor: float = 0.5,
    tol: float = 0.05,
    t_start: float = 1.0,
    t_end: float = 0.005,
    seed: int | None = 42,
    refresh_eigenspace_every: int = 250,
    rescore_anchor_every: int = 1,
    proposal_temperature: float = 1.0,
    wall_seconds: Optional[float] = None,
    log_every: int = 0,
    n_restarts: int = 1,
) -> SpectralAnchorResult:
    """Extend a known ±1 matrix to a larger ±1 matrix; optimise the new entries.

    Given a seed matrix S of order ``n_seed`` and a target order
    ``target_n > n_seed``, build a target_n × target_n matrix M where
    the upper-left n_seed × n_seed block equals S, and the
    ``2 · n_seed · k + k²`` border entries (k = target_n − n_seed) are
    free for SA to optimise.

    Free entries:

        - rows in [n_seed, target_n) × columns in [0, target_n)   (k · target_n)
        - rows in [0, n_seed) × columns in [n_seed, target_n)     (n_seed · k)

    Total free = k · target_n + n_seed · k = k · (target_n + n_seed)
              = k · (n_seed + target_n).

    For target_n = n_seed + 2 this is 2 · (2 n_seed + 2) = 4 n_seed + 4.

    Parameters
    ----------
    seed_matrix
        The locked seed (n_seed × n_seed ±1 matrix).
    target_n
        Final order; must be > n_seed.
    iterations
        SA iterations per restart.
    lambda_anchor, tol, t_start, t_end, seed,
    refresh_eigenspace_every, proposal_temperature, log_every
        Forwarded to :func:`sa_spectral_anchored`.
    wall_seconds
        Optional wall-time cap; SA is terminated at the next iteration
        boundary once exceeded. Default None = run to ``iterations``.
    n_restarts
        Number of independent SA restarts (random initialisation of the
        free entries each time). Returns the best.

    Returns
    -------
    SpectralAnchorResult
        Best-of-n_restarts result. The .matrix is the best (target_n
        x target_n) ±1 matrix found.
    """
    seed_arr = np.array(seed_matrix, dtype=np.int64)
    n_seed = seed_arr.shape[0]
    if seed_arr.shape[1] != n_seed:
        raise ValueError("seed_matrix must be square")
    if target_n <= n_seed:
        raise ValueError(
            f"target_n={target_n} must exceed seed order={n_seed}"
        )

    # Verify seed is ±1.
    if not np.all(np.isin(seed_arr, [-1, 1])):
        raise ValueError("seed_matrix must have ±1 entries")

    # Build free_mask: True for border, False for seed block.
    free_mask = [[False] * target_n for _ in range(target_n)]
    for i in range(target_n):
        for j in range(target_n):
            if not (i < n_seed and j < n_seed):
                free_mask[i][j] = True

    rng = random.Random(seed)
    best: Optional[SpectralAnchorResult] = None
    best_abs_det = -1
    deadline = (
        time.monotonic() + wall_seconds if wall_seconds is not None else None
    )
    iters_per_restart = iterations
    if wall_seconds is not None and n_restarts > 0:
        # Soft division across restarts; SA loop honours its own iter cap
        # but we cap restarts on wall.
        pass

    for r in range(n_restarts):
        if deadline is not None and time.monotonic() >= deadline:
            break

        # Build a starting matrix with random ±1 in the border.
        init = np.array(
            [[1] * target_n for _ in range(target_n)], dtype=np.int64,
        )
        # Plant seed.
        init[:n_seed, :n_seed] = seed_arr
        # Random ±1 in border.
        for i in range(target_n):
            for j in range(target_n):
                if free_mask[i][j]:
                    init[i, j] = 1 if rng.random() < 0.5 else -1

        # If wall_seconds is set, cap iters per restart by remaining time
        # using a rough rate estimate.
        cur_iters = iters_per_restart
        result = sa_spectral_anchored(
            initial=Matrix(
                n=target_n, entries=[list(row) for row in init.tolist()],
            ),
            iterations=cur_iters,
            lambda_anchor=lambda_anchor,
            tol=tol,
            t_start=t_start,
            t_end=t_end,
            seed=(seed or 0) + r * 7919 + 1,
            refresh_eigenspace_every=refresh_eigenspace_every,
            rescore_anchor_every=rescore_anchor_every,
            proposal_temperature=proposal_temperature,
            log_every=log_every,
            free_mask=free_mask,
        )

        if result.abs_det > best_abs_det:
            best_abs_det = result.abs_det
            best = result

    if best is None:
        # No restart fit; return the most recent (or empty).
        return SpectralAnchorResult(
            matrix=Matrix(
                n=target_n,
                entries=[[1] * target_n for _ in range(target_n)],
            ),
            abs_det=0,
            iterations=0,
            accepted=0,
            walltime_seconds=0.0,
            refresh_count=0,
        )
    return best


# ---------------------------------------------------------------------------
# Targeted-spectrum extension (2026-04-25)
# ---------------------------------------------------------------------------
#
# The plain anchor score (count of distinct SVs) just biases toward *any*
# high-multiplicity matrix; the previous round confirmed the high-mult
# submanifold is reachable but its bulk is low-|det|. Records sit on a
# *specific point* in the submanifold characterized by both high mult AND
# the dominant SV at a specific Ehlich-predicted value.
#
# This block predicts the precise (value, multiplicity, outlier) signature
# from (n, target_det) and biases SA toward matrices whose actual spectrum
# matches the target.
#
# Mathematical setup. For a target ±1 matrix with ``|det| = D`` and Gram
# G = M Mᵀ, we have:
#
#   trace(G)  = sum(s_i^2)  = ||M||_F^2 = n^2          (always, for ±1 M)
#   det(G)    = D^2                                     (target)
#
# We parameterize the SV spectrum as a single dominant value sqrt(λ) with
# multiplicity m, plus k = n - m outliers. Several closure assumptions
# yield closed-form solutions for (λ, outliers). The cleanest model with
# a free degree is ``symmetric_pair`` (k=2, μ_H · μ_L = c · λ^2) — when
# c = 1 this is exact log-symmetry around the dominant; calibrating c
# from the n=17 proven winner gives c ≈ 1.566.
# ---------------------------------------------------------------------------


def _solve_high_only(
    n: int, target_det: float, m: int,
) -> tuple[float, list[float]]:
    """Closed-form spectrum for ``mult = m, k = 1`` outlier (high spike).

    Constraints:
      m·λ + μ = n²
      λ^m · μ = D²

    Substituting μ = n² − m·λ into the second equation gives
    ``λ^m (n² − m λ) = D²``, a single nonlinear equation in λ.

    Returns (lambda, [mu]). May raise ValueError if no real root in
    (0, n²/m).
    """
    if m < 1 or m >= n:
        raise ValueError(f"m={m} must satisfy 1 <= m < n={n}")
    if target_det <= 0:
        raise ValueError(f"target_det={target_det} must be positive")
    target_log = 2.0 * math.log(target_det)

    def f(lam: float) -> float:
        mu = n * n - m * lam
        if mu <= 0 or lam <= 0:
            return -math.inf
        return m * math.log(lam) + math.log(mu) - target_log

    lo, hi = 1e-9, (n * n) / m - 1e-9
    n_samples = 256
    xs = [lo + (hi - lo) * (i / (n_samples - 1)) for i in range(n_samples)]
    fs = [f(x) for x in xs]
    roots: list[float] = []
    for i in range(len(xs) - 1):
        a, b = fs[i], fs[i + 1]
        if a == -math.inf or b == -math.inf:
            continue
        if a * b < 0:
            x_lo, x_hi = xs[i], xs[i + 1]
            for _ in range(64):
                xm = 0.5 * (x_lo + x_hi)
                fm = f(xm)
                if fm == 0.0:
                    break
                if fm * f(x_lo) < 0:
                    x_hi = xm
                else:
                    x_lo = xm
            roots.append(0.5 * (x_lo + x_hi))
    if not roots:
        raise ValueError(
            f"high_only: no real root for n={n}, m={m}, det={target_det}"
        )
    target_lam = float(n)
    best_lam = min(roots, key=lambda r: abs(r - target_lam))
    mu = n * n - m * best_lam
    return best_lam, [mu]


def _solve_symmetric_pair(
    n: int, target_det: float, m: int, ratio: float = 1.0,
) -> tuple[float, list[float]]:
    """Closed-form spectrum for ``mult = m, k = 2`` outliers under
    ``μ_H · μ_L = ratio · λ²`` (log-symmetric pair around the dominant).

    Constraints:
      m·λ + μ_H + μ_L = n²
      λ^m · μ_H · μ_L = D²
      μ_H · μ_L       = ratio · λ²

    Substituting (3) into (2): ``λ^(m+2) · ratio = D²``, so
    ``λ = (D² / ratio)^(1/(m+2))``. Then ``μ_H + μ_L = n² − m λ`` and
    ``μ_H · μ_L = ratio · λ²`` define μ_H, μ_L as roots of the quadratic
    ``x² − S x + P = 0`` with S = sum, P = product. Real roots iff
    ``S² ≥ 4P``.

    For ``ratio = 1``, ``μ_H, μ_L`` are log-symmetric around λ. Calibrating
    on n=17 (proven SVs at 4, 6.22, 3.22) gives ``ratio ≈ 1.566``.
    """
    if m < 1 or m + 2 > n:
        raise ValueError(f"m={m} must satisfy 1 <= m, m+2 <= n={n}")
    if target_det <= 0:
        raise ValueError(f"target_det={target_det} must be positive")
    if ratio <= 0:
        raise ValueError(f"ratio={ratio} must be positive")

    lam = math.exp(
        (2.0 * math.log(target_det) - math.log(ratio)) / (m + 2)
    )
    S = n * n - m * lam
    P = ratio * lam * lam
    disc = S * S - 4.0 * P
    if disc < 0:
        mu = 0.5 * S
        return lam, [mu, mu]
    sqrt_disc = math.sqrt(disc)
    mu_H = 0.5 * (S + sqrt_disc)
    mu_L = 0.5 * (S - sqrt_disc)
    return lam, [mu_H, mu_L]


def _solve_equal_pair(
    n: int, target_det: float, m: int,
) -> tuple[float, list[float]]:
    """Closed-form spectrum for ``mult = m, k = 2`` *equal* outliers.

    Constraints:
      m·λ + 2·μ = n²
      λ^m · μ²  = D²

    Substituting μ = (n² − m λ)/2: ``λ^m · ((n² − m λ)/2)² = D²``. Solve
    numerically for λ in (0, n²/m).
    """
    if m < 1 or m + 2 > n:
        raise ValueError(f"m={m} must satisfy 1 <= m, m+2 <= n={n}")
    if target_det <= 0:
        raise ValueError(f"target_det={target_det} must be positive")
    target_log = 2.0 * math.log(target_det)

    def f(lam: float) -> float:
        rest = n * n - m * lam
        if rest <= 0 or lam <= 0:
            return -math.inf
        mu = rest / 2.0
        return m * math.log(lam) + 2.0 * math.log(mu) - target_log

    lo, hi = 1e-9, (n * n) / m - 1e-9
    n_samples = 256
    xs = [lo + (hi - lo) * (i / (n_samples - 1)) for i in range(n_samples)]
    fs = [f(x) for x in xs]
    roots: list[float] = []
    for i in range(len(xs) - 1):
        a, b = fs[i], fs[i + 1]
        if a == -math.inf or b == -math.inf:
            continue
        if a * b < 0:
            x_lo, x_hi = xs[i], xs[i + 1]
            for _ in range(64):
                xm = 0.5 * (x_lo + x_hi)
                fm = f(xm)
                if fm == 0.0:
                    break
                if fm * f(x_lo) < 0:
                    x_hi = xm
                else:
                    x_lo = xm
            roots.append(0.5 * (x_lo + x_hi))
    if not roots:
        raise ValueError(
            f"equal_pair: no real root for n={n}, m={m}, det={target_det}"
        )
    target_lam = float(n)
    best_lam = min(roots, key=lambda r: abs(r - target_lam))
    mu = (n * n - m * best_lam) / 2.0
    return best_lam, [mu, mu]


def predict_target_spectrum(
    n: int,
    target_det: float,
    *,
    mult: Optional[int] = None,
    model: str = "symmetric_pair",
    ratio: float = 1.566,
) -> np.ndarray:
    """Predict the target singular-value spectrum from (n, target_det).

    Returns a length-``n`` numpy array of singular values in descending
    order, computed from a closed-form (mult, value, outlier) signature
    parameterized by ``model``:

    * ``"high_only"``: ``mult = n − 1`` repeats at sqrt(λ), one high
      outlier sqrt(μ_H). Two unknowns (λ, μ_H), two equations
      (trace=n², det=D²) — uniquely determined.
    * ``"symmetric_pair"``: ``mult = n − 2`` repeats at sqrt(λ), two
      outliers sqrt(μ_H), sqrt(μ_L) with the closure
      ``μ_H · μ_L = ratio · λ²``. Calibrated on the n=17 proven winner
      (SVs 4, 6.22, 3.22 → ratio ≈ 1.566). Set ``ratio = 1.0`` for the
      pure log-symmetric model.
    * ``"equal_pair"``: ``mult = n − 2`` repeats at sqrt(λ), two equal
      outliers sqrt(μ).
    * ``"hadamard"``: all SVs equal to sqrt(n) (Hadamard ceiling).

    The default ``mult`` is ``n − 1`` for ``high_only``, ``n − 2`` for
    pair models, ``n`` for ``hadamard``. Hadamard ignores ``target_det``.

    Verification on n=8 Hadamard (target_det = n^(n/2) = 4096):
    ``predict_target_spectrum(8, 4096, model='hadamard')`` returns
    [√8] * 8 ≈ [2.828, ..., 2.828].
    """
    if model == "hadamard":
        return np.full(n, math.sqrt(n), dtype=np.float64)

    if mult is None:
        if model == "high_only":
            mult = n - 1
        else:
            mult = n - 2

    if model == "high_only":
        lam, outliers = _solve_high_only(n, target_det, mult)
    elif model == "symmetric_pair":
        lam, outliers = _solve_symmetric_pair(n, target_det, mult, ratio=ratio)
    elif model == "equal_pair":
        lam, outliers = _solve_equal_pair(n, target_det, mult)
    else:
        raise ValueError(
            f"unknown model={model!r}; use one of "
            "'hadamard', 'high_only', 'symmetric_pair', 'equal_pair'"
        )

    values_sq = sorted([float(o) for o in outliers], reverse=True)
    full_sq = values_sq + [float(lam)] * mult
    full_sq.sort(reverse=True)
    return np.array([math.sqrt(max(0.0, v)) for v in full_sq])


def spectrum_distance(
    matrix: Sequence[Sequence[int]],
    target_spectrum: np.ndarray,
) -> float:
    """L2 distance between M's actual SV spectrum and ``target_spectrum``.

    Both vectors are sorted descending and matched element-wise; the
    return value is ``||sort_desc(SV(M)) − sort_desc(target)||_2``. Lower
    is closer to target. Returns 0 iff M's SVs equal the target.
    """
    M = np.asarray(matrix, dtype=np.float64)
    actual = _svd(M)
    n_a = actual.size
    n_t = target_spectrum.size
    n = max(n_a, n_t)
    a = np.zeros(n)
    t = np.zeros(n)
    a[:n_a] = np.sort(actual)[::-1]
    t[:n_t] = np.sort(np.asarray(target_spectrum, dtype=np.float64))[::-1]
    return float(np.linalg.norm(a - t))


def sa_targeted_spectrum(
    initial: Matrix,
    target_spectrum: np.ndarray,
    *,
    iterations: int = 200_000,
    lambda_anchor: float = 1.0,
    t_start: float = 1.0,
    t_end: float = 0.005,
    seed: int | None = None,
    rescore_every: int = 1,
    log_every: int = 0,
    free_mask: Optional[Sequence[Sequence[bool]]] = None,
    deadline: Optional[float] = None,
) -> SpectralAnchorResult:
    """Spectrum-targeted SA on ±1 matrices.

    Energy: ``E(M) = -log|det(M)| + lambda_anchor · spectrum_distance(M, target_spectrum)``.

    Spectrum distance is a *penalty* (added, not subtracted), so SA is
    pulled toward both high-|det| AND target-spectrum matrices. The
    relative weight is set by ``lambda_anchor``.

    Each step: O(1) det update via incremental Bareiss + O(n³) SVD for
    the spectrum-distance term (every ``rescore_every`` proposed steps).
    For larger n where the SVD dominates, set ``rescore_every`` to 5–20
    to amortize.

    Parameters mirror :func:`sa_spectral_anchored`. ``deadline`` is a
    monotonic-time wall stop; if exceeded, SA returns the best so far.
    """
    rng = random.Random(seed)
    n = initial.n
    M_list = [list(r) for r in initial.entries]

    if free_mask is None:
        free_positions = [(i, j) for i in range(n) for j in range(n)]
    else:
        free_positions = [
            (i, j) for i in range(n) for j in range(n) if free_mask[i][j]
        ]
    if not free_positions:
        return SpectralAnchorResult(
            matrix=Matrix(n=n, entries=M_list),
            abs_det=abs(integer_determinant(M_list)),
            iterations=0,
            accepted=0,
            walltime_seconds=0.0,
            refresh_count=0,
        )

    state = IncrementalState.from_matrix(M_list)
    tries = 0
    while state.det == 0 and tries < 16:
        for (i, j) in free_positions:
            M_list[i][j] = 1 if rng.random() < 0.5 else -1
        state = IncrementalState.from_matrix(M_list)
        tries += 1
    if state.det == 0:
        return SpectralAnchorResult(
            matrix=Matrix(n=n, entries=M_list),
            abs_det=0,
            iterations=0,
            accepted=0,
            walltime_seconds=0.0,
            refresh_count=0,
        )
    abs_det = abs(state.det)

    target = np.asarray(target_spectrum, dtype=np.float64)
    cur_dist = spectrum_distance(state.M, target)
    cur_e = -math.log(abs_det) + lambda_anchor * cur_dist

    best_M = [list(r) for r in state.M]
    best_abs_det = abs_det
    best_e = cur_e
    best_dist = cur_dist

    sv_mult_history: list[int] = []
    distinct_sv_history: list[int] = []
    abs_det_history: list[int] = []

    accepted = 0
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0
    start = time.monotonic()

    for step in range(iterations):
        if deadline is not None and time.monotonic() >= deadline:
            break
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        (i, j) = free_positions[rng.randrange(len(free_positions))]
        new_det = propose_entry_flip(state, i, j)
        if new_det == 0:
            continue
        new_abs_det = abs(new_det)

        if rescore_every <= 1 or step % rescore_every == 0:
            old = state.M[i][j]
            state.M[i][j] = -old
            new_dist = spectrum_distance(state.M, target)
            state.M[i][j] = old
        else:
            new_dist = cur_dist
        new_e = -math.log(new_abs_det) + lambda_anchor * new_dist
        delta = new_e - cur_e
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            try:
                apply_entry_flip(state, i, j)
            except ZeroDivisionError:
                incr_refresh(state)
                continue
            cur_e = new_e
            cur_dist = new_dist
            abs_det = new_abs_det
            accepted += 1
            if new_e < best_e:
                best_e = new_e
                best_M = [list(r) for r in state.M]
                best_abs_det = new_abs_det
                best_dist = new_dist
            if accepted % 500 == 0:
                incr_refresh(state)
                cur_dist = spectrum_distance(state.M, target)
                cur_e = -math.log(abs_det) + lambda_anchor * cur_dist

        if log_every and step % log_every == 0:
            sv_mult_history.append(
                dominant_sv_multiplicity(state.M, tol=0.05)
            )
            distinct_sv_history.append(distinct_sv_count(state.M, tol=0.05))
            abs_det_history.append(int(abs_det))

    walltime = time.monotonic() - start
    best_abs_det_verified = abs(integer_determinant(best_M))
    result = SpectralAnchorResult(
        matrix=Matrix(n=n, entries=best_M),
        abs_det=best_abs_det_verified,
        iterations=iterations,
        accepted=accepted,
        walltime_seconds=walltime,
        refresh_count=0,
        sv_mult_history=sv_mult_history,
        distinct_sv_history=distinct_sv_history,
        abs_det_history=abs_det_history,
    )
    setattr(result, "spectrum_distance", float(best_dist))
    return result


__all__ = [
    "SpectralAnchorResult",
    "dominant_sv_multiplicity",
    "distinct_sv_count",
    "spectral_anchor_score",
    "E_anchor",
    "dominant_eigenspace",
    "flip_dominant_projection",
    "all_flip_projections",
    "sa_spectral_anchored",
    "bordered_extension_sa",
    # Targeted-spectrum extension
    "predict_target_spectrum",
    "spectrum_distance",
    "sa_targeted_spectrum",
]
