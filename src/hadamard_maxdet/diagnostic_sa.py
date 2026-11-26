"""Diagnostic-driven SA variants.

Three improvements identified from the H19 move-profile + small-n
winner-decode crawl:

1. **Hot-only SA** — the bottom temperature deciles use 40% of compute
   for 1.7% of useful gain at H19. ``sa_hot_only`` clips the schedule
   at ``t_end=0.05`` (10× higher than ``sa_incremental``'s default
   ``0.005``), so each restart is ~5× shorter and we can afford ~4× as
   many restarts in the same wall-clock budget.

2. **Balance-bonus energy** — proven winners at n ∈ {11, 13, 15, 17}
   have BALANCED gram off-diagonal counts of −1 vs +1 (ratio
   |c(−1)−c(+1)| / n² near 0). Our H19 best and H23 BTC have skewed
   distributions. ``sa_balanced`` adds a penalty
   ``λ · |count(−1) − count(+1)| / n²`` to the energy, with O(n)
   incremental update via :func:`count_imbalance_delta`.

3. **Row-weight-diversity move** — proven winners have 5–7 distinct
   row weights. BTC has only 2. :func:`weight_perturbation_flip` picks
   a random row, then two columns where that row has the SAME sign,
   and flips both — changing the row weight by ±2 in a single
   selectively-targeted way (vs. an entry flip, which also changes
   weight by ±2 but at random and so explores weight space more
   diffusely).

Public API
----------

* :func:`count_imbalance(M)` — ``|c(−1) − c(+1)| / n²``
* :func:`count_imbalance_delta(M, MMt, i, j)` — O(n) per-flip change
* :func:`weight_perturbation_flip(M, rng)` — pair-flip move
* :func:`sa_hot_only(...)` — short hot-temp SA
* :func:`sa_balanced(...)` — combined SA with balance bonus and
  optional weight-perturbation moves

All ``sa_*`` functions return :class:`IncrementalAnnealResult` for
drop-in compatibility with ``sa_incremental``.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from hadamard_maxdet.incremental import (
    IncrementalAnnealResult,
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    refresh,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Imbalance helpers
# ---------------------------------------------------------------------------


def gram_offdiag_counts(M: list[list[int]]) -> dict[int, int]:
    """Counts of off-diagonal values of MMᵀ.

    Off-diagonal pairs (i, j) with i < j; each pair counted once.
    """
    n = len(M)
    out: dict[int, int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            s = 0
            for k in range(n):
                s += M[i][k] * M[j][k]
            out[s] = out.get(s, 0) + 1
    return out


def count_imbalance(M: list[list[int]]) -> float:
    """``|count(MMᵀ_off, -1) − count(MMᵀ_off, +1)| / n²``.

    Off-diagonal pairs are counted with multiplicity 1 each (i < j).
    Returns 0 for a Hadamard matrix (whose off-diagonals are all 0,
    so c(−1) = c(+1) = 0).
    """
    n = len(M)
    if n == 0:
        return 0.0
    cm = 0
    cp = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = 0
            for k in range(n):
                s += M[i][k] * M[j][k]
            if s == -1:
                cm += 1
            elif s == 1:
                cp += 1
    return abs(cm - cp) / (n * n)


def compute_gram_full(M: list[list[int]]) -> list[list[int]]:
    """Full MMᵀ as a list of lists. O(n³). Used to seed incremental
    tracking; not for the hot loop."""
    n = len(M)
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(n):
                s += M[i][k] * M[j][k]
            G[i][j] = s
    return G


def count_minus_plus(G: list[list[int]]) -> tuple[int, int]:
    """Count off-diagonal entries (i < j) of G equal to −1 and +1.

    Strict-upper-triangular sum (each pair counted once).
    """
    n = len(G)
    cm = 0
    cp = 0
    for i in range(n):
        for j in range(i + 1, n):
            v = G[i][j]
            if v == -1:
                cm += 1
            elif v == 1:
                cp += 1
    return cm, cp


def count_imbalance_delta(
    G: list[list[int]],
    M: list[list[int]],
    i: int,
    j: int,
) -> tuple[int, int]:
    """Return the (Δc(−1), Δc(+1)) caused by flipping ``M[i][j]``.

    Flipping ``M[i][j] → −M[i][j]`` changes Gram entry G[i][r]
    (for r ≠ i) by ``−2 · M_old[i][j] · M[r][j]`` and symmetrically
    for G[r][i]. We only need to touch the n−1 off-diagonal entries
    in row i (== column i by symmetry). Each i-vs-r pair is counted
    once in the strict-upper-triangle.

    The diagonal G[i][i] = n is unchanged by a row-i entry flip
    (since (M[i][k])² = 1 either way), so it does not enter c(−1)/c(+1).
    """
    n = len(G)
    old = M[i][j]
    delta_factor = -2 * old  # G_new[i][r] = G_old[i][r] + delta_factor·M[r][j]
    dcm = 0
    dcp = 0
    for r in range(n):
        if r == i:
            continue
        old_g = G[i][r]
        new_g = old_g + delta_factor * M[r][j]
        # decrement old class
        if old_g == -1:
            dcm -= 1
        elif old_g == 1:
            dcp -= 1
        # increment new class
        if new_g == -1:
            dcm += 1
        elif new_g == 1:
            dcp += 1
    return dcm, dcp


def apply_gram_flip(
    G: list[list[int]],
    M: list[list[int]],
    i: int,
    j: int,
) -> None:
    """Update ``G = M Mᵀ`` in place to reflect ``M[i][j]`` having been
    flipped (caller's responsibility to flip ``M[i][j]`` itself
    afterwards or before — this routine assumes ``M`` still has the OLD
    value at ``[i][j]`` so it can compute the delta).

    Only row i and column i of G change.
    """
    n = len(G)
    old = M[i][j]
    delta_factor = -2 * old
    for r in range(n):
        if r == i:
            # G[i][i] = sum_k M[i][k]^2 = n; unchanged.
            continue
        d = delta_factor * M[r][j]
        G[i][r] += d
        G[r][i] += d


# ---------------------------------------------------------------------------
# Weight-perturbation move
# ---------------------------------------------------------------------------


def weight_perturbation_flip(
    M: list[list[int]],
    rng: random.Random,
) -> tuple[int, int, int] | None:
    """Pick a random row i; pick two distinct columns (j, k) where
    ``M[i][j] == M[i][k]``; flip both. Changes row i's weight by ±2.

    Returns ``(i, j, k)`` for the caller to commit, or ``None`` if no
    valid pair exists in the chosen row (a row of all +1s or all −1s
    is fine too — any same-sign pair works).
    """
    n = len(M)
    i = rng.randrange(n)
    row = M[i]
    pluses = [c for c in range(n) if row[c] > 0]
    minuses = [c for c in range(n) if row[c] < 0]
    # Pick the larger pool with probability proportional to pair count
    # (favours flipping in whichever direction has more entries).
    n_pp = len(pluses) * (len(pluses) - 1) // 2
    n_mm = len(minuses) * (len(minuses) - 1) // 2
    total = n_pp + n_mm
    if total == 0:
        return None
    if rng.random() < n_pp / total:
        j, k = rng.sample(pluses, 2)
    else:
        j, k = rng.sample(minuses, 2)
    return (i, j, k)


# ---------------------------------------------------------------------------
# Hot-only SA — wraps sa_incremental with t_end clipped at 0.05
# ---------------------------------------------------------------------------


def sa_hot_only(
    n: int,
    *,
    iterations: int = 50_000,
    t_start: float = 1.0,
    t_end: float = 0.05,
    seed: int | None = None,
    initial: Matrix | None = None,
    refresh_every: int = 0,
    log_every: int = 0,
) -> IncrementalAnnealResult:
    """Short hot-temperature SA.

    Same engine as :func:`sa_incremental`, but ``t_end`` defaults to
    ``0.05`` (10× higher than ``sa_incremental``'s default), and
    ``iterations`` defaults to 50k (4× shorter). The intent is to spend
    each restart on the productive hot deciles only and use the saved
    wall time on more restarts elsewhere.
    """
    # Defer to the existing sa_incremental engine for correctness;
    # diverge only on the schedule.
    from hadamard_maxdet.incremental import sa_incremental

    return sa_incremental(
        n,
        iterations=iterations,
        t_start=t_start,
        t_end=t_end,
        seed=seed,
        initial=initial,
        refresh_every=refresh_every,
        log_every=log_every,
    )


# ---------------------------------------------------------------------------
# Balanced SA — incremental |det| + balance penalty + optional weight-flip
# ---------------------------------------------------------------------------


def sa_balanced(
    n: int,
    *,
    iterations: int = 50_000,
    t_start: float = 1.0,
    t_end: float = 0.05,
    lambda_balance: float = 0.0,
    weight_perturb_prob: float = 0.0,
    seed: int | None = None,
    initial: Matrix | None = None,
    refresh_every: int = 2000,
    log_every: int = 0,
) -> IncrementalAnnealResult:
    """SA over

        E(M) = −log|det(M)| + λ_balance · |c(−1) − c(+1)| / n²

    with hot-only schedule (``t_end=0.05`` default) and an optional
    fraction of moves drawn from :func:`weight_perturbation_flip`
    instead of single-entry flips.

    The Gram MMᵀ is carried alongside the rank-1 (det, adj) state and
    updated incrementally on every accepted entry flip in O(n). On a
    weight-perturbation step (a *pair* of entry flips on the same row),
    we update Gram for each flip and call ``apply_entry_flip`` twice;
    the second propose computes the new det from the post-first-flip
    state, so the move is exact.
    """
    rng = random.Random(seed)

    if initial is None:
        mat = Matrix.random(n, rng)
    elif initial.n != n:
        raise ValueError(f"initial order {initial.n} ≠ n={n}")
    else:
        mat = initial.copy()

    state = IncrementalState.from_matrix(mat.entries)
    tries = 0
    while state.det == 0 and tries < 16:
        mat = Matrix.random(n, rng)
        state = IncrementalState.from_matrix(mat.entries)
        tries += 1
    if state.det == 0:
        return IncrementalAnnealResult(
            matrix=Matrix(n=n, entries=[row[:] for row in state.M]),
            abs_det=0,
            iterations=0,
            accepted=0,
            walltime_seconds=0.0,
            refreshes=0,
            residual_triggered=0,
        )

    # Track Gram (full n×n; n²·O(1) updates per flip, only row/col i).
    G = compute_gram_full(state.M)
    cm, cp = count_minus_plus(G)
    n2 = n * n

    def imbalance() -> float:
        return abs(cm - cp) / n2

    def energy(det_abs: int) -> float:
        return -math.log(det_abs) + lambda_balance * imbalance()

    current_e = energy(abs(state.det))
    best_entries = [row[:] for row in state.M]
    best_abs = abs(state.det)
    best_e = current_e

    accepted = 0
    refreshes = 0
    residual_triggered = 0
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0
    start = time.monotonic()
    history: list[int] = []

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        use_weight = (
            weight_perturb_prob > 0.0
            and rng.random() < weight_perturb_prob
        )

        if use_weight:
            triple = weight_perturbation_flip(state.M, rng)
            if triple is None:
                continue
            i, j, k = triple
            # Step 1: flip M[i][j].
            new_det1 = propose_entry_flip(state, i, j)
            if new_det1 == 0:
                continue
            # Predicted Gram delta from flipping (i, j) — but we don't
            # commit the energy yet; we evaluate the *combined* move.
            dcm1, dcp1 = count_imbalance_delta(G, state.M, i, j)
            # Apply the first flip to state and Gram (provisionally).
            apply_gram_flip(G, state.M, i, j)
            apply_entry_flip(state, i, j)
            cm += dcm1
            cp += dcp1
            # Step 2: flip M[i][k]; propose using updated state.
            new_det2 = propose_entry_flip(state, i, k)
            if new_det2 == 0:
                # Roll back step 1.
                # Reverse the Gram update by flipping (i, j) back.
                apply_gram_flip(G, state.M, i, j)
                apply_entry_flip(state, i, j)
                cm -= dcm1
                cp -= dcp1
                continue
            dcm2, dcp2 = count_imbalance_delta(G, state.M, i, k)
            # Predicted post-move counts:
            new_cm = cm + dcm2
            new_cp = cp + dcp2
            new_imbal = abs(new_cm - new_cp) / n2
            proposal_e = -math.log(abs(new_det2)) + lambda_balance * new_imbal
            delta = proposal_e - current_e
            if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
                # Accept: commit step 2.
                apply_gram_flip(G, state.M, i, k)
                apply_entry_flip(state, i, k)
                cm = new_cm
                cp = new_cp
                current_e = proposal_e
                accepted += 1
                if abs(state.det) > best_abs:
                    best_abs = abs(state.det)
                    best_entries = [row[:] for row in state.M]
                if current_e < best_e:
                    best_e = current_e
            else:
                # Reject: roll back step 1.
                apply_gram_flip(G, state.M, i, j)
                apply_entry_flip(state, i, j)
                cm -= dcm1
                cp -= dcp1
        else:
            # Single-entry flip.
            i = rng.randrange(n)
            j = rng.randrange(n)
            new_det = propose_entry_flip(state, i, j)
            if new_det == 0:
                continue
            dcm, dcp = count_imbalance_delta(G, state.M, i, j)
            new_cm = cm + dcm
            new_cp = cp + dcp
            new_imbal = abs(new_cm - new_cp) / n2
            proposal_e = -math.log(abs(new_det)) + lambda_balance * new_imbal
            delta = proposal_e - current_e
            if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
                apply_gram_flip(G, state.M, i, j)
                apply_entry_flip(state, i, j)
                cm = new_cm
                cp = new_cp
                current_e = proposal_e
                accepted += 1
                if abs(state.det) > best_abs:
                    best_abs = abs(state.det)
                    best_entries = [row[:] for row in state.M]
                if current_e < best_e:
                    best_e = current_e
                if refresh_every > 0 and accepted % refresh_every == 0:
                    old_det = state.det
                    refresh(state)
                    refreshes += 1
                    if state.det != old_det:
                        residual_triggered += 1
                        if state.det != 0:
                            current_e = -math.log(abs(state.det)) + (
                                lambda_balance * imbalance()
                            )
                    # Resync Gram (cheap insurance — full O(n³) recompute).
                    G2 = compute_gram_full(state.M)
                    if G2 != G:
                        G = G2
                        cm, cp = count_minus_plus(G)
                        current_e = -math.log(abs(state.det)) + (
                            lambda_balance * abs(cm - cp) / n2
                        )

        if log_every and step % log_every == 0:
            history.append(best_abs)

    walltime = time.monotonic() - start
    best_abs_verified = abs(integer_determinant(best_entries))
    return IncrementalAnnealResult(
        matrix=Matrix(n=n, entries=best_entries),
        abs_det=best_abs_verified,
        iterations=iterations,
        accepted=accepted,
        walltime_seconds=walltime,
        refreshes=refreshes,
        residual_triggered=residual_triggered,
        history=history,
    )


__all__ = [
    "gram_offdiag_counts",
    "count_imbalance",
    "compute_gram_full",
    "count_minus_plus",
    "count_imbalance_delta",
    "apply_gram_flip",
    "weight_perturbation_flip",
    "sa_hot_only",
    "sa_balanced",
]
