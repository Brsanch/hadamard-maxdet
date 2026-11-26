"""Gram-asymmetry energy and diagnostic-driven SA moves for ±1 matrices.

Background
----------
For an ``n × n`` ±1 matrix ``M`` with rows ``v_i``, the Gram off-diagonal
distribution ``{<v_i, v_j> : i < j}`` is the central diagnostic for how
close ``M`` is to a Hadamard / maxdet target. For ``n ≡ 3 (mod 4)`` the
known maxdet champions have nearly all off-diagonals at ``-1`` (the value
closest to zero in the residue class ``n mod 4 = 3``), with a small count
at ``+3`` from row pairs that are too aligned.

The current in-repo H23 best (a bordered two-circulant SA result) sits at
0.6704·Barba with Gram fingerprint ``{-3: 11, -1: 209, +1: 11, +3: 22}``.
The 22 pairs at ``+3`` are the "badness budget" we want to reduce.

Energy
------
We define a non-negative scalar that vanishes iff every off-diagonal
Gram entry is ≤ -1 (the asymmetric regime that maximises ``|det|``):

    E_gram(M) = sum_{i < j} max(0, <v_i, v_j> + 1)^2

For ``n ≡ 3 (mod 4)`` and parity reasons the off-diagonals are odd, so
the lowest-energy configurations have all off-diagonals at ``-1`` (which
contributes 0 each); a single ``+1`` contributes ``(1+1)^2 = 4``; a
``+3`` contributes ``(3+1)^2 = 16``; a ``+5`` contributes ``36``; ...

O(n) delta
----------
When entry ``M[i][k]`` is flipped (``s = M[i][k]`` before; new value
``-s``), the only Gram entries that change are ``<v_i, v_j>`` for ``j ≠ i``,
each shifting by ``-2 * s * M[j][k]``. This gives an O(n) per-flip
energy delta that does not touch any determinant.

This module provides:

  * :func:`gram_offdiag_matrix` — the symmetric n×n Gram off-diagonal
    matrix as a list of lists (diagonal entries set to 0; we never use
    them in the energy).
  * :func:`gram_energy` — exact energy of a matrix (O(n^3); for tests).
  * :func:`gram_energy_delta` — O(n) predicted delta from a single
    entry flip.
  * :func:`apply_flip_update_gram` — flip an entry in place and update
    the cached Gram matrix in O(n).
  * :func:`targeted_plus3_flip` — biased proposer that 70 % of the time
    targets the worst-aligned row pair and proposes a flip that reduces
    that pair's inner product.
  * :func:`sa_gram_energy` — SA over ±1 matrices using ``E_gram`` as the
    energy with the O(n) delta. Reports best |det| of the best-E_gram
    state via exact Bareiss at the end.
  * :func:`sa_targeted` — SA over ±1 matrices using the biased proposer
    with the standard ``-log|det|`` energy via the rank-1 incremental
    path (same machinery as ``sa_incremental``).
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field

from hadamard_maxdet.incremental import (
    IncrementalState,
    _compute_adjugate,
    apply_entry_flip,
    propose_entry_flip,
    refresh,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ----------------------------------------------------------------------------
# Energy primitives
# ----------------------------------------------------------------------------


def gram_offdiag_matrix(M: list[list[int]]) -> list[list[int]]:
    """Return the n×n Gram matrix ``G[i][j] = <v_i, v_j>`` as a list-of-lists.

    Diagonals are set to 0 (we never use the diagonal in the energy; the
    real value is ``n``). Cost O(n^3); call once per state, then keep
    incremental via :func:`apply_flip_update_gram`.
    """
    n = len(M)
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        row_i = M[i]
        for j in range(i + 1, n):
            row_j = M[j]
            s = 0
            for k in range(n):
                s += row_i[k] * row_j[k]
            G[i][j] = s
            G[j][i] = s
    return G


def gram_energy(M: list[list[int]]) -> int:
    """Exact ``E_gram(M) = sum_{i<j} max(0, <v_i, v_j> + 1)^2``.

    O(n^3) reference implementation used only in tests / verification.
    """
    n = len(M)
    e = 0
    for i in range(n):
        row_i = M[i]
        for j in range(i + 1, n):
            row_j = M[j]
            s = 0
            for k in range(n):
                s += row_i[k] * row_j[k]
            t = s + 1
            if t > 0:
                e += t * t
    return e


def gram_energy_from_G(G: list[list[int]]) -> int:
    """``E_gram`` from a precomputed Gram matrix. O(n^2)."""
    n = len(G)
    e = 0
    for i in range(n):
        row = G[i]
        for j in range(i + 1, n):
            t = row[j] + 1
            if t > 0:
                e += t * t
    return e


def gram_energy_delta(
    M: list[list[int]],
    G: list[list[int]],
    i: int,
    k: int,
) -> int:
    """O(n) predicted change in ``E_gram`` from flipping ``M[i][k]``.

    Parameters
    ----------
    M
        Current ±1 matrix.
    G
        Cached Gram off-diagonal matrix consistent with ``M`` (as
        produced by :func:`gram_offdiag_matrix` or maintained by
        :func:`apply_flip_update_gram`).
    i, k
        Position of the proposed flip.

    Returns
    -------
    int
        ``E_gram(M_after_flip) - E_gram(M_before_flip)``. Computed as a
        sum of ``n - 1`` constant-time per-row updates: for each row
        ``j != i``, the inner product ``<v_i, v_j>`` shifts by
        ``shift = -2 * M[i][k] * M[j][k]``; the per-pair energy
        contribution shifts from ``max(0, G[i][j] + 1)^2`` to
        ``max(0, G[i][j] + shift + 1)^2``.
    """
    n = len(M)
    s = M[i][k]
    Gi = G[i]
    delta = 0
    for j in range(n):
        if j == i:
            continue
        # shift = -2 * s * M[j][k]
        shift = -2 * s * M[j][k]
        old = Gi[j] + 1
        new = old + shift
        old_sq = old * old if old > 0 else 0
        new_sq = new * new if new > 0 else 0
        delta += new_sq - old_sq
    return delta


def apply_flip_update_gram(
    M: list[list[int]],
    G: list[list[int]],
    i: int,
    k: int,
) -> None:
    """Flip ``M[i][k]`` in place and update the cached Gram matrix in O(n).

    For each ``j != i``, ``G[i][j]`` and ``G[j][i]`` shift by
    ``-2 * s * M[j][k]`` where ``s`` is the value of ``M[i][k]`` *before*
    the flip.
    """
    n = len(M)
    s = M[i][k]
    for j in range(n):
        if j == i:
            continue
        shift = -2 * s * M[j][k]
        G[i][j] += shift
        G[j][i] += shift
    M[i][k] = -s


# ----------------------------------------------------------------------------
# Targeted +3-pair proposer (Move C).
# ----------------------------------------------------------------------------


def targeted_plus3_flip(
    M: list[list[int]],
    G: list[list[int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    """Pick (i, k) such that flipping ``M[i][k]`` reduces a worst-aligned pair.

    Algorithm:
      * Find ``max_g`` = the maximum off-diagonal Gram value in ``G``.
      * Collect all pairs ``(i, j)`` with ``G[i][j] == max_g``.
      * Pick one such pair uniformly at random.
      * Among columns where ``M[i][k] == M[j][k]`` ("agree" columns —
        those that contribute ``+1`` to ``<v_i, v_j>``), pick one
        uniformly. Flipping ``M[i][k]`` converts an agree into a
        disagree, so ``<v_i, v_j>`` decreases by 2.
      * Return ``(i, k)``.

    Returns ``None`` only if no such pair exists (e.g. ``n < 2``) or
    the worst-aligned pair already disagrees on every column (which
    cannot happen with a real Gram entry > -n).
    """
    n = len(M)
    if n < 2:
        return None
    # Find max off-diagonal Gram value.
    max_g: int | None = None
    for i in range(n):
        Gi = G[i]
        for j in range(i + 1, n):
            v = Gi[j]
            if max_g is None or v > max_g:
                max_g = v
    if max_g is None:
        return None
    # Collect all pairs at the max.
    worst_pairs: list[tuple[int, int]] = []
    for i in range(n):
        Gi = G[i]
        for j in range(i + 1, n):
            if Gi[j] == max_g:
                worst_pairs.append((i, j))
    if not worst_pairs:
        return None
    i, j = worst_pairs[rng.randrange(len(worst_pairs))]
    # Agree columns for (i, j).
    agree_cols = [k for k in range(n) if M[i][k] == M[j][k]]
    if not agree_cols:
        return None
    k = agree_cols[rng.randrange(len(agree_cols))]
    # Randomly pick which row's entry to flip (i or j); both reduce <v_i, v_j>.
    if rng.random() < 0.5:
        return (i, k)
    return (j, k)


# ----------------------------------------------------------------------------
# SA driver A: E_gram energy with O(n) delta.
# ----------------------------------------------------------------------------


@dataclass
class GramEnergyAnnealResult:
    """Outcome of :func:`sa_gram_energy`."""

    matrix: Matrix
    abs_det: int
    best_e_gram: int
    final_e_gram: int
    iterations: int
    accepted: int
    walltime_seconds: float
    # (step, e_gram, abs_det) sampled at log_every; abs_det is exact.
    correlation_samples: list[tuple[int, int, int]] = field(default_factory=list)
    starting_e_gram: int = 0
    gram_fingerprint: dict[int, int] = field(default_factory=dict)


def _gram_offdiag_counter(G: list[list[int]]) -> dict[int, int]:
    n = len(G)
    c: Counter = Counter()
    for i in range(n):
        Gi = G[i]
        for j in range(i + 1, n):
            c[Gi[j]] += 1
    return dict(sorted(c.items()))


def sa_gram_energy(
    initial: Matrix,
    *,
    iterations: int = 100_000,
    t_start: float = 50.0,
    t_end: float = 0.5,
    seed: int | None = None,
    log_every: int = 0,
    correlation_every: int = 0,
    wall_time_budget: float | None = None,
) -> GramEnergyAnnealResult:
    """Simulated annealing on ±1 matrices with energy ``E_gram``.

    All proposals are uniform random entry flips; the O(n) delta makes
    each step cheap and detached from any determinant computation.

    Parameters
    ----------
    initial
        Starting matrix (a :class:`Matrix`).
    iterations
        Maximum MC steps (capped early by ``wall_time_budget`` if set).
    t_start, t_end
        Geometric temperature schedule in ``E_gram`` units. Defaults
        target the typical scale of the H23 problem (single-flip
        ``|delta|`` ranges from a few to a few hundred).
    seed
        PRNG seed.
    log_every
        Reserved for future use; not currently emitted.
    correlation_every
        If positive, sample ``(step, E_gram(current), |det|(current))``
        every that-many *accepted* moves. ``|det|`` is computed via
        exact Bareiss — non-trivial cost, so use a coarse interval.
    wall_time_budget
        If set, terminate as soon as wall-clock exceeds this many
        seconds (checked once per 1024 iterations).

    Returns
    -------
    GramEnergyAnnealResult
        Best matrix by E_gram, with its exact |det|, the E_gram trace,
        and the final Gram fingerprint.
    """
    n = initial.n
    rng = random.Random(seed)

    M = [row[:] for row in initial.entries]
    G = gram_offdiag_matrix(M)
    e = gram_energy_from_G(G)
    best_M = [row[:] for row in M]
    best_e = e

    accepted = 0
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0
    samples: list[tuple[int, int, int]] = []
    starting_e = e
    start = time.monotonic()

    step = 0
    for step in range(iterations):
        if wall_time_budget is not None and (step & 1023) == 0:
            if time.monotonic() - start >= wall_time_budget:
                break
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)
        i = rng.randrange(n)
        k = rng.randrange(n)
        d = gram_energy_delta(M, G, i, k)
        if d <= 0 or rng.random() < math.exp(-d / max(t, 1e-12)):
            apply_flip_update_gram(M, G, i, k)
            e += d
            accepted += 1
            if e < best_e:
                best_e = e
                best_M = [row[:] for row in M]
            if (
                correlation_every > 0
                and (accepted % correlation_every) == 0
            ):
                d_now = abs(integer_determinant(M))
                samples.append((step, e, d_now))

    walltime = time.monotonic() - start
    # Authoritative |det| of the best-E_gram state.
    best_abs_det = abs(integer_determinant(best_M))
    fp = _gram_offdiag_counter(gram_offdiag_matrix(best_M))
    return GramEnergyAnnealResult(
        matrix=Matrix(n=n, entries=best_M),
        abs_det=best_abs_det,
        best_e_gram=best_e,
        final_e_gram=e,
        iterations=step + 1,
        accepted=accepted,
        walltime_seconds=walltime,
        correlation_samples=samples,
        starting_e_gram=starting_e,
        gram_fingerprint=fp,
    )


# ----------------------------------------------------------------------------
# SA driver C: targeted +3-pair proposer with -log|det| energy.
# ----------------------------------------------------------------------------


@dataclass
class TargetedAnnealResult:
    """Outcome of :func:`sa_targeted`."""

    matrix: Matrix
    abs_det: int
    iterations: int
    accepted: int
    targeted_proposed: int
    targeted_accepted: int
    uniform_proposed: int
    uniform_accepted: int
    walltime_seconds: float
    refreshes: int
    gram_fingerprint: dict[int, int] = field(default_factory=dict)
    starting_abs_det: int = 0
    starting_gram_fingerprint: dict[int, int] = field(default_factory=dict)


def sa_targeted(
    initial: Matrix,
    *,
    iterations: int = 100_000,
    t_start: float = 1.0,
    t_end: float = 0.01,
    seed: int | None = None,
    targeted_fraction: float = 0.7,
    refresh_every: int = 2000,
    wall_time_budget: float | None = None,
) -> TargetedAnnealResult:
    """SA with biased proposer (Move C) + standard ``-log|det|`` energy.

    With probability ``targeted_fraction`` (default 0.7), propose a flip
    via :func:`targeted_plus3_flip` (attacks the worst-aligned row pair).
    Otherwise, propose a uniform random entry flip. Both use the
    rank-1 incremental adjugate path for fast ``|det|`` evaluation.

    Parameters mirror :func:`hadamard_maxdet.incremental.sa_incremental`.
    """
    n = initial.n
    rng = random.Random(seed)

    M = [row[:] for row in initial.entries]
    state = IncrementalState.from_matrix(M)
    if state.det == 0:
        # Refuse to anneal from a singular start; caller should reseed.
        return TargetedAnnealResult(
            matrix=Matrix(n=n, entries=M),
            abs_det=0,
            iterations=0,
            accepted=0,
            targeted_proposed=0,
            targeted_accepted=0,
            uniform_proposed=0,
            uniform_accepted=0,
            walltime_seconds=0.0,
            refreshes=0,
        )
    starting_abs_det = abs(state.det)
    G = gram_offdiag_matrix(state.M)
    starting_fp = _gram_offdiag_counter(G)

    current_e = -math.log(abs(state.det))
    best_entries = [row[:] for row in state.M]
    best_e = current_e

    accepted = 0
    targeted_proposed = 0
    targeted_accepted = 0
    uniform_proposed = 0
    uniform_accepted = 0
    refreshes = 0
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0
    start = time.monotonic()

    step = 0
    for step in range(iterations):
        if wall_time_budget is not None and (step & 1023) == 0:
            if time.monotonic() - start >= wall_time_budget:
                break
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        used_targeted = rng.random() < targeted_fraction
        if used_targeted:
            pick = targeted_plus3_flip(state.M, G, rng)
            if pick is None:
                pick = (rng.randrange(n), rng.randrange(n))
                used_targeted = False
            i, k = pick
            targeted_proposed += 1
        else:
            i = rng.randrange(n)
            k = rng.randrange(n)
            uniform_proposed += 1

        new_det = propose_entry_flip(state, i, k)
        if new_det == 0:
            continue
        proposal_e = -math.log(abs(new_det))
        delta = proposal_e - current_e
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            apply_entry_flip(state, i, k)
            apply_flip_update_gram_no_M(G, state.M, i, k)
            current_e = proposal_e
            accepted += 1
            if used_targeted:
                targeted_accepted += 1
            else:
                uniform_accepted += 1
            if current_e < best_e:
                best_e = current_e
                best_entries = [row[:] for row in state.M]
            if refresh_every > 0 and accepted % refresh_every == 0:
                old_det = state.det
                refresh(state)
                refreshes += 1
                if state.det != old_det:
                    if state.det != 0:
                        current_e = -math.log(abs(state.det))
                    # Resync Gram cache as well.
                    G = gram_offdiag_matrix(state.M)

    walltime = time.monotonic() - start
    best_abs_det = abs(integer_determinant(best_entries))
    fp = _gram_offdiag_counter(gram_offdiag_matrix(best_entries))
    return TargetedAnnealResult(
        matrix=Matrix(n=n, entries=best_entries),
        abs_det=best_abs_det,
        iterations=step + 1,
        accepted=accepted,
        targeted_proposed=targeted_proposed,
        targeted_accepted=targeted_accepted,
        uniform_proposed=uniform_proposed,
        uniform_accepted=uniform_accepted,
        walltime_seconds=walltime,
        refreshes=refreshes,
        gram_fingerprint=fp,
        starting_abs_det=starting_abs_det,
        starting_gram_fingerprint=starting_fp,
    )


def apply_flip_update_gram_no_M(
    G: list[list[int]],
    M_after: list[list[int]],
    i: int,
    k: int,
) -> None:
    """Update Gram cache when ``M`` has *already* been flipped at (i, k).

    Used by :func:`sa_targeted` because :func:`apply_entry_flip` mutates
    ``state.M`` itself; we then need to update ``G`` to match. ``M[i][k]``
    is the *new* value (post-flip), so the previous value was ``-M[i][k]``;
    the shift in ``<v_i, v_j>`` is ``-2 * (-M[i][k]) * M[j][k] =
    2 * M[i][k] * M[j][k]``.
    """
    n = len(M_after)
    s_new = M_after[i][k]
    for j in range(n):
        if j == i:
            continue
        shift = 2 * s_new * M_after[j][k]
        G[i][j] += shift
        G[j][i] += shift


# Suppress unused-import warning when callers only need the SA drivers.
_ = _compute_adjugate
