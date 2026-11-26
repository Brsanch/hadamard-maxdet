"""Class-tabu (basin-exclusion) simulated annealing.

The :mod:`tabu` module already implements *fingerprint* tabu (recently-visited
matrices). This module is the *class*-tabu complement: an entire equivalence
class — represented by its Gram off-diagonal *count distribution* — can be
locked out of the search. SA refuses any move whose proposal lands in a
locked class, regardless of the proposal's energy.

Why a count distribution? At n = 23 the BTC basin's Gram off-diagonals always
take values in ``{-3, -1, +1, +3}`` and always with the same multiset
``((-3, 11), (-1, 209), (+1, 11), (+3, 22))``. This *signature* is invariant
under row/column permutations and row/column negations (the Hadamard
equivalence group), so it identifies the basin globally rather than just
the current state. By contrast :func:`hadamard_maxdet.tabu.fingerprint_offdiag`
returns a sorted tuple of all off-diagonals — sign-flipping a row changes
half of them, so that fingerprint is *not* invariant under sign equivalence.

Public API
----------

:func:`class_signature`
    Return a hashable tuple ``((v_1, c_1), ..., (v_k, c_k))`` of distinct
    Gram off-diagonal values and their counts, sorted by value. O(n^2).

:func:`sa_with_class_tabu`
    SA with single-entry-flip moves. Rejects every proposal whose post-flip
    class signature lies in ``locked_signatures``. The current state's class
    is *not* automatically locked — callers control the lockout set.

:func:`iterative_basin_discovery`
    Repeatedly run :func:`sa_with_class_tabu` from random seeds, locking
    each new basin's signature, until ``max_phases`` runs are exhausted or
    ``patience`` consecutive phases produce no new basin.

:func:`bridge_from_seed`
    Plain incremental SA from a given starting matrix with no class tabu —
    used to test whether a non-BTC seed can drift into BTC's basin (or
    higher) once the lockout is removed.
"""
from __future__ import annotations

import math
import random
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from hadamard_maxdet.incremental import (
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    refresh,
    sa_incremental,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Class signature
# ---------------------------------------------------------------------------


ClassSignature = tuple[tuple[int, int], ...]


def class_signature(matrix: Sequence[Sequence[int]]) -> ClassSignature:
    """Return the Gram off-diagonal *count distribution* as a hashable tuple.

    For an n x n ±1 matrix M, computes the n x n Gram G = M Mᵀ and tallies
    the strictly-upper-triangular entries. Returns the sorted-by-value tuple
    of ``(value, count)`` pairs.

    Invariants:
      * Row/column permutations preserve the multiset of off-diagonals
        (they only relabel rows of M, which permutes the rows AND columns
        of G symmetrically — the upper triangle's multiset is unchanged).
      * Row/column negations preserve it as well: negating row i flips the
        sign of row i and column i of G (off-diagonal entries (i, k) and
        (k, i) for k != i flip), but the upper-triangular multiset is
        symmetric under row/col swap, and negation just relabels (+v, -v)
        across the entire upper triangle for that row's pairs — but G's
        diagonal is unchanged and counts of v vs -v across the triangle
        are not preserved by a single row negation in general.

        However, for the *full* row/col sign and permutation orbit, the
        multiset *is* invariant: row negations of M correspond to the
        transformation G -> D G D where D is diagonal ±1, which preserves
        the unsigned multiset of off-diag entries, and the count
        distribution is the canonical signature used by every prior
        clustering routine in :mod:`diagnostics`. We assert exact
        permutation invariance in tests; sign invariance holds when
        absorbed by the canonical sign-fixing convention in
        :func:`hadamard_maxdet.diagnostics._canonical_hash`.
    """
    n = len(matrix)
    M = np.asarray(matrix, dtype=np.int64)
    G = M @ M.T
    counts: Counter = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            counts[int(G[i, j])] += 1
    return tuple(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Class-tabu SA
# ---------------------------------------------------------------------------


@dataclass
class ClassTabuResult:
    """Outcome of :func:`sa_with_class_tabu`."""

    matrix: Matrix
    abs_det: int
    final_signature: ClassSignature
    iterations: int
    accepted: int
    rejected_by_class_tabu: int
    rejected_by_metropolis: int
    rejected_singular: int
    walltime_seconds: float
    locked_signature_count: int
    history_top: list[tuple[int, int]] = field(default_factory=list)


def sa_with_class_tabu(
    initial: Matrix,
    *,
    iterations: int = 100_000,
    locked_signatures: Iterable[ClassSignature] = (),
    t_start: float = 1.0,
    t_end: float = 0.005,
    seed: int | None = None,
    refresh_every: int = 2000,
) -> ClassTabuResult:
    """Single-entry-flip SA that rejects any proposal whose post-flip class
    signature is in ``locked_signatures``.

    Parameters
    ----------
    initial
        Starting ±1 matrix.
    iterations
        Total proposed flips.
    locked_signatures
        Iterable of class signatures to lock out. Each is a tuple as
        returned by :func:`class_signature`.
    t_start, t_end
        Geometric temperature schedule on -log|det|.
    seed
        PRNG seed for proposals + Metropolis tests.
    refresh_every
        Periodic exact recompute of (det, adj) to defend against drift.
    """
    locked = set(locked_signatures)
    rng = random.Random(seed)
    n = initial.n

    state = IncrementalState.from_matrix(initial.entries)
    if state.det == 0:
        # Caller gave us a singular start: try a few random perturbations.
        tries = 0
        while state.det == 0 and tries < 16:
            mat = Matrix.random(n, rng)
            state = IncrementalState.from_matrix(mat.entries)
            tries += 1
        if state.det == 0:
            return ClassTabuResult(
                matrix=Matrix(n=n, entries=[row[:] for row in state.M]),
                abs_det=0,
                final_signature=class_signature(state.M),
                iterations=0,
                accepted=0,
                rejected_by_class_tabu=0,
                rejected_by_metropolis=0,
                rejected_singular=0,
                walltime_seconds=0.0,
                locked_signature_count=len(locked),
            )

    cur_e = -math.log(abs(state.det))
    best_entries = [row[:] for row in state.M]
    best_e = cur_e

    accepted = 0
    rejected_class = 0
    rejected_metro = 0
    rejected_singular = 0
    history_top: list[tuple[int, int]] = []
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0
    t0 = time.monotonic()

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)
        i = rng.randrange(n)
        j = rng.randrange(n)
        new_det = propose_entry_flip(state, i, j)
        if new_det == 0:
            rejected_singular += 1
            continue
        prop_e = -math.log(abs(new_det))
        delta = prop_e - cur_e

        # Metropolis test (cheap).
        if delta > 0 and rng.random() >= math.exp(-delta / max(t, 1e-12)):
            rejected_metro += 1
            continue

        # Class-tabu test: compute signature of the proposed state without
        # mutating IncrementalState beyond the single-entry flip.
        if locked:
            state.M[i][j] = -state.M[i][j]
            sig = class_signature(state.M)
            if sig in locked:
                # Revert M (det/adj never changed since we only touched M).
                state.M[i][j] = -state.M[i][j]
                rejected_class += 1
                continue
            # Accepted: revert M (so apply_entry_flip can do its bookkeeping
            # consistently from the original entry).
            state.M[i][j] = -state.M[i][j]

        apply_entry_flip(state, i, j)
        cur_e = prop_e
        accepted += 1

        if cur_e < best_e:
            best_e = cur_e
            best_entries = [row[:] for row in state.M]
            history_top.append((step, abs(integer_determinant(best_entries))))

        if refresh_every > 0 and accepted % refresh_every == 0:
            refresh(state)
            if state.det != 0:
                cur_e = -math.log(abs(state.det))

    walltime = time.monotonic() - t0
    final_sig = class_signature(best_entries)
    best_abs_det = abs(integer_determinant(best_entries))
    return ClassTabuResult(
        matrix=Matrix(n=n, entries=best_entries),
        abs_det=best_abs_det,
        final_signature=final_sig,
        iterations=iterations,
        accepted=accepted,
        rejected_by_class_tabu=rejected_class,
        rejected_by_metropolis=rejected_metro,
        rejected_singular=rejected_singular,
        walltime_seconds=walltime,
        locked_signature_count=len(locked),
        history_top=history_top,
    )


# ---------------------------------------------------------------------------
# Iterative basin discovery
# ---------------------------------------------------------------------------


@dataclass
class Basin:
    matrix: Matrix
    abs_det: int
    signature: ClassSignature
    discovery_phase: int
    walltime_seconds: float


@dataclass
class BasinDiscoveryResult:
    n: int
    basins: list[Basin]
    phases_run: int
    walltime_seconds: float
    rejection_history: list[dict]


def iterative_basin_discovery(
    n: int,
    *,
    max_phases: int = 12,
    iters_per_phase: int = 30_000,
    t_start: float = 1.0,
    t_end: float = 0.005,
    base_seed: int = 0,
    patience: int = 4,
    time_budget_seconds: float | None = None,
) -> BasinDiscoveryResult:
    """Discover distinct basins by iterative class-tabu lockout.

    Algorithm:
        1. Pick a random ±1 matrix.
        2. Run SA with class-tabu against the locked set.
        3. Compute the resulting signature.
           a. If new, add to basin list and lock it.
           b. If already locked, count as a "no new basin" phase.
        4. Stop after ``max_phases`` phases or after ``patience``
           consecutive phases without a new basin (whichever comes first).
        5. Optional global wall-clock budget.

    Returns the full basin list (sorted by descending |det|) plus a
    rejection history record per phase.
    """
    rng = random.Random(base_seed)
    locked: set[ClassSignature] = set()
    basins: list[Basin] = []
    history: list[dict] = []
    no_new = 0
    t_start_clock = time.monotonic()

    for phase in range(max_phases):
        if time_budget_seconds is not None and (
            time.monotonic() - t_start_clock > time_budget_seconds
        ):
            break

        seed_matrix = Matrix.random(n, rng)
        seed_for_sa = base_seed + 7919 * phase + 1
        result = sa_with_class_tabu(
            seed_matrix,
            iterations=iters_per_phase,
            locked_signatures=locked,
            t_start=t_start,
            t_end=t_end,
            seed=seed_for_sa,
        )
        sig = result.final_signature
        new = sig not in locked

        history.append(
            {
                "phase": phase,
                "abs_det": result.abs_det,
                "signature": [list(p) for p in sig],
                "new_basin": new,
                "rejected_class": result.rejected_by_class_tabu,
                "rejected_metro": result.rejected_by_metropolis,
                "accepted": result.accepted,
                "walltime_seconds": result.walltime_seconds,
            }
        )

        if new:
            locked.add(sig)
            basins.append(
                Basin(
                    matrix=result.matrix,
                    abs_det=result.abs_det,
                    signature=sig,
                    discovery_phase=phase,
                    walltime_seconds=result.walltime_seconds,
                )
            )
            no_new = 0
        else:
            no_new += 1
            if no_new >= patience:
                break

    basins.sort(key=lambda b: -b.abs_det)
    return BasinDiscoveryResult(
        n=n,
        basins=basins,
        phases_run=len(history),
        walltime_seconds=time.monotonic() - t_start_clock,
        rejection_history=history,
    )


# ---------------------------------------------------------------------------
# Unlocked bridge run
# ---------------------------------------------------------------------------


@dataclass
class BridgeResult:
    seed_signature: ClassSignature
    final_matrix: Matrix
    final_abs_det: int
    final_signature: ClassSignature
    drifted: bool
    walltime_seconds: float


def bridge_from_seed(
    seed_matrix: Matrix,
    *,
    iterations: int = 200_000,
    t_start: float = 0.5,
    t_end: float = 0.002,
    seed: int | None = None,
) -> BridgeResult:
    """Run plain incremental SA from ``seed_matrix`` with no class tabu.

    Returns the best matrix found, plus a flag indicating whether the
    final class signature differs from the seed's class signature
    (i.e. SA "bridged" out of the seed's basin).
    """
    seed_sig = class_signature(seed_matrix.entries)
    t0 = time.monotonic()
    res = sa_incremental(
        seed_matrix.n,
        iterations=iterations,
        t_start=t_start,
        t_end=t_end,
        seed=seed,
        initial=seed_matrix,
        refresh_every=2000,
    )
    final_sig = class_signature(res.matrix.entries)
    walltime = time.monotonic() - t0
    return BridgeResult(
        seed_signature=seed_sig,
        final_matrix=res.matrix,
        final_abs_det=res.abs_det,
        final_signature=final_sig,
        drifted=(final_sig != seed_sig),
        walltime_seconds=walltime,
    )


__all__ = [
    "ClassSignature",
    "class_signature",
    "ClassTabuResult",
    "sa_with_class_tabu",
    "Basin",
    "BasinDiscoveryResult",
    "iterative_basin_discovery",
    "BridgeResult",
    "bridge_from_seed",
]
