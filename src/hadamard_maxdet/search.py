"""Simulated annealing baseline for Hadamard maximal determinant search.

The solver starts from a random ±1 matrix (or a provided seed), proposes
moves via :mod:`hadamard_maxdet.moves`, and accepts them under a Metropolis
criterion with a geometrically-decreasing temperature schedule.

Energy is ``-log|det M|``. Working in log space keeps the Metropolis
exponent well-scaled at all orders: |det| spans many orders of magnitude,
and annealing on raw |det| would require wildly different temperatures
for different n. Log-space energy makes a single t_start / t_end pair
sensible across orders 5 through 40.

This module is deliberately simple: no gradient information, no exact
rank-1 updates (which would accelerate entry_flip scoring by an order of
magnitude). It exists to establish a reproducible lower bound against
which more elaborate solvers can be compared. Iterate on ``solver.py``.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.moves import random_move
from hadamard_maxdet.verifier import integer_determinant

# Re-export so callers can pick up the fast incremental SA via `search`.
# (Kept as a late import at module bottom to avoid circularity.)


@dataclass
class AnnealResult:
    """Outcome of a single simulated-annealing run."""

    matrix: Matrix
    abs_det: int
    iterations: int
    accepted: int
    walltime_seconds: float
    history: list[int] = field(default_factory=list)


def _log_abs_det(M: list[list[int]]) -> float:
    d = integer_determinant(M)
    if d == 0:
        return -math.inf
    return math.log(abs(d))


def simulated_annealing(
    n: int,
    *,
    iterations: int = 100_000,
    t_start: float = 1.0,
    t_end: float = 0.01,
    seed: int | None = None,
    move_fn: Callable = random_move,
    initial: Matrix | None = None,
    log_every: int = 0,
) -> AnnealResult:
    """Run simulated annealing to maximise |det M|.

    Parameters
    ----------
    n
        Matrix order.
    iterations
        Number of MC steps.
    t_start, t_end
        Temperatures at start and end; geometric schedule interpolates
        between them in log|det| units.
    seed
        PRNG seed; if ``None``, a system-entropy seed is used.
    move_fn
        Callable ``(entries, rng) -> new_entries | None`` proposing a move.
    initial
        Optional starting matrix; defaults to a fresh uniform random ±1 matrix.
    log_every
        If positive, append best-|det|-so-far to ``history`` every this
        many iterations.
    """
    rng = random.Random(seed)
    if initial is not None:
        if initial.n != n:
            raise ValueError(f"initial matrix has order {initial.n}, expected {n}")
        current = initial.copy()
    else:
        current = Matrix.random(n, rng)

    current_e = -_log_abs_det(current.entries)
    best_entries = [row[:] for row in current.entries]
    best_e = current_e

    accepted = 0
    history: list[int] = []
    start = time.monotonic()
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        proposal = move_fn(current.entries, rng)
        if proposal is None:
            if log_every and step % log_every == 0 and best_e != math.inf:
                history.append(int(round(math.exp(-best_e))) if best_e != math.inf else 0)
            continue

        proposal_e = -_log_abs_det(proposal)
        if proposal_e == math.inf:
            # det = 0; skip.
            continue
        delta = proposal_e - current_e
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            current = Matrix(n=n, entries=proposal)
            current_e = proposal_e
            accepted += 1
            if current_e < best_e:
                best_e = current_e
                best_entries = [row[:] for row in current.entries]

        if log_every and step % log_every == 0 and best_e != math.inf:
            history.append(int(round(math.exp(-best_e))))

    walltime = time.monotonic() - start
    best_abs_det = abs(integer_determinant(best_entries))
    return AnnealResult(
        matrix=Matrix(n=n, entries=best_entries),
        abs_det=best_abs_det,
        iterations=iterations,
        accepted=accepted,
        walltime_seconds=walltime,
        history=history,
    )


def multi_start(
    n: int,
    *,
    n_starts: int = 4,
    iterations: int = 100_000,
    t_start: float = 1.0,
    t_end: float = 0.01,
    base_seed: int = 0,
    initial: Matrix | None = None,
    log_every: int = 0,
) -> AnnealResult:
    """Run ``n_starts`` independent annealing runs and return the best."""
    best: AnnealResult | None = None
    for k in range(n_starts):
        r = simulated_annealing(
            n,
            iterations=iterations,
            t_start=t_start,
            t_end=t_end,
            seed=base_seed + k,
            initial=initial,
            log_every=log_every,
        )
        if best is None or r.abs_det > best.abs_det:
            best = r
    assert best is not None
    return best


def basin_hopping(
    n: int,
    *,
    iterations_per_basin: int = 50_000,
    n_basins: int = 10,
    reheat_t_start: float = 1.5,
    reheat_t_end: float = 0.01,
    seed: int | None = None,
    move_fn: Callable = random_move,
) -> AnnealResult:
    """Iterated simulated annealing.

    Each basin is one SA run that starts from the best configuration found
    so far (not from a random matrix), with a fresh high-temperature reheat
    followed by cooling. Escapes local basins that a single long SA gets
    stuck in.
    """
    rng = random.Random(seed)
    start = time.monotonic()

    current_result = simulated_annealing(
        n,
        iterations=iterations_per_basin,
        t_start=reheat_t_start,
        t_end=reheat_t_end,
        seed=rng.randrange(1 << 30),
        move_fn=move_fn,
    )
    best_matrix = current_result.matrix.copy()
    best_abs_det = current_result.abs_det
    total_accepted = current_result.accepted

    for _ in range(n_basins - 1):
        r = simulated_annealing(
            n,
            iterations=iterations_per_basin,
            t_start=reheat_t_start,
            t_end=reheat_t_end,
            seed=rng.randrange(1 << 30),
            move_fn=move_fn,
            initial=best_matrix,
        )
        total_accepted += r.accepted
        if r.abs_det > best_abs_det:
            best_matrix = r.matrix.copy()
            best_abs_det = r.abs_det

    walltime = time.monotonic() - start
    return AnnealResult(
        matrix=best_matrix,
        abs_det=best_abs_det,
        iterations=iterations_per_basin * n_basins,
        accepted=total_accepted,
        walltime_seconds=walltime,
    )


# Re-export the rank-1 incremental SA so callers can import it from
# ``hadamard_maxdet.search`` alongside the baseline.
from hadamard_maxdet.incremental import (  # noqa: E402
    IncrementalAnnealResult,
    sa_incremental,
)

__all__ = [  # explicit so the re-exports are discoverable
    "AnnealResult",
    "simulated_annealing",
    "multi_start",
    "basin_hopping",
    "IncrementalAnnealResult",
    "sa_incremental",
]
