"""Hippocampal Replay with Row-Crossover Consolidation for H23 maxdet.

Neuroscience analogy
--------------------
During waking exploration a rat's hippocampal place cells fire in sequences
that encode the visited trajectory. During subsequent sleep, sharp-wave ripple
(SWR) events replay compressed versions of these sequences (Wilson & McNaughton
1994; Skaggs & McNaughton 1996). The replay is not exact: it includes forward
replay (predicting upcoming states), reverse replay (propagating reward back
along paths), and pattern-completion (reconstructing a full memory from a
partial cue). Over many consolidation cycles the hippocampus-neocortex system
abstracts structural regularities from discrete episodic memories.

Algorithm
---------
We maintain an experience buffer of the top-K ±1 matrices encountered during
exploration (episodic memory). Two phases alternate:

Exploration phase
  Run short SA chains from random starts. Any matrix exceeding a quality
  threshold is stored in the buffer (replacing the weakest incumbent if the
  buffer is full).

Consolidation phase (SWR replay)
  1. Sample two matrices A, B from the buffer, weighted by |det|.
  2. For each row i independently draw a Bernoulli(0.5) coin: take row_i
     from A if heads, from B if tails. This is "row crossover" — a direct
     analog of hippocampal pattern-completion from two stored episodes.
  3. SA-refine the hybrid from a warm start (lower initial temperature
     because the starting point is already structured).
  4. If the refined result exceeds the buffer's minimum, replace it.

Key distinction from CEM (information_theory domain)
  CEM updates marginal entry-level Bernoulli distributions, averaging over
  all stored elites and discarding all row structure. Row crossover preserves
  entire ±1 rows from actual high-quality matrices; inner-product relationships
  within each borrowed row are maintained intact.

Key distinction from PT (thermodynamics domain)
  PT couples replicas via temperature swaps; its recombination mechanism is
  indirect (a hot replica visiting a good basin, then swapping toward cold).
  Hippocampal replay explicitly recombines stored memories into structured
  hybrid starting points.

Key distinction from BRD (game_theory domain)
  BRD does single-entry greedy ascent from a fixed start. Hippocampal replay
  creates a population of starting points via row crossover and runs SA on
  each, achieving coverage of the landscape through structured recombination
  rather than through many independent random starts.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from hadamard_maxdet.incremental import (
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    refresh,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant

N23 = 23
ORRICK_ABS_DET_H23: int = 2_779_447_296_000_000


# ---------------------------------------------------------------------------
# Result carrier
# ---------------------------------------------------------------------------


@dataclass
class HRResult:
    """Outcome of a hippocampal replay run."""

    best_matrix: list[list[int]]
    best_abs_det: int
    total_sa_steps: int
    walltime_seconds: float
    n_exploration_chains: int
    n_consolidation_chains: int
    buffer_size_final: int
    best_ratio_to_orrick: float
    above_orrick: bool


# ---------------------------------------------------------------------------
# Experience buffer
# ---------------------------------------------------------------------------


class ExperienceBuffer:
    """Fixed-capacity priority buffer of (abs_det, matrix) pairs.

    Maintains the top-K matrices seen so far.  New entries replace the worst
    incumbent only if they improve upon it.
    """

    def __init__(self, capacity: int = 20, threshold_ratio: float = 0.5) -> None:
        self.capacity = capacity
        # Minimum |det| required to enter the buffer at all.
        self.threshold = int(threshold_ratio * ORRICK_ABS_DET_H23)
        # items: list of (abs_det, matrix_entries)
        self._items: list[tuple[int, list[list[int]]]] = []

    def try_add(self, abs_det: int, entries: list[list[int]]) -> bool:
        """Add entry if it improves buffer.  Returns True if accepted."""
        if abs_det < self.threshold:
            return False
        if len(self._items) < self.capacity:
            self._items.append((abs_det, [row[:] for row in entries]))
            self._items.sort(key=lambda x: -x[0])  # descending
            return True
        if abs_det > self._items[-1][0]:
            self._items[-1] = (abs_det, [row[:] for row in entries])
            self._items.sort(key=lambda x: -x[0])
            return True
        return False

    def __len__(self) -> int:
        return len(self._items)

    def min_det(self) -> int:
        return self._items[-1][0] if self._items else 0

    def max_det(self) -> int:
        return self._items[0][0] if self._items else 0

    def weighted_sample_pair(self, rng: random.Random) -> tuple[
        list[list[int]], list[list[int]]
    ]:
        """Sample two distinct matrices weighted proportional to |det|^2.

        We use |det|^2 weighting so that higher-quality matrices are much more
        likely to be chosen as parents, mimicking the stronger reinstatement of
        frequent/salient memories in hippocampal replay.

        Falls back to uniform if all weights are equal.
        """
        n = len(self._items)
        if n < 2:
            raise ValueError("Buffer has fewer than 2 entries; cannot sample a pair.")
        dets = [x[0] for x in self._items]
        weights = [d * d for d in dets]
        total = sum(weights)
        # Sample index i
        r = rng.random() * total
        cum = 0.0
        idx_a = 0
        for k, w in enumerate(weights):
            cum += w
            if cum >= r:
                idx_a = k
                break
        # Sample index j ≠ i
        total2 = total - weights[idx_a]
        r2 = rng.random() * total2
        cum2 = 0.0
        idx_b = -1
        for k, w in enumerate(weights):
            if k == idx_a:
                continue
            cum2 += w
            if cum2 >= r2:
                idx_b = k
                break
        if idx_b == -1:
            idx_b = 1 if idx_a != 1 else 0
        return self._items[idx_a][1], self._items[idx_b][1]

    def all_matrices(self) -> list[tuple[int, list[list[int]]]]:
        return list(self._items)


# ---------------------------------------------------------------------------
# Row crossover (pattern completion)
# ---------------------------------------------------------------------------


def row_crossover(
    A: list[list[int]],
    B: list[list[int]],
    rng: random.Random,
    bias_a: float = 0.5,
) -> list[list[int]]:
    """Create a hybrid ±1 matrix by row-wise crossover of A and B.

    For each row i: take row from A with probability bias_a, from B otherwise.
    bias_a=0.5 gives equal weight; bias_a > 0.5 biases toward parent A.

    The resulting hybrid preserves complete ±1 row vectors from real
    high-quality matrices, unlike per-entry sampling (CEM) which destroys
    all row structure.
    """
    n = len(A)
    hybrid = []
    for i in range(n):
        if rng.random() < bias_a:
            hybrid.append(A[i][:])
        else:
            hybrid.append(B[i][:])
    return hybrid


def column_crossover(
    A: list[list[int]],
    B: list[list[int]],
    rng: random.Random,
) -> list[list[int]]:
    """Create a hybrid by column-wise crossover of A and B."""
    n = len(A)
    # Pick which columns come from A
    use_a = [rng.random() < 0.5 for _ in range(n)]
    hybrid = [[0] * n for _ in range(n)]
    for j in range(n):
        src = A if use_a[j] else B
        for i in range(n):
            hybrid[i][j] = src[i][j]
    return hybrid


# ---------------------------------------------------------------------------
# Fast single-thread SA (inline, reuses IncrementalState)
# ---------------------------------------------------------------------------


def _sa_chain(
    n: int,
    *,
    initial: list[list[int]] | None = None,
    iterations: int = 50_000,
    t_start: float = 1.0,
    t_end: float = 0.003,
    rng: random.Random,
) -> tuple[list[list[int]], int]:
    """Run SA on a ±1 matrix; return (best_matrix, best_abs_det).

    Uses IncrementalState for O(n^2) per-step adjugate updates.
    Falls back to a fresh random matrix if initial is singular.
    """
    if initial is None:
        mat = Matrix.random(n, rng)
        state = IncrementalState.from_matrix(mat.entries)
    else:
        state = IncrementalState.from_matrix(initial)

    # Ensure non-singular start
    attempts = 0
    while state.det == 0 and attempts < 16:
        mat = Matrix.random(n, rng)
        state = IncrementalState.from_matrix(mat.entries)
        attempts += 1
    if state.det == 0:
        # Degenerate; return zero-quality result
        return [row[:] for row in state.M], 0

    log_t_start = math.log(t_start)
    log_t_end = math.log(t_end)
    t_range = log_t_start - log_t_end

    best_abs = abs(state.det)
    best_M = [row[:] for row in state.M]

    refresh_counter = 0
    REFRESH_EVERY = 2000

    for step in range(iterations):
        # Geometric temperature schedule
        frac = step / iterations
        t = math.exp(log_t_start - frac * t_range)
        inv_t = 1.0 / t

        i = rng.randrange(n)
        j = rng.randrange(n)

        new_det = propose_entry_flip(state, i, j)
        new_abs = abs(new_det)

        if new_det == 0:
            continue  # never accept singular

        old_abs = abs(state.det)
        if new_abs >= old_abs:
            apply_entry_flip(state, i, j)
        else:
            log_ratio = math.log(new_abs) - math.log(old_abs)  # negative
            if math.log(rng.random()) < log_ratio * inv_t:
                apply_entry_flip(state, i, j)

        cur_abs = abs(state.det)
        if cur_abs > best_abs:
            best_abs = cur_abs
            best_M = [row[:] for row in state.M]

        refresh_counter += 1
        if refresh_counter >= REFRESH_EVERY:
            refresh(state)
            refresh_counter = 0

    return best_M, best_abs


# ---------------------------------------------------------------------------
# Main hippocampal replay driver
# ---------------------------------------------------------------------------


def hippocampal_replay(
    n: int = N23,
    *,
    wall_budget_seconds: float = 600.0,
    buffer_capacity: int = 20,
    buffer_threshold_ratio: float = 0.50,
    explore_iters: int = 50_000,
    consolidate_iters: int = 40_000,
    consolidate_t_start: float = 0.40,
    consolidate_t_end: float = 0.003,
    explore_t_start: float = 2.0,
    explore_t_end: float = 0.003,
    min_buffer_for_consolidation: int = 3,
    consolidation_fraction: float = 0.40,
    crossover_mode: str = "row",
    seed: int = 20260426,
) -> HRResult:
    """Run hippocampal replay optimisation on n×n ±1 matrices.

    Parameters
    ----------
    wall_budget_seconds
        Total wall-clock time allowed.
    buffer_capacity
        Maximum number of matrices in the experience buffer.
    buffer_threshold_ratio
        Minimum |det|/Orrick required to enter the buffer.
    explore_iters
        SA iterations per exploration chain.
    consolidate_iters
        SA iterations per consolidation chain (warm start → fewer needed).
    consolidate_t_start
        SA start temperature for consolidation (lower than explore since
        hybrid starting points are already structured).
    consolidate_t_end
        SA end temperature for both phases.
    explore_t_start
        SA start temperature for exploration chains.
    min_buffer_for_consolidation
        Minimum buffer occupancy before consolidation is enabled.
    consolidation_fraction
        Fraction of time steps allocated to consolidation once buffer is
        large enough.  The rest go to exploration.
    crossover_mode
        "row" (row crossover) or "col" (column crossover).
    seed
        PRNG seed.
    """
    rng = random.Random(seed)
    buf = ExperienceBuffer(capacity=buffer_capacity, threshold_ratio=buffer_threshold_ratio)

    best_overall_abs: int = 0
    best_overall_M: list[list[int]] = []

    total_sa_steps: int = 0
    n_explore: int = 0
    n_consolidate: int = 0

    t0 = time.monotonic()

    def elapsed() -> float:
        return time.monotonic() - t0

    def update_best(M: list[list[int]], abs_det: int) -> None:
        nonlocal best_overall_abs, best_overall_M
        if abs_det > best_overall_abs:
            best_overall_abs = abs_det
            best_overall_M = [row[:] for row in M]
        buf.try_add(abs_det, M)

    # -----------------------------------------------------------------------
    # Main loop: alternate exploration and consolidation
    # -----------------------------------------------------------------------
    while elapsed() < wall_budget_seconds:
        do_consolidate = (
            len(buf) >= min_buffer_for_consolidation
            and rng.random() < consolidation_fraction
        )

        if do_consolidate:
            # ---- Consolidation phase: SWR replay ----
            A, B = buf.weighted_sample_pair(rng)
            if crossover_mode == "row":
                hybrid = row_crossover(A, B, rng, bias_a=0.5)
            else:
                hybrid = column_crossover(A, B, rng)

            M_out, abs_det_out = _sa_chain(
                n,
                initial=hybrid,
                iterations=consolidate_iters,
                t_start=consolidate_t_start,
                t_end=consolidate_t_end,
                rng=rng,
            )
            total_sa_steps += consolidate_iters
            n_consolidate += 1
            update_best(M_out, abs_det_out)
        else:
            # ---- Exploration phase: independent SA from random start ----
            M_out, abs_det_out = _sa_chain(
                n,
                initial=None,
                iterations=explore_iters,
                t_start=explore_t_start,
                t_end=explore_t_end,
                rng=rng,
            )
            total_sa_steps += explore_iters
            n_explore += 1
            update_best(M_out, abs_det_out)

    wall = elapsed()

    if not best_overall_M:
        # Fallback: take best from buffer
        if buf:
            best_overall_abs, best_overall_M = buf.all_matrices()[0]

    ratio = best_overall_abs / ORRICK_ABS_DET_H23 if ORRICK_ABS_DET_H23 > 0 else 0.0

    return HRResult(
        best_matrix=best_overall_M,
        best_abs_det=best_overall_abs,
        total_sa_steps=total_sa_steps,
        walltime_seconds=wall,
        n_exploration_chains=n_explore,
        n_consolidation_chains=n_consolidate,
        buffer_size_final=len(buf),
        best_ratio_to_orrick=ratio,
        above_orrick=(best_overall_abs > ORRICK_ABS_DET_H23),
    )


# ---------------------------------------------------------------------------
# Triple verification (required for any above-Orrick claim)
# ---------------------------------------------------------------------------


def triple_verify(M: list[list[int]]) -> tuple[int, int, int]:
    """Return (det_M, det_MT, det_copy); all three must agree and be non-zero."""
    det_M = integer_determinant([row[:] for row in M])
    MT = [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]
    det_MT = integer_determinant(MT)
    copy = [row[:] for row in M]
    det_copy = integer_determinant(copy)
    return det_M, det_MT, det_copy
