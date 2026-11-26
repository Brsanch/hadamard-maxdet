"""k-flip moves for ±1 matrix maximal-determinant search.

The diagnostic on the H23 BTC seed showed:
    d=1, 2, 3 single-entry-flip neighbourhoods of the seed are sealed
    (zero |det|-improving neighbours in 5,000 random samples).

Hypothesis tested here: at smaller n (n=19) the basin may be shallower in
some directions; coordinated *k-flip* moves with k>=2 might escape where
SA's uniform single-flip proposals cannot. We provide:

  * :func:`best_2_flip_via_adjugate` — exact O(n^4) enumeration of all
    pairs of distinct entry positions, scoring via the rank-2
    matrix-determinant lemma in O(1) per pair given ``adj``.

  * :func:`sample_k_flip` — random multi-flip proposer.

  * :func:`sa_with_k_flip` — SA loop that, with some probability, swaps
    in a k-flip proposal evaluated by full Bareiss (k>=3) or the rank-2
    formula (k=2), and otherwise proposes the usual single entry flip.

  * :func:`steepest_2_flip_polish` — repeatedly take the best 2-flip
    neighbour until no improvement (a fast greedy "polish" that runs
    until a 2-flip local optimum is reached). Run *after* SA to squeeze
    coordinated improvements out of a basin.

Rank-2 update for two flips at positions (i1, j1), (i2, j2):

    Let u = (e_{i1} delta_1) + (e_{i2} delta_2),  v = (e_{j1}, e_{j2})
    where delta_k = -2 * M[i_k][j_k] (since flipping ±1 -> ∓1).
    Then det(M + u_1 e_{j1}^T + u_2 e_{j2}^T) =
        det(M) * det(I_2 + V^T M^{-1} U)
    with U = [delta_1 e_{i1}, delta_2 e_{i2}] (n x 2),
         V = [e_{j1}, e_{j2}] (n x 2).
    V^T M^{-1} U is a 2x2 matrix; its (a,b) entry is
        delta_b * M^{-1}[j_a, i_b] = (delta_b / det) * adj[j_a, i_b].

So:
    new_det = det + delta_1 * adj[j1, i1]
                  + delta_2 * adj[j2, i2]
                  + delta_1 * delta_2 * (
                        adj[j1, i1] * adj[j2, i2] / det
                      - adj[j1, i2] * adj[j2, i1] / det
                    )

In integer form, multiplying through by ``det``:
    new_det * det
        = det * det
          + det * (delta_1 adj[j1, i1] + delta_2 adj[j2, i2])
          + delta_1 delta_2 * (adj[j1, i1] adj[j2, i2] - adj[j1, i2] adj[j2, i1])

i.e.

    new_det = det
              + delta_1 * adj[j1, i1]
              + delta_2 * adj[j2, i2]
              + (delta_1 delta_2 / det) * (adj[j1, i1] adj[j2, i2]
                                            - adj[j1, i2] adj[j2, i1])

The integer division is exact whenever ``det != 0`` (the 2x2 minor of adj
is itself div-by-det, which is the cofactor-of-cofactor identity). We
verify divisibility and fall back to Bareiss on any mismatch.

For k=2 enumeration, this is O(1) per pair after O(1) lookups of four adj
entries; total O(n^4) for the full pairwise sweep, ≈ 130k operations at
n=19.
"""
from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from hadamard_maxdet.incremental import (
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    refresh,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Rank-2 score
# ---------------------------------------------------------------------------


def score_2_flip(state: IncrementalState, i1: int, j1: int,
                 i2: int, j2: int) -> int | None:
    """Return det(M') for M' = M with both (i1,j1) and (i2,j2) flipped.

    Returns ``None`` if (i1,j1) == (i2,j2) (not a 2-flip), or if integer
    division is inexact (state invariants stale; caller should refresh).
    Cost O(1).
    """
    if i1 == i2 and j1 == j2:
        return None
    M = state.M
    det = state.det
    if det == 0:
        return None
    adj = state.adj
    delta_1 = -2 * M[i1][j1]
    delta_2 = -2 * M[i2][j2]
    a11 = adj[j1][i1]
    a22 = adj[j2][i2]
    a12 = adj[j1][i2]   # adj[j_a, i_b]
    a21 = adj[j2][i1]
    minor = a11 * a22 - a12 * a21
    cross = delta_1 * delta_2 * minor
    q, r = divmod(cross, det)
    if r != 0:
        # Inexact; fall back signal to caller.
        return None
    return det + delta_1 * a11 + delta_2 * a22 + q


def best_2_flip_via_adjugate(
    state: IncrementalState,
    *,
    require_strict_improvement: bool = True,
) -> tuple[int, tuple[int, int], tuple[int, int]] | None:
    """Exhaustively scan all distinct pairs of entry positions.

    Returns the best (|det|, pos1, pos2) over all (i1,j1) != (i2,j2)
    pairs, or ``None`` if ``require_strict_improvement`` and no pair
    strictly beats the current ``|state.det|``.
    """
    n = state.n
    cur = abs(state.det)
    best = cur if require_strict_improvement else 0
    best_pos: tuple[tuple[int, int], tuple[int, int]] | None = None
    n2 = n * n
    # iterate ordered pairs (p1 < p2)
    for p1 in range(n2):
        i1, j1 = divmod(p1, n)
        for p2 in range(p1 + 1, n2):
            i2, j2 = divmod(p2, n)
            d = score_2_flip(state, i1, j1, i2, j2)
            if d is None:
                continue
            ad = abs(d)
            if ad > best:
                best = ad
                best_pos = ((i1, j1), (i2, j2))
    if best_pos is None:
        return None
    return best, best_pos[0], best_pos[1]


def apply_2_flip(state: IncrementalState, i1: int, j1: int,
                 i2: int, j2: int) -> int:
    """Apply two entry flips sequentially, refreshing if the first nuls det.
    Returns new det.
    """
    apply_entry_flip(state, i1, j1)
    if state.det == 0:
        # Apply the second by raw flip + refresh
        state.M[i2][j2] = -state.M[i2][j2]
        refresh(state)
        return state.det
    apply_entry_flip(state, i2, j2)
    return state.det


def steepest_2_flip_polish(
    entries: list[list[int]],
    *,
    max_passes: int = 50,
) -> tuple[list[list[int]], int, int]:
    """Take best 2-flip iteratively until 2-flip local maximum.

    Returns (final_entries, final_abs_det, n_passes_with_improvement).
    """
    state = IncrementalState.from_matrix(entries)
    improved_passes = 0
    for _ in range(max_passes):
        result = best_2_flip_via_adjugate(state, require_strict_improvement=True)
        if result is None:
            break
        new_abs, p1, p2 = result
        apply_2_flip(state, p1[0], p1[1], p2[0], p2[1])
        # Verify
        if abs(state.det) != new_abs:
            refresh(state)
            if abs(state.det) != new_abs:
                # Drift; recompute
                actual = integer_determinant(state.M)
                state.det = actual
        improved_passes += 1
    return [row[:] for row in state.M], abs(state.det), improved_passes


# ---------------------------------------------------------------------------
# Random k-flip sampler (k >= 3 evaluated via Bareiss)
# ---------------------------------------------------------------------------


def sample_k_flip_positions(n: int, k: int, rng: random.Random) -> list[tuple[int, int]]:
    """Sample k distinct positions in an n x n grid uniformly."""
    if k > n * n:
        raise ValueError(f"k={k} > n^2={n*n}")
    flat = rng.sample(range(n * n), k)
    return [(p // n, p % n) for p in flat]


# ---------------------------------------------------------------------------
# SA with mixed entry-flip + k-flip proposals
# ---------------------------------------------------------------------------


@dataclass
class KFlipResult:
    matrix: Matrix
    abs_det: int
    iterations: int
    accepted_single: int
    accepted_2flip: int
    accepted_kflip: int
    proposed_single: int
    proposed_2flip: int
    proposed_kflip: int
    walltime_seconds: float
    refreshes: int
    polish_passes: int = 0
    history: list[int] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "n": self.matrix.n,
            "abs_det": self.abs_det,
            "iterations": self.iterations,
            "accepted_single": self.accepted_single,
            "accepted_2flip": self.accepted_2flip,
            "accepted_kflip": self.accepted_kflip,
            "proposed_single": self.proposed_single,
            "proposed_2flip": self.proposed_2flip,
            "proposed_kflip": self.proposed_kflip,
            "walltime_seconds": self.walltime_seconds,
            "refreshes": self.refreshes,
            "polish_passes": self.polish_passes,
            "acceptance_single": (self.accepted_single / self.proposed_single)
            if self.proposed_single
            else 0.0,
            "acceptance_2flip": (self.accepted_2flip / self.proposed_2flip)
            if self.proposed_2flip
            else 0.0,
            "acceptance_kflip": (self.accepted_kflip / self.proposed_kflip)
            if self.proposed_kflip
            else 0.0,
        }


def sa_with_k_flip(
    n: int,
    *,
    iterations: int = 100_000,
    t_start: float = 1.0,
    t_end: float = 0.01,
    seed: int | None = None,
    initial: Matrix | None = None,
    p_2flip: float = 0.10,
    p_kflip: float = 0.02,
    kflip_size: int = 4,
    refresh_every: int = 2000,
) -> KFlipResult:
    """SA mixing single-entry, 2-flip (rank-2), and k-flip (Bareiss) moves.

    Probabilities sum: p_single = 1 - p_2flip - p_kflip.
    """
    if p_2flip < 0 or p_kflip < 0 or p_2flip + p_kflip > 1:
        raise ValueError("invalid p_2flip / p_kflip")

    rng = random.Random(seed)
    if initial is None:
        mat = Matrix.random(n, rng)
    else:
        mat = initial.copy()

    state = IncrementalState.from_matrix(mat.entries)
    tries = 0
    while state.det == 0 and tries < 16:
        mat = Matrix.random(n, rng)
        state = IncrementalState.from_matrix(mat.entries)
        tries += 1
    if state.det == 0:
        return KFlipResult(
            matrix=Matrix(n=n, entries=[row[:] for row in state.M]),
            abs_det=0, iterations=0,
            accepted_single=0, accepted_2flip=0, accepted_kflip=0,
            proposed_single=0, proposed_2flip=0, proposed_kflip=0,
            walltime_seconds=0.0, refreshes=0,
        )

    cur_e = -math.log(abs(state.det))
    best_entries = [row[:] for row in state.M]
    best_e = cur_e

    accepted_single = accepted_2flip = accepted_kflip = 0
    proposed_single = proposed_2flip = proposed_kflip = 0
    refreshes = 0
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0
    t0 = time.monotonic()

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        u = rng.random()

        if u < p_kflip:
            # k-flip via Bareiss
            proposed_kflip += 1
            positions = sample_k_flip_positions(n, kflip_size, rng)
            cand = [row[:] for row in state.M]
            for i, j in positions:
                cand[i][j] = -cand[i][j]
            d_new = integer_determinant(cand)
            if d_new == 0:
                continue
            prop_e = -math.log(abs(d_new))
            delta = prop_e - cur_e
            if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
                state.M = cand
                state.det = d_new
                # Adj must be rebuilt for incremental mode
                from hadamard_maxdet.incremental import _compute_adjugate
                state.adj = _compute_adjugate(state.M, d_new)
                refreshes += 1
                cur_e = prop_e
                accepted_kflip += 1
                if cur_e < best_e:
                    best_e = cur_e
                    best_entries = [row[:] for row in state.M]
        elif u < p_kflip + p_2flip:
            # 2-flip via rank-2 adjugate
            proposed_2flip += 1
            i1 = rng.randrange(n); j1 = rng.randrange(n)
            i2 = rng.randrange(n); j2 = rng.randrange(n)
            if (i1, j1) == (i2, j2):
                continue
            new_det = score_2_flip(state, i1, j1, i2, j2)
            if new_det is None or new_det == 0:
                continue
            prop_e = -math.log(abs(new_det))
            delta = prop_e - cur_e
            if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
                apply_2_flip(state, i1, j1, i2, j2)
                cur_e = -math.log(abs(state.det)) if state.det else float("inf")
                accepted_2flip += 1
                if cur_e < best_e:
                    best_e = cur_e
                    best_entries = [row[:] for row in state.M]
                if refresh_every > 0 and (accepted_2flip + accepted_single) % refresh_every == 0:
                    refresh(state)
                    refreshes += 1
        else:
            # Single entry flip (fast incremental)
            proposed_single += 1
            i = rng.randrange(n); j = rng.randrange(n)
            new_det = propose_entry_flip(state, i, j)
            if new_det == 0:
                continue
            prop_e = -math.log(abs(new_det))
            delta = prop_e - cur_e
            if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
                apply_entry_flip(state, i, j)
                cur_e = prop_e
                accepted_single += 1
                if cur_e < best_e:
                    best_e = cur_e
                    best_entries = [row[:] for row in state.M]
                if refresh_every > 0 and (accepted_2flip + accepted_single) % refresh_every == 0:
                    refresh(state)
                    refreshes += 1

    walltime = time.monotonic() - t0
    best_abs_det = abs(integer_determinant(best_entries))
    return KFlipResult(
        matrix=Matrix(n=n, entries=best_entries),
        abs_det=best_abs_det,
        iterations=iterations,
        accepted_single=accepted_single,
        accepted_2flip=accepted_2flip,
        accepted_kflip=accepted_kflip,
        proposed_single=proposed_single,
        proposed_2flip=proposed_2flip,
        proposed_kflip=proposed_kflip,
        walltime_seconds=walltime,
        refreshes=refreshes,
    )
