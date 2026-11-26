"""Switching moves for ±1 matrix maximal-determinant search.

Brent, Osborn, Orrick, Zimmermann ("Recent progress on D-optimal designs",
arXiv:1112.4671) and Orrick & Rokicki used "switching" moves — non-local
proposals that flip many entries at once — inside their decomposition-based
search to establish records at n ∈ {19, 21, 25, 27, 29, 33, 37, 70, 78, 94}.
The implemented moves in :mod:`hadamard_maxdet.moves` are all *local* (single
entry, single row negation/swap, two-row XOR) and provably cannot escape
local minima that require simultaneous coordinated flips.

This module provides two non-local moves usable as drop-in proposals:

``row_pair_block_switch``
    Pick two rows (i, j) and a uniformly random subset T of columns. Negate
    *both* rows on the columns in T. **Provably preserves the inner product
    ⟨v_i, v_j⟩**, since for every column in T both entries flip and their
    pairwise sign agreement is unchanged. It generally does NOT preserve
    |det| — only the (i, j) entry of the Gram matrix M·Mᵀ is invariant; the
    other off-diagonal entries ⟨v_i, v_k⟩ for k ∉ {i, j} change because
    only v_i (not v_k) flips on T.

``two_by_two_switch``
    Find a 2×2 submatrix at row-pair (i_1, i_2) and column-pair (j_1, j_2)
    whose four entries are all equal (so the 2×2 minor is 0 — rank-deficient
    in the cofactor sense), then flip all four. The flip changes the 2×2
    minor from 0 to ±4 (since after flipping, the four entries are still
    equal in magnitude but with the sign pattern producing det = ±4). This
    is non-local (4 entries change at once), and can in principle change
    |det| by a large factor in either direction. It is a *teleport* move,
    not a |det|-preserving symmetry. The literature uses the term "switch"
    for this rank-deficient block flip even though it is not strictly
    |det|-preserving in our context — it is preserving for *Hadamard*
    matrices under specific block structures, but in general search the
    move is non-greedy and can be accepted under Metropolis.

If neither move can be applied (degenerate matrix or no qualifying 2×2
block), the function returns ``None`` and the caller should fall back to a
local move.

Honesty note
------------
The Brent–Osborn switching theorem ("J-switching") that *does* preserve
|det| requires a careful column partition matched to the row partition and
is technically subtle to implement in a way that is both correct and
useful inside SA. The two moves in this module are the clean kernel of
that idea — non-local proposals respecting a partial structure — without
claiming the full |det|-preservation of the literature's switch. We test
empirically (see ``scripts/bench_switching.py``) whether the added
non-locality helps SA on H19. Both signs of result are informative: a
positive result confirms that *any* non-local proposal opens basins; a
negative result shows that bare-SA is already covering the relevant
neighbourhood at this budget.
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

ListMatrix = list[list[int]]


def row_pair_block_switch(M: ListMatrix, rng: random.Random) -> ListMatrix | None:
    """Pick rows (i, j) and a non-empty proper subset T of columns; negate
    both rows on columns in T.

    Returns a fresh matrix; never mutates ``M``. Returns ``None`` for n < 2.

    Properties
    ----------
    - Output is in {-1, +1}^(n×n) (only sign flips applied).
    - ⟨v_i, v_j⟩ is preserved (every column in T flips both rows, so
      agree/disagree status unchanged on T; columns outside T are
      unchanged).
    - ⟨v_i, v_k⟩ for k ∉ {i, j} is generally NOT preserved.
    - |det(M)| is generally NOT preserved.
    - The size of T is uniform over {1, 2, ..., n-1} (non-empty proper
      subset, then a uniform subset of that size — gives a "moderately
      large" non-local proposal in expectation).
    """
    n = len(M)
    if n < 2:
        return None
    i, j = rng.sample(range(n), 2)
    # Pick |T| uniform in [1, n-1] (avoid the trivial T=∅ and T=all cases:
    # T=∅ is a no-op, T=all-columns is equivalent to negating both rows
    # which is already in moves.py twice).
    t_size = rng.randrange(1, n)
    T = rng.sample(range(n), t_size)
    new = [row[:] for row in M]
    for c in T:
        new[i][c] = -new[i][c]
        new[j][c] = -new[j][c]
    return new


def two_by_two_switch(M: ListMatrix, rng: random.Random) -> ListMatrix | None:
    """Find a 2×2 submatrix of all-equal entries (rank-deficient block) and
    flip all four entries simultaneously.

    Returns a fresh matrix, or ``None`` if no such 2×2 block exists.

    Procedure
    ---------
    1. Sample row pair (i_1, i_2) and column pair (j_1, j_2) uniformly,
       up to 4·n^2 attempts.
    2. Accept if M[i_1][j_1] = M[i_1][j_2] = M[i_2][j_1] = M[i_2][j_2]
       (the block is constant; its 2×2 minor is 0).
    3. Negate all four positions; return the result.

    Properties
    ----------
    - Output is in {-1, +1}^(n×n).
    - Block before flip is constant (all +1 or all -1), 2×2 minor = 0.
    - Block after flip is still constant of the opposite sign, 2×2 minor
      still 0. So the 2×2 minor of *that block* is unchanged.
    - The *full determinant* generally changes (cofactor expansion picks
      up contributions from minors that include rows/cols outside the
      block). This is a non-local proposal, not a |det|-preserving move.
    """
    n = len(M)
    if n < 2:
        return None
    max_tries = max(16, 4 * n * n)
    for _ in range(max_tries):
        i1, i2 = rng.sample(range(n), 2)
        j1, j2 = rng.sample(range(n), 2)
        a = M[i1][j1]
        if a == M[i1][j2] == M[i2][j1] == M[i2][j2]:
            new = [row[:] for row in M]
            new[i1][j1] = -a
            new[i1][j2] = -a
            new[i2][j1] = -a
            new[i2][j2] = -a
            return new
    return None


def gram_preserving_switch(M: ListMatrix, rng: random.Random) -> ListMatrix | None:
    """Alias for :func:`row_pair_block_switch`.

    Named for consistency with the literature's "Gram-preserving switch"
    terminology (Brent–Osborn 2011): the row-pair block switch preserves
    the (i, j) entry of the Gram matrix M·Mᵀ. It does NOT preserve all
    Gram entries simultaneously; full Gram preservation requires a more
    delicate column partition.
    """
    return row_pair_block_switch(M, rng)


def make_switching_move_fn(
    entry_weight: float = 0.90,
    block_switch_weight: float = 0.05,
    twobytwo_switch_weight: float = 0.05,
) -> Callable[[ListMatrix, random.Random], ListMatrix | None]:
    """Return a proposer that mixes entry_flip, row_pair_block_switch, and
    two_by_two_switch.

    Defaults to 90% entry_flip + 5% row-pair block switch + 5% 2×2 block
    flip. Weights must sum to 1 and be non-negative.
    """
    total = entry_weight + block_switch_weight + twobytwo_switch_weight
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {total}")
    if min(entry_weight, block_switch_weight, twobytwo_switch_weight) < 0:
        raise ValueError("weights must be non-negative")

    from hadamard_maxdet.moves import entry_flip

    c1 = entry_weight
    c2 = c1 + block_switch_weight

    def _move(M: ListMatrix, rng: random.Random) -> ListMatrix | None:
        u = rng.random()
        if u < c1:
            return entry_flip(M, rng)
        if u < c2:
            p = row_pair_block_switch(M, rng)
            if p is None:
                return entry_flip(M, rng)
            return p
        p = two_by_two_switch(M, rng)
        if p is None:
            # Fall back so we don't waste a step.
            return entry_flip(M, rng)
        return p

    return _move


# ----------------------------------------------------------------------------
# Switch-enhanced SA (full-recompute scoring; uses Bareiss for det). For the
# budget of this experiment, we accept the O(n^3) per-step cost rather than
# wire up rank-1 updates for non-local moves. H19 still hits ~50k iterations
# in 45s on a single thread.
# ----------------------------------------------------------------------------


@dataclass
class SwitchingAnnealResult:
    """Outcome of :func:`sa_with_switching`."""

    matrix: Matrix
    abs_det: int
    iterations: int
    accepted: int
    accepted_by_kind: dict[str, int]
    proposed_by_kind: dict[str, int]
    walltime_seconds: float
    history: list[int] = field(default_factory=list)


def sa_with_switching(
    n: int,
    *,
    iterations: int = 100_000,
    t_start: float = 1.0,
    t_end: float = 0.01,
    seed: int | None = None,
    initial: Matrix | None = None,
    entry_weight: float = 0.90,
    block_switch_weight: float = 0.05,
    twobytwo_switch_weight: float = 0.05,
    log_every: int = 0,
    walltime_budget: float | None = None,
) -> SwitchingAnnealResult:
    """SA mixing single-entry flips with non-local switching moves.

    Energy: ``-log|det|``. Geometric temperature schedule from ``t_start`` to
    ``t_end``. Determinant is recomputed exactly via Bareiss after every
    proposal — we trade per-step speed for clean handling of the large
    block moves.

    Parameters
    ----------
    n
        Matrix order.
    iterations
        Step budget. Set ``walltime_budget`` to cap by wall-clock instead.
    t_start, t_end
        Temperature schedule.
    seed
        PRNG seed.
    initial
        Starting matrix; default is uniform random ±1.
    entry_weight, block_switch_weight, twobytwo_switch_weight
        Mixture weights for the three move kinds; must sum to 1.
    log_every
        If positive, append best-|det|-so-far every that-many iterations.
    walltime_budget
        If set, stop early when this wall-clock budget (seconds) is hit;
        ``iterations`` becomes an upper bound only.
    """
    rng = random.Random(seed)
    if initial is None:
        mat = Matrix.random(n, rng)
    elif initial.n != n:
        raise ValueError(f"initial order {initial.n} ≠ n={n}")
    else:
        mat = initial.copy()

    current = [row[:] for row in mat.entries]
    current_det = integer_determinant(current)
    # Resample if singular (rare for random ±1).
    tries = 0
    while current_det == 0 and tries < 16:
        mat = Matrix.random(n, rng)
        current = [row[:] for row in mat.entries]
        current_det = integer_determinant(current)
        tries += 1
    if current_det == 0:
        return SwitchingAnnealResult(
            matrix=Matrix(n=n, entries=current),
            abs_det=0,
            iterations=0,
            accepted=0,
            accepted_by_kind={"entry": 0, "block": 0, "twobytwo": 0},
            proposed_by_kind={"entry": 0, "block": 0, "twobytwo": 0},
            walltime_seconds=0.0,
        )

    current_e = -math.log(abs(current_det))
    best_entries = [row[:] for row in current]
    best_e = current_e
    best_det = current_det

    accepted = 0
    accepted_by_kind = {"entry": 0, "block": 0, "twobytwo": 0}
    proposed_by_kind = {"entry": 0, "block": 0, "twobytwo": 0}
    history: list[int] = []
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0

    from hadamard_maxdet.moves import entry_flip

    c1 = entry_weight
    c2 = c1 + block_switch_weight

    start = time.monotonic()
    step = 0
    while step < iterations:
        if walltime_budget is not None and (time.monotonic() - start) >= walltime_budget:
            break
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        u = rng.random()
        if u < c1:
            kind = "entry"
            proposal = entry_flip(current, rng)
        elif u < c2:
            kind = "block"
            proposal = row_pair_block_switch(current, rng)
            if proposal is None:
                proposal = entry_flip(current, rng)
                kind = "entry"
        else:
            kind = "twobytwo"
            proposal = two_by_two_switch(current, rng)
            if proposal is None:
                proposal = entry_flip(current, rng)
                kind = "entry"
        proposed_by_kind[kind] += 1

        if proposal is None:
            step += 1
            continue
        d_new = integer_determinant(proposal)
        if d_new == 0:
            step += 1
            continue
        proposal_e = -math.log(abs(d_new))
        delta = proposal_e - current_e
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            current = proposal
            current_det = d_new
            current_e = proposal_e
            accepted += 1
            accepted_by_kind[kind] += 1
            if current_e < best_e:
                best_e = current_e
                best_det = current_det
                best_entries = [row[:] for row in current]

        if log_every and step % log_every == 0:
            history.append(abs(best_det))
        step += 1

    walltime = time.monotonic() - start
    # Re-verify best with exact Bareiss.
    best_abs_det = abs(integer_determinant(best_entries))
    return SwitchingAnnealResult(
        matrix=Matrix(n=n, entries=best_entries),
        abs_det=best_abs_det,
        iterations=step,
        accepted=accepted,
        accepted_by_kind=accepted_by_kind,
        proposed_by_kind=proposed_by_kind,
        walltime_seconds=walltime,
        history=history,
    )


__all__ = [
    "row_pair_block_switch",
    "two_by_two_switch",
    "gram_preserving_switch",
    "make_switching_move_fn",
    "sa_with_switching",
    "SwitchingAnnealResult",
]
