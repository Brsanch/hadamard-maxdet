"""Move operators for Monte Carlo / simulated annealing on ±1 matrices.

All moves take an ``n × n`` ±1 matrix (as a list of rows) plus a
``random.Random`` instance, and return a new matrix describing the proposal.
The input matrix is never modified.

Implemented moves:

entry_flip
    Flip a single entry sign. Changes |det| by 2·|cofactor|, i.e. this is
    the canonical local move in the ±1 lattice.

row_negate / col_negate
    Negate an entire row or column. |det| is invariant; these are pure
    symmetries, useful only when combined with other moves to escape
    reflective basins during annealing.

row_swap / col_swap
    Swap two rows or two columns. |det| is invariant (det changes sign);
    also a symmetry move.

two_row_xor
    Replace row i by the entrywise product of rows i and j (i.e. row_i ⊙
    row_j in ±1 arithmetic). This is NOT a symmetry of the problem — it
    is a larger-scale perturbation, analogous to a pull move in HP folding.

random_move
    Sample from the implemented move set with default weights.
"""

from __future__ import annotations

import random

Matrix = list[list[int]]


def entry_flip(M: Matrix, rng: random.Random) -> Matrix:
    """Flip a single ±1 entry sign at a uniformly random position."""
    n = len(M)
    i = rng.randrange(n)
    j = rng.randrange(n)
    new = [row[:] for row in M]
    new[i][j] = -new[i][j]
    return new


def row_negate(M: Matrix, rng: random.Random) -> Matrix:
    """Negate a random row. |det| is invariant."""
    n = len(M)
    i = rng.randrange(n)
    new = [row[:] for row in M]
    new[i] = [-x for x in new[i]]
    return new


def col_negate(M: Matrix, rng: random.Random) -> Matrix:
    """Negate a random column. |det| is invariant."""
    n = len(M)
    j = rng.randrange(n)
    new = [row[:] for row in M]
    for i in range(n):
        new[i][j] = -new[i][j]
    return new


def row_swap(M: Matrix, rng: random.Random) -> Matrix | None:
    """Swap two distinct random rows. |det| is invariant."""
    n = len(M)
    if n < 2:
        return None
    i, j = rng.sample(range(n), 2)
    new = [row[:] for row in M]
    new[i], new[j] = new[j], new[i]
    return new


def col_swap(M: Matrix, rng: random.Random) -> Matrix | None:
    """Swap two distinct random columns. |det| is invariant."""
    n = len(M)
    if n < 2:
        return None
    i, j = rng.sample(range(n), 2)
    new = [row[:] for row in M]
    for r in range(n):
        new[r][i], new[r][j] = new[r][j], new[r][i]
    return new


def two_row_xor(M: Matrix, rng: random.Random) -> Matrix | None:
    """Replace row i with the entrywise product row_i ⊙ row_j.

    In ±1 arithmetic, this is equivalent to XOR of the corresponding
    {0, 1} encodings. Not a symmetry; alters |det| nontrivially.
    """
    n = len(M)
    if n < 2:
        return None
    i, j = rng.sample(range(n), 2)
    new = [row[:] for row in M]
    new[i] = [new[i][k] * new[j][k] for k in range(n)]
    return new


def random_move(M: Matrix, rng: random.Random) -> Matrix | None:
    """Propose a random move by sampling from the implemented move set.

    Default weights: 80% entry_flip, 10% two_row_xor, 5% row_swap, 5% col_swap.
    entry_flip dominates because it is the only move that changes |det| by
    a small amount; the others are either invariants or large jumps.
    """
    u = rng.random()
    if u < 0.80:
        return entry_flip(M, rng)
    if u < 0.90:
        return two_row_xor(M, rng)
    if u < 0.95:
        return row_swap(M, rng)
    return col_swap(M, rng)


def make_move_fn(
    entry_weight: float = 0.80,
    xor_weight: float = 0.10,
    row_swap_weight: float = 0.05,
    col_swap_weight: float = 0.05,
):
    """Return a move proposer with custom weights. Weights must sum to 1."""
    total = entry_weight + xor_weight + row_swap_weight + col_swap_weight
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"move weights must sum to 1 (got {total})")
    if min(entry_weight, xor_weight, row_swap_weight, col_swap_weight) < 0:
        raise ValueError("move weights must be non-negative")

    c1 = entry_weight
    c2 = c1 + xor_weight
    c3 = c2 + row_swap_weight

    def _move(M: Matrix, rng: random.Random) -> Matrix | None:
        u = rng.random()
        if u < c1:
            return entry_flip(M, rng)
        if u < c2:
            return two_row_xor(M, rng)
        if u < c3:
            return row_swap(M, rng)
        return col_swap(M, rng)

    return _move


MOVE_STRATEGIES: dict[str, tuple[float, float, float, float]] = {
    "mixed":       (0.80, 0.10, 0.05, 0.05),
    "entry_only":  (1.00, 0.00, 0.00, 0.00),
    "entry_heavy": (0.95, 0.05, 0.00, 0.00),
    "with_xor":    (0.70, 0.20, 0.05, 0.05),
}
