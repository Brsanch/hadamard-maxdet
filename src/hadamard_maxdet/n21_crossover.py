"""Cross-equivalence-class genetic crossover for n=21.

Take two distinct local optima (BTC seed at 51.2T and block-cyclic enum
at 41.2T) and recombine row-by-row: each row of the offspring is taken
from one parent uniformly at random. Then refine with SA-incremental
on the unconstrained ±1 lattice.

Tests whether the n=21 optimum lives in a hybrid space between the two
known basins. If both parents are in the same Hadamard-equivalence class
(post-SA convergence), crossover should produce nothing new; if they are
in distinct classes, the offspring may seed a new basin.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from hadamard_maxdet.matrix import Matrix


def row_crossover(
    parent_a: list[list[int]],
    parent_b: list[list[int]],
    rng: random.Random,
) -> list[list[int]]:
    """Row-wise uniform crossover: each row from parent_a or parent_b 50/50."""
    if len(parent_a) != len(parent_b):
        raise ValueError("parents must have same order")
    return [parent_a[i][:] if rng.random() < 0.5 else parent_b[i][:] for i in range(len(parent_a))]


def column_crossover(
    parent_a: list[list[int]],
    parent_b: list[list[int]],
    rng: random.Random,
) -> list[list[int]]:
    """Column-wise uniform crossover."""
    n = len(parent_a)
    out = [[0] * n for _ in range(n)]
    for j in range(n):
        src = parent_a if rng.random() < 0.5 else parent_b
        for i in range(n):
            out[i][j] = src[i][j]
    return out


def block_crossover(
    parent_a: list[list[int]],
    parent_b: list[list[int]],
    rng: random.Random,
    block_size: int = 7,
) -> list[list[int]]:
    """Block-wise uniform crossover (default 7x7 blocks for n=21)."""
    n = len(parent_a)
    out = [[0] * n for _ in range(n)]
    for bi in range(0, n, block_size):
        for bj in range(0, n, block_size):
            src = parent_a if rng.random() < 0.5 else parent_b
            for di in range(min(block_size, n - bi)):
                for dj in range(min(block_size, n - bj)):
                    out[bi + di][bj + dj] = src[bi + di][bj + dj]
    return out


__all__ = [
    "row_crossover",
    "column_crossover",
    "block_crossover",
]
