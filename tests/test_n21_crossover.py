"""Sanity tests for n=21 crossover operators."""

from __future__ import annotations

import random

from hadamard_maxdet.n21_crossover import (
    block_crossover,
    column_crossover,
    row_crossover,
)


def test_row_crossover_preserves_pm1_and_size():
    rng = random.Random(0)
    A = [[1] * 21 for _ in range(21)]
    B = [[-1] * 21 for _ in range(21)]
    C = row_crossover(A, B, rng)
    assert len(C) == 21
    for row in C:
        assert len(row) == 21
        # Each row must come from A (all +1) or B (all -1)
        assert all(v == row[0] for v in row)
        assert row[0] in (-1, 1)


def test_column_crossover_size_and_pm1():
    rng = random.Random(0)
    A = [[1] * 21 for _ in range(21)]
    B = [[-1] * 21 for _ in range(21)]
    C = column_crossover(A, B, rng)
    assert len(C) == 21
    # Each column must come from A or B
    for j in range(21):
        col_vals = [C[i][j] for i in range(21)]
        assert all(v == col_vals[0] for v in col_vals)


def test_block_crossover_block_consistency():
    rng = random.Random(42)
    A = [[1] * 21 for _ in range(21)]
    B = [[-1] * 21 for _ in range(21)]
    C = block_crossover(A, B, rng, block_size=7)
    # Each 7x7 block should come from one parent: all-+1 or all--1
    for bi in (0, 7, 14):
        for bj in (0, 7, 14):
            v = C[bi][bj]
            for i in range(bi, bi + 7):
                for j in range(bj, bj + 7):
                    assert C[i][j] == v
