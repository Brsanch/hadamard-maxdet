"""Block-cyclic ansatz for n=21: M = [[A,B,C],[C,A,B],[B,C,A]] with A,B,C of order 7.

If each of A, B, C is itself a 7x7 circulant, the state space is 2^(3·7) = 2^21
(roughly 2 million configurations) — exhaustively enumerable.

If A, B, C are free 7x7 ±1 matrices, the state space is 2^(3·49) = 2^147 — far
too big to enumerate but tractable for SA over a richer space than bordered.

This module supports both regimes:
  * `enumerate_circulant_blocks(n_workers=...)` — exhaustive enum of the
    2^21 circulant-block subfamily.
  * `sa_block_cyclic_free(...)` — SA over the 147 free bits.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def _circulant_7(generator: Sequence[int]) -> list[list[int]]:
    g = list(generator)
    return [[g[(j - i) % 7] for j in range(7)] for i in range(7)]


def block_cyclic_from_blocks(A: list[list[int]], B: list[list[int]], C: list[list[int]]) -> Matrix:
    """Build the 21x21 block-cyclic matrix [[A,B,C],[C,A,B],[B,C,A]].

    A, B, C must each be 7x7 ±1 matrices.
    """
    for blk_name, blk in (("A", A), ("B", B), ("C", C)):
        if len(blk) != 7 or any(len(r) != 7 for r in blk):
            raise ValueError(f"block {blk_name} must be 7x7")
        for row in blk:
            for v in row:
                if v not in (-1, 1):
                    raise ValueError(f"block {blk_name} entries must be ±1; got {v}")

    n = 21
    entries = [[0] * n for _ in range(n)]
    blocks = [[A, B, C], [C, A, B], [B, C, A]]
    for bi in range(3):
        for bj in range(3):
            blk = blocks[bi][bj]
            for i in range(7):
                for j in range(7):
                    entries[bi * 7 + i][bj * 7 + j] = blk[i][j]
    return Matrix(n=n, entries=entries)


def block_cyclic_circulant(a: Sequence[int], b: Sequence[int], c: Sequence[int]) -> Matrix:
    """Block-cyclic with each of A,B,C being a 7x7 circulant from generators a,b,c."""
    A = _circulant_7(a)
    B = _circulant_7(b)
    C = _circulant_7(c)
    return block_cyclic_from_blocks(A, B, C)


# ---------------------------------------------------------------------------
# Exhaustive enumeration over 2^21 circulant-block configurations
# ---------------------------------------------------------------------------


def _bits7(idx: int) -> list[int]:
    return [1 if (idx >> k) & 1 else -1 for k in range(7)]


def enumerate_circulant_blocks(
    *,
    a_start: int = 0,
    a_stop: int = 128,
    progress_every: int = 16,
) -> tuple[int, tuple[list[int], list[int], list[int]]]:
    """Exhaustively enumerate the 2^21 circulant-block subfamily.

    Returns (best_abs_det, (a_gen, b_gen, c_gen)).

    Uses the symmetry: simultaneous negation of all of A,B,C flips |det| by
    sign × 1, so |det| invariant; equivalently, fix a[0]=+1 to halve the
    space. We don't bother quotienting further since 2^20 = 1M scoring at
    O(n^3 = 9k) Bareiss is ~1.5h single-threaded — too slow. So we restrict
    a_start, a_stop and let the caller chunk.

    Defaults: scan the full a-space (0..127) — but with b,c also scanning
    0..127 each gives 128^3 = 2M Bareiss calls — at ~1 ms/call = ~30 min.
    Caller should chunk via a_start/a_stop.
    """
    best_det = 0
    best_gens: tuple[list[int], list[int], list[int]] = ([1] * 7, [1] * 7, [1] * 7)

    for ai in range(a_start, a_stop):
        a = _bits7(ai)
        for bi in range(128):
            b = _bits7(bi)
            for ci in range(128):
                c = _bits7(ci)
                M = block_cyclic_circulant(a, b, c)
                d = abs(integer_determinant(M.entries))
                if d > best_det:
                    best_det = d
                    best_gens = (list(a), list(b), list(c))
    return best_det, best_gens


# ---------------------------------------------------------------------------
# SA over the free 147-bit block-cyclic space
# ---------------------------------------------------------------------------


def sa_block_cyclic_free(
    *,
    iterations: int = 50_000,
    t_start: float = 1.0,
    t_end: float = 0.005,
    seed: int = 0,
    initial: tuple[list[list[int]], list[list[int]], list[list[int]]] | None = None,
) -> tuple[int, tuple[list[list[int]], list[list[int]], list[list[int]]], dict]:
    """SA over the 3*49 = 147 free bits of A,B,C (no circulant constraint).

    Returns (best_abs_det, (A,B,C), telemetry).
    """
    rng = random.Random(seed)

    def make_random_block() -> list[list[int]]:
        return [[1 if rng.random() < 0.5 else -1 for _ in range(7)] for _ in range(7)]

    if initial is None:
        A = make_random_block()
        B = make_random_block()
        C = make_random_block()
    else:
        A = [r[:] for r in initial[0]]
        B = [r[:] for r in initial[1]]
        C = [r[:] for r in initial[2]]

    def cur_M() -> Matrix:
        return block_cyclic_from_blocks(A, B, C)

    cur_det = abs(integer_determinant(cur_M().entries))
    tries = 0
    while cur_det == 0 and tries < 16:
        A = make_random_block(); B = make_random_block(); C = make_random_block()
        cur_det = abs(integer_determinant(cur_M().entries))
        tries += 1
    cur_e = -math.log(cur_det) if cur_det > 0 else math.inf

    best_det = cur_det
    best_blocks = ([r[:] for r in A], [r[:] for r in B], [r[:] for r in C])

    accepted = 0
    log_ratio = math.log(t_end / t_start)
    blocks = (A, B, C)

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        which = rng.randrange(3)
        i = rng.randrange(7)
        j = rng.randrange(7)
        old = blocks[which][i][j]
        blocks[which][i][j] = -old

        new_det = abs(integer_determinant(cur_M().entries))
        if new_det == 0:
            blocks[which][i][j] = old
            continue

        new_e = -math.log(new_det)
        delta = new_e - cur_e
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            cur_e = new_e
            cur_det = new_det
            accepted += 1
            if cur_det > best_det:
                best_det = cur_det
                best_blocks = (
                    [r[:] for r in blocks[0]],
                    [r[:] for r in blocks[1]],
                    [r[:] for r in blocks[2]],
                )
        else:
            blocks[which][i][j] = old

    telemetry = {
        "iterations": iterations,
        "accepted": accepted,
    }
    return best_det, best_blocks, telemetry


__all__ = [
    "block_cyclic_from_blocks",
    "block_cyclic_circulant",
    "enumerate_circulant_blocks",
    "sa_block_cyclic_free",
]
