"""Richer structured ansätze for the Hadamard maximal-determinant problem.

The simple bordered-circulant ansatz (one border row + column + an
(n-1)×(n-1) circulant core) saturates well below the proven maximum at
many small n: at n=11 its exhaustive ceiling is 281,600 vs the proven
327,680, and at n=13 it tops out at 9,175,040 vs 14,929,920. To improve
on the established 2.42×10¹⁵ bordered-circulant ceiling at the open
order n=23, this module implements three richer constructions and
matching simulated-annealing routines:

  double_bordered_circulant
      Two top border rows + two left border columns + (n-2)×(n-2)
      circulant core. Border pattern is parameterised by signs in the
      2×2 corner block; core has n-2 free generator bits.

  bordered_two_circulant
      Order n = 2m+1. Inner 2m×2m block is a two-circulant
      [[A, B], [B^T, -A^T]] with A, B m×m circulants; surround with a
      border row + column. Free bits: 2m (generators) + 2m (border
      patterns up to symmetry).

  paley_deletion_seed / sa_from_paley
      Take Paley-type-I H_{n+1} (when n+1 = q+1 with q ≡ 3 (mod 4)
      prime; n=23 ⇒ q=23 prime, so this fires), enumerate the (n+1)²
      single row+column deletions, take the top-K by |det|, and run SA
      from each. Concrete construction route — no ansatz cage on the SA
      proper.

All constructors return ±1 matrices that pass :func:`hadamard_maxdet.verify`.
SA scoring is exact integer Bareiss via :func:`integer_determinant` (or
the rank-1 incremental path when the SA is over flat ±1 entries).
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass

from hadamard_maxdet.incremental import sa_incremental, IncrementalAnnealResult
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.search import AnnealResult
from hadamard_maxdet.structured import _circulant_rows, _sa_over_bits, two_circulant
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# 1. Double-bordered constructor
# ---------------------------------------------------------------------------


def double_bordered_circulant(
    core_first_row: Sequence[int],
    border_pattern: Sequence[int] | None = None,
) -> Matrix:
    """Construct a double-bordered matrix of order ``n = 2 + len(core)``.

    Layout::

        [ s00  s01 |  +1   +1  ...  +1 ]
        [ s10  s11 |  +1   +1  ...  +1 ]
        [ ----+----+--------------------]
        [  +1   +1 |                    ]
        [ -1   +1  |       CORE         ]
        [ -1   +1  |       (circulant)  ]
        [   ...    |                    ]

    The 2×2 top-left "corner" block carries 4 bits in
    ``border_pattern[0:4]`` (or defaults to ``[+1, +1, +1, -1]``). The
    next 2 columns of length n-2 below the corner carry the left
    "border column" pattern (default: alternating ±1 in each column),
    and the next 2 rows of length n-2 to the right of the corner carry
    the top "border row" pattern (default: all +1).

    To keep the parameter count manageable we use a fixed canonical
    border:
      - top two rows (excluding the 2×2 corner): both rows all +1.
      - left two columns (excluding the 2×2 corner): col 0 all -1,
        col 1 all +1.
    Free parameters are then the 4 corner bits and the n-2 core
    generator bits. The construction reduces to "ordinary" bordered
    circulant if you collapse one row/col, so it strictly contains
    that ansatz.

    Parameters
    ----------
    core_first_row
        Length-(n-2) ±1 sequence; the first row of the (n-2)×(n-2)
        circulant core.
    border_pattern
        Optional length-4 ±1 sequence ``[s00, s01, s10, s11]``
        specifying the 2×2 corner. Defaults to ``[+1, +1, +1, -1]``.
    """
    for v in core_first_row:
        if v not in (-1, 1):
            raise ValueError(f"core entries must be ±1; got {v!r}")
    m = len(core_first_row)
    n = m + 2
    if n < 4:
        raise ValueError(
            f"double-bordered ansatz requires n >= 4; got n={n}"
        )

    if border_pattern is None:
        border_pattern = [1, 1, 1, -1]
    if len(border_pattern) != 4:
        raise ValueError(
            f"border_pattern must have length 4; got {len(border_pattern)}"
        )
    for v in border_pattern:
        if v not in (-1, 1):
            raise ValueError(f"border_pattern entries must be ±1; got {v!r}")

    s00, s01, s10, s11 = border_pattern
    core = _circulant_rows(core_first_row)

    entries = [[0] * n for _ in range(n)]
    # 2x2 corner
    entries[0][0] = s00
    entries[0][1] = s01
    entries[1][0] = s10
    entries[1][1] = s11
    # Top two rows, columns 2..n-1: all +1
    for j in range(2, n):
        entries[0][j] = 1
        entries[1][j] = 1
    # Left two columns, rows 2..n-1: col 0 all -1, col 1 all +1
    for i in range(2, n):
        entries[i][0] = -1
        entries[i][1] = 1
    # Inner core
    for i in range(m):
        for j in range(m):
            entries[i + 2][j + 2] = core[i][j]
    return Matrix(n=n, entries=entries)


def sa_double_bordered_circulant(
    n: int,
    *,
    iterations: int = 100_000,
    t_start: float = 1.0,
    t_end: float = 0.01,
    seed: int | None = None,
    initial_core: Sequence[int] | None = None,
    initial_border: Sequence[int] | None = None,
    log_every: int = 0,
) -> AnnealResult:
    """SA over the (n-2)+4 free bits of a double-bordered circulant.

    Bits 0..n-3: core generator. Bits n-2..n+1: 2×2 corner pattern.
    Proposals flip one bit and rebuild the full matrix.
    """
    if n < 4:
        raise ValueError(f"double-bordered ansatz requires n >= 4; got n={n}")
    m = n - 2
    n_bits = m + 4

    initial_bits: list[int] | None = None
    if initial_core is not None or initial_border is not None:
        if initial_core is None:
            initial_core = [1] * m
        if initial_border is None:
            initial_border = [1, 1, 1, -1]
        if len(initial_core) != m:
            raise ValueError(f"initial_core must have length n-2={m}")
        if len(initial_border) != 4:
            raise ValueError("initial_border must have length 4")
        initial_bits = list(initial_core) + list(initial_border)

    def build(bits: list[int]) -> list[list[int]]:
        core = bits[:m]
        border = bits[m:]
        return double_bordered_circulant(core, border).entries

    return _sa_over_bits(
        build,
        n_bits,
        n_order=n,
        iterations=iterations,
        t_start=t_start,
        t_end=t_end,
        seed=seed,
        initial_bits=initial_bits,
        log_every=log_every,
    )


# ---------------------------------------------------------------------------
# 2. Bordered two-circulant
# ---------------------------------------------------------------------------


def bordered_two_circulant(
    a: Sequence[int],
    b: Sequence[int],
    border_row: Sequence[int] | None = None,
    border_col: Sequence[int] | None = None,
    corner: int = 1,
) -> Matrix:
    """Construct a bordered two-circulant of order ``n = 2m+1``.

    Layout::

        [ corner |   border_row   ]
        [ -------+----------------]
        [        |  [[ A,    B ],  ]
        [ b_col  |   [B^T, -A^T]] ]

    where the inner 2m×2m block is ``two_circulant(a, b)`` with A, B
    m×m circulants generated by ``a, b``.

    Parameters
    ----------
    a, b
        Length-m ±1 generators for A, B.
    border_row
        Length-2m ±1 row pattern (default: all +1).
    border_col
        Length-2m ±1 column pattern (default: all +1).
    corner
        Top-left scalar entry (default +1).
    """
    if len(a) != len(b):
        raise ValueError(
            f"bordered_two_circulant: |a|={len(a)} ≠ |b|={len(b)}"
        )
    m = len(a)
    inner_n = 2 * m
    n = inner_n + 1
    inner = two_circulant(a, b).entries

    if border_row is None:
        border_row = [1] * inner_n
    if border_col is None:
        border_col = [1] * inner_n
    if len(border_row) != inner_n:
        raise ValueError(
            f"border_row must have length 2m={inner_n}; got {len(border_row)}"
        )
    if len(border_col) != inner_n:
        raise ValueError(
            f"border_col must have length 2m={inner_n}; got {len(border_col)}"
        )
    if corner not in (-1, 1):
        raise ValueError(f"corner must be ±1; got {corner}")
    for v in list(border_row) + list(border_col):
        if v not in (-1, 1):
            raise ValueError(f"border entries must be ±1; got {v!r}")

    entries = [[0] * n for _ in range(n)]
    entries[0][0] = corner
    for j in range(inner_n):
        entries[0][j + 1] = border_row[j]
    for i in range(inner_n):
        entries[i + 1][0] = border_col[i]
        for j in range(inner_n):
            entries[i + 1][j + 1] = inner[i][j]
    return Matrix(n=n, entries=entries)


def sa_bordered_two_circulant(
    n: int,
    *,
    iterations: int = 100_000,
    t_start: float = 1.0,
    t_end: float = 0.01,
    seed: int | None = None,
    log_every: int = 0,
    full_border: bool = True,
) -> AnnealResult:
    """SA over the bits of a bordered two-circulant of odd order ``n=2m+1``.

    Free bits:
      - m bits for generator a
      - m bits for generator b
      - 2m bits for border_row (if ``full_border``)
      - 2m bits for border_col (if ``full_border``)
      - 1 bit for the top-left corner (if ``full_border``)

    Total: 2m if not full_border (border fixed all +1, corner +1),
    else 6m + 1.
    """
    if n % 2 == 0:
        raise ValueError(
            f"bordered_two_circulant requires odd n=2m+1; got n={n}"
        )
    m = (n - 1) // 2
    if full_border:
        n_bits = 2 * m + 4 * m + 1   # 6m+1
    else:
        n_bits = 2 * m

    def build(bits: list[int]) -> list[list[int]]:
        a = bits[:m]
        b = bits[m : 2 * m]
        if full_border:
            br = bits[2 * m : 4 * m]
            bc = bits[4 * m : 6 * m]
            corner = bits[6 * m]
        else:
            br = [1] * (2 * m)
            bc = [1] * (2 * m)
            corner = 1
        return bordered_two_circulant(a, b, br, bc, corner).entries

    return _sa_over_bits(
        build,
        n_bits,
        n_order=n,
        iterations=iterations,
        t_start=t_start,
        t_end=t_end,
        seed=seed,
        log_every=log_every,
    )


# ---------------------------------------------------------------------------
# 3. Paley-deletion seeds + SA
# ---------------------------------------------------------------------------


@dataclass
class PaleyDeletionSeed:
    """A single Paley-deletion candidate."""

    row_deleted: int
    col_deleted: int
    matrix: Matrix
    abs_det: int


def paley_deletion_seeds(n: int) -> list[PaleyDeletionSeed]:
    """Return all (n+1)² Paley-deletion seeds at order ``n``.

    Requires ``n+1 = q + 1`` with ``q`` a prime ≡ 3 (mod 4). This holds
    for n = 7 (q=7? no; q=8 not prime), n = 11 (q=11 prime ≡ 3 mod 4 ✓),
    n = 23 (q=23 prime ≡ 3 mod 4 ✓), and so on.

    The seeds are sorted in *descending* |det| order; ties broken by
    ``(row, col)``.
    """
    parent = Matrix.paley_type_i(n + 1)
    P = parent.entries
    seeds: list[PaleyDeletionSeed] = []
    for i in range(n + 1):
        for j in range(n + 1):
            sub = [
                [P[r][c] for c in range(n + 1) if c != j]
                for r in range(n + 1)
                if r != i
            ]
            d = abs(integer_determinant(sub))
            seeds.append(
                PaleyDeletionSeed(
                    row_deleted=i,
                    col_deleted=j,
                    matrix=Matrix(n=n, entries=sub),
                    abs_det=d,
                )
            )
    seeds.sort(key=lambda s: (-s.abs_det, s.row_deleted, s.col_deleted))
    return seeds


def sa_from_paley_deletion(
    n: int,
    *,
    top_k: int = 8,
    iterations_per_seed: int = 30_000,
    t_start: float = 0.5,
    t_end: float = 0.001,
    seed: int = 0,
    refresh_every: int = 0,
) -> IncrementalAnnealResult:
    """Run rank-1 SA from each of the top-K Paley deletion seeds, return best.

    The Paley-deletion seed gives a strong starting point: the parent
    H_{n+1} attains the Hadamard bound, so the (n × n) cofactors are
    large by construction. SA then explores around each via single-entry
    flips; the rank-1 incremental path (sa_incremental) makes this fast.
    """
    seeds = paley_deletion_seeds(n)
    seeds = seeds[:top_k]

    best: IncrementalAnnealResult | None = None
    for k, s in enumerate(seeds):
        r = sa_incremental(
            n,
            iterations=iterations_per_seed,
            t_start=t_start,
            t_end=t_end,
            seed=seed + k,
            initial=s.matrix,
            refresh_every=refresh_every,
        )
        if best is None or r.abs_det > best.abs_det:
            best = r
    assert best is not None
    return best


def sa_from_paley_deletion_budgeted(
    n: int,
    *,
    wall_seconds: float,
    top_k: int = 8,
    iterations_per_chunk: int = 5_000,
    t_start: float = 0.5,
    t_end: float = 0.001,
    seed: int = 0,
) -> IncrementalAnnealResult:
    """Wall-clock-budgeted Paley-deletion SA. Cycles through top-K seeds
    in chunks of ``iterations_per_chunk`` until ``wall_seconds`` exhausted.

    Returns the best :class:`IncrementalAnnealResult` over all chunks.
    """
    seeds = paley_deletion_seeds(n)[:top_k]
    start = time.monotonic()
    best: IncrementalAnnealResult | None = None
    chunk = 0
    seed_cycle = 0
    while time.monotonic() - start < wall_seconds:
        s = seeds[seed_cycle % len(seeds)]
        # Use the previous best as warm-start within a seed's basin.
        warm = (
            best.matrix if (best is not None and chunk % top_k != 0) else s.matrix
        )
        r = sa_incremental(
            n,
            iterations=iterations_per_chunk,
            t_start=t_start,
            t_end=t_end,
            seed=seed + chunk * 17 + seed_cycle,
            initial=warm,
            refresh_every=0,
        )
        if best is None or r.abs_det > best.abs_det:
            best = r
        chunk += 1
        seed_cycle += 1
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Wall-clock budgeted wrappers for benchmarking
# ---------------------------------------------------------------------------


def sa_budgeted(
    sa_fn,
    n: int,
    *,
    wall_seconds: float,
    iterations_per_chunk: int = 10_000,
    t_start: float = 1.0,
    t_end: float = 0.01,
    base_seed: int = 0,
    **kwargs,
) -> AnnealResult:
    """Run ``sa_fn`` repeatedly with fresh seeds until the budget is spent.

    ``sa_fn`` must accept ``(n, iterations=..., t_start=..., t_end=...,
    seed=..., **kwargs)`` and return an :class:`AnnealResult`.
    """
    start = time.monotonic()
    best: AnnealResult | None = None
    seed_offset = 0
    while time.monotonic() - start < wall_seconds:
        r = sa_fn(
            n,
            iterations=iterations_per_chunk,
            t_start=t_start,
            t_end=t_end,
            seed=base_seed + seed_offset,
            **kwargs,
        )
        if best is None or r.abs_det > best.abs_det:
            best = r
        seed_offset += 1
    assert best is not None
    return best


__all__ = [
    "double_bordered_circulant",
    "sa_double_bordered_circulant",
    "bordered_two_circulant",
    "sa_bordered_two_circulant",
    "PaleyDeletionSeed",
    "paley_deletion_seeds",
    "sa_from_paley_deletion",
    "sa_from_paley_deletion_budgeted",
    "sa_budgeted",
]
