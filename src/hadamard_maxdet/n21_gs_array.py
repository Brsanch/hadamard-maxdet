"""Goethals–Seidel array for n = 4m+1 = 21 (m = 5).

The classical Goethals–Seidel (GS) array of order 4m has the form

    GS(A, B, C, D) =
        [[ A,    BR,   CR,   DR ],
         [-BR,    A,   D^TR, -C^TR ],
         [-CR,  -D^TR,  A,    B^TR ],
         [-DR,   C^TR, -B^TR,  A   ]]

where A, B, C, D are m×m ±1 circulant blocks and R is the m×m back-
permutation (R_{i,j} = δ_{i+j, m-1}). For an *Hadamard* GS we need

    A A^T + B B^T + C C^T + D D^T = 4m · I_m,

which forces a strong constraint on the autocorrelation function of the
generators. Equivalently, in terms of generators a, b, c, d ∈ {±1}^m, the
periodic autocorrelations must sum to 4m at lag 0 and 0 at every other
lag. Such 4-quad sequences are studied as base sequences / T-sequences /
near-Williamson families.

For n = 4m+1 we don't have a 4m × 4m → 4m+1 direct extension, but we can
use a *bordered* GS: take any GS-shaped 4m matrix and surround with a
border row + column to get order 4m+1 = 21 (with m = 5).

This module implements:
  * `gs_array(a,b,c,d, R=None)` — build the GS(A,B,C,D) order-4m matrix.
  * `bordered_gs(a,b,c,d, border_row, border_col, corner)` — order 4m+1.
  * `enum_gs_circulant_quads(m=5, ...)` — enumerate base-sequence quads
    by autocorrelation properties + score.
  * `sa_bordered_gs(...)` — SA over generator + border bits.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import random
import time
from collections.abc import Sequence

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Core constructors
# ---------------------------------------------------------------------------


def _circulant(generator: Sequence[int]) -> list[list[int]]:
    m = len(generator)
    g = list(generator)
    return [[g[(j - i) % m] for j in range(m)] for i in range(m)]


def _back_perm(m: int) -> list[list[int]]:
    """R_{i,j} = 1 if i + j == m-1 else 0."""
    return [[1 if (i + j) == (m - 1) else 0 for j in range(m)] for i in range(m)]


def _matmul_pm1(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    m = len(A)
    p = len(B[0])
    k = len(B)
    out = [[0] * p for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        for kk in range(k):
            aik = Ai[kk]
            if aik == 0:
                continue
            Bk = B[kk]
            for j in range(p):
                out[i][j] += aik * Bk[j]
    return out


def _transpose(A: list[list[int]]) -> list[list[int]]:
    m = len(A)
    n = len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def _neg(A: list[list[int]]) -> list[list[int]]:
    return [[-x for x in row] for row in A]


def gs_array(
    a: Sequence[int],
    b: Sequence[int],
    c: Sequence[int],
    d: Sequence[int],
) -> Matrix:
    """Build the Goethals–Seidel array of order 4m from 4 length-m generators.

    Layout (block 4m × 4m, blocks of size m × m):

        [[ A,    BR,    CR,    DR  ],
         [-BR,    A,    D^TR,  -C^TR ],
         [-CR,  -D^TR,   A,    B^TR ],
         [-DR,   C^TR,  -B^TR,  A  ]]
    """
    m = len(a)
    if not (len(b) == len(c) == len(d) == m):
        raise ValueError("a,b,c,d must have equal length m")
    A = _circulant(a)
    B = _circulant(b)
    C = _circulant(c)
    D = _circulant(d)
    R = _back_perm(m)
    BR = _matmul_pm1(B, R)
    CR = _matmul_pm1(C, R)
    DR = _matmul_pm1(D, R)
    BTR = _matmul_pm1(_transpose(B), R)
    CTR = _matmul_pm1(_transpose(C), R)
    DTR = _matmul_pm1(_transpose(D), R)

    n = 4 * m
    entries = [[0] * n for _ in range(n)]

    blocks = [
        [A,         BR,        CR,        DR],
        [_neg(BR),  A,         DTR,       _neg(CTR)],
        [_neg(CR),  _neg(DTR), A,         BTR],
        [_neg(DR),  CTR,       _neg(BTR), A],
    ]
    for bi in range(4):
        for bj in range(4):
            blk = blocks[bi][bj]
            for i in range(m):
                for j in range(m):
                    entries[bi * m + i][bj * m + j] = blk[i][j]
    return Matrix(n=n, entries=entries)


def bordered_gs(
    a: Sequence[int],
    b: Sequence[int],
    c: Sequence[int],
    d: Sequence[int],
    border_row: Sequence[int] | None = None,
    border_col: Sequence[int] | None = None,
    corner: int = 1,
) -> Matrix:
    """Bordered GS of order 4m + 1 from generators of length m + a border row/col.

    Inner = gs_array(a,b,c,d), surrounded by border_row (length 4m) on top
    and border_col (length 4m) on left + corner.
    """
    inner = gs_array(a, b, c, d).entries
    m = len(a)
    inner_n = 4 * m
    n = inner_n + 1

    if border_row is None:
        border_row = [1] * inner_n
    if border_col is None:
        border_col = [1] * inner_n
    if len(border_row) != inner_n or len(border_col) != inner_n:
        raise ValueError("border_row/col must have length 4m")
    if corner not in (-1, 1):
        raise ValueError("corner must be ±1")

    entries = [[0] * n for _ in range(n)]
    entries[0][0] = corner
    for j in range(inner_n):
        entries[0][j + 1] = border_row[j]
    for i in range(inner_n):
        entries[i + 1][0] = border_col[i]
        for j in range(inner_n):
            entries[i + 1][j + 1] = inner[i][j]
    return Matrix(n=n, entries=entries)


# ---------------------------------------------------------------------------
# Enumeration: 4 base sequences of length m, scored by |det| of bordered GS
# ---------------------------------------------------------------------------


def _bits_m(idx: int, m: int) -> list[int]:
    return [1 if (idx >> k) & 1 else -1 for k in range(m)]


def _enum_chunk_gs(args) -> tuple[int, tuple[int, int, int, int]]:
    """Score a slab of (a,b,c,d) ∈ [a_lo,a_hi) × [0,2^m)³ with default border.

    For m=5: 2^5=32, full quad enum is 32^4 = ~1M — tractable.
    """
    a_lo, a_hi, m = args
    best_det = 0
    best_idx = (0, 0, 0, 0)
    space = 1 << m
    for ai in range(a_lo, a_hi):
        a = _bits_m(ai, m)
        for bi in range(space):
            b = _bits_m(bi, m)
            for ci in range(space):
                c = _bits_m(ci, m)
                for di in range(space):
                    d = _bits_m(di, m)
                    # Bordered GS with default all-+1 border
                    M = bordered_gs(a, b, c, d)
                    dval = abs(integer_determinant(M.entries))
                    if dval > best_det:
                        best_det = dval
                        best_idx = (ai, bi, ci, di)
    return best_det, best_idx


def enumerate_gs_quads(m: int = 5, n_workers: int | None = None) -> tuple[int, tuple[int, int, int, int]]:
    """Exhaustively enum all 4-quad generators of length m, default border.

    For m=5: 2^20 ≈ 1M Bareiss-on-21 calls = ~5 min single-threaded.
    """
    n_workers = n_workers or max(1, mp.cpu_count() // 2)
    space = 1 << m
    chunk_size = max(1, space // n_workers)
    chunks = []
    for a_lo in range(0, space, chunk_size):
        chunks.append((a_lo, min(a_lo + chunk_size, space), m))
    with mp.Pool(n_workers) as pool:
        results = pool.map(_enum_chunk_gs, chunks)
    best_det = 0
    best_idx = (0, 0, 0, 0)
    for d, idx in results:
        if d > best_det:
            best_det = d
            best_idx = idx
    return best_det, best_idx


# ---------------------------------------------------------------------------
# SA over generator bits + full border
# ---------------------------------------------------------------------------


def sa_bordered_gs(
    *,
    m: int = 5,
    iterations: int = 30_000,
    t_start: float = 1.0,
    t_end: float = 0.005,
    seed: int = 0,
    full_border: bool = True,
    initial_quad: tuple[list[int], list[int], list[int], list[int]] | None = None,
) -> tuple[int, dict, Matrix]:
    """SA over the 4m generator + (8m+1) border bits of bordered_gs.

    Total free bits = 4m + (8m+1) = 12m + 1 if full_border, else 4m.
    For m=5 → 61 bits with full border, 20 without.
    """
    rng = random.Random(seed)
    inner_n = 4 * m
    n_order = inner_n + 1

    if initial_quad is None:
        a = [1 if rng.random() < 0.5 else -1 for _ in range(m)]
        b = [1 if rng.random() < 0.5 else -1 for _ in range(m)]
        c = [1 if rng.random() < 0.5 else -1 for _ in range(m)]
        d = [1 if rng.random() < 0.5 else -1 for _ in range(m)]
    else:
        a, b, c, d = (list(x) for x in initial_quad)

    if full_border:
        br = [1 if rng.random() < 0.5 else -1 for _ in range(inner_n)]
        bc = [1 if rng.random() < 0.5 else -1 for _ in range(inner_n)]
        corner = 1 if rng.random() < 0.5 else -1
    else:
        br = [1] * inner_n
        bc = [1] * inner_n
        corner = 1

    def build() -> Matrix:
        return bordered_gs(a, b, c, d, br, bc, corner)

    cur_M = build()
    cur_det = abs(integer_determinant(cur_M.entries))
    tries = 0
    while cur_det == 0 and tries < 16:
        a = [1 if rng.random() < 0.5 else -1 for _ in range(m)]
        b = [1 if rng.random() < 0.5 else -1 for _ in range(m)]
        c = [1 if rng.random() < 0.5 else -1 for _ in range(m)]
        d = [1 if rng.random() < 0.5 else -1 for _ in range(m)]
        cur_M = build()
        cur_det = abs(integer_determinant(cur_M.entries))
        tries += 1
    cur_e = -math.log(cur_det) if cur_det > 0 else math.inf

    best_det = cur_det
    best_M = cur_M

    n_bits = 4 * m + (2 * inner_n + 1 if full_border else 0)

    accepted = 0
    log_ratio = math.log(t_end / t_start)
    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        k = rng.randrange(n_bits)
        if k < m:
            a[k] = -a[k]; reset = lambda: a.__setitem__(k, -a[k])
        elif k < 2 * m:
            i2 = k - m; b[i2] = -b[i2]; reset = lambda: b.__setitem__(i2, -b[i2])
        elif k < 3 * m:
            i2 = k - 2 * m; c[i2] = -c[i2]; reset = lambda: c.__setitem__(i2, -c[i2])
        elif k < 4 * m:
            i2 = k - 3 * m; d[i2] = -d[i2]; reset = lambda: d.__setitem__(i2, -d[i2])
        elif k < 4 * m + inner_n:
            i2 = k - 4 * m; br[i2] = -br[i2]; reset = lambda: br.__setitem__(i2, -br[i2])
        elif k < 4 * m + 2 * inner_n:
            i2 = k - 4 * m - inner_n; bc[i2] = -bc[i2]; reset = lambda: bc.__setitem__(i2, -bc[i2])
        else:
            corner = -corner; reset = None

        prop_M = build()
        prop_det = abs(integer_determinant(prop_M.entries))
        if prop_det == 0:
            if reset is not None:
                reset()
            else:
                corner = -corner
            continue
        prop_e = -math.log(prop_det)
        delta = prop_e - cur_e
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            cur_e = prop_e
            cur_det = prop_det
            cur_M = prop_M
            accepted += 1
            if cur_det > best_det:
                best_det = cur_det
                best_M = cur_M
        else:
            if reset is not None:
                reset()
            else:
                corner = -corner

    telemetry = {"iterations": iterations, "accepted": accepted}
    return best_det, telemetry, best_M


__all__ = [
    "gs_array",
    "bordered_gs",
    "enumerate_gs_quads",
    "sa_bordered_gs",
]
