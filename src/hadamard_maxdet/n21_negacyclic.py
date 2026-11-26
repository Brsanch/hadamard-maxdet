"""Negacyclic ansätze for n=21.

A negacirculant (or "skew-circulant") m×m matrix from generator a has
entries N[i][j] = a[(j - i) mod m] with a sign flip when (j - i) wraps:

    N[i][j] = a[(j - i) mod m] · (-1)^{[(j - i) < 0]}

Equivalently, shifting a row to the right negates entries that fall off
the right side. This is distinct from the "skew_circulant" in
structured.py (which is an ordinary circulant satisfying A + A^T = 2I).

Negacirculants are the matrix realisation of the cyclic group algebra
F_2[Z/n]/(x^n + 1) instead of (x^n - 1). They have a different
orthogonality structure that has been used for record ±1 determinants
at certain n in the literature (e.g. Fletcher–Gysin–Seberry 2001).

We supply:
  * `negacirculant(generator)` — m × m negacirculant.
  * `bordered_negacyclic_two(a, b, ...)` — bordered, inner block is
    [[A, B], [B^T, -A^T]] with A, B negacirculant of order m. Order
    n = 2m + 1 = 21 ⇒ m = 10.
  * `sa_bordered_negacyclic_two(...)` — SA over generator + border bits.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


def negacirculant(generator: Sequence[int]) -> list[list[int]]:
    """Return the m × m negacirculant from generator ``a``.

    N[i][j] = a[(j - i) mod m] · sign, where sign = -1 if (j - i) mod m
    came from a negative wrap (i.e., j < i and we wrap), else +1.
    Equivalently: N[i][j] = a[(j - i) mod m] · (-1)^{floor((j - i)/m)}.
    """
    m = len(generator)
    g = list(generator)
    out = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            diff = j - i
            wraps = (diff % m) - diff  # 0 if no wrap, m·k if wrapped k times
            sign = -1 if (wraps // m) % 2 == 1 else 1
            out[i][j] = g[diff % m] * sign
    return out


def two_negacirculant(a: Sequence[int], b: Sequence[int]) -> Matrix:
    """Two-block-negacirculant of order ``2m`` from generators ``a, b``.

    M = [[ A,   B  ],
         [ B^T, -A^T ]]
    with A, B m × m negacirculant. (Same block sign convention as
    structured.two_circulant; the negacyclic version is conjectured
    to access a different basin.)
    """
    if len(a) != len(b):
        raise ValueError("a, b must have equal length m")
    m = len(a)
    n = 2 * m
    A = negacirculant(a)
    B = negacirculant(b)
    AT = [[A[j][i] for j in range(m)] for i in range(m)]
    BT = [[B[j][i] for j in range(m)] for i in range(m)]
    nAT = [[-AT[i][j] for j in range(m)] for i in range(m)]
    entries = [[0] * n for _ in range(n)]
    for i in range(m):
        for j in range(m):
            entries[i][j] = A[i][j]
            entries[i][m + j] = B[i][j]
            entries[m + i][j] = BT[i][j]
            entries[m + i][m + j] = nAT[i][j]
    # Validate ±1.
    for r in entries:
        for v in r:
            if v not in (-1, 1):
                raise ValueError(f"two_negacirculant: non-±1 entry {v}")
    return Matrix(n=n, entries=entries)


def bordered_negacyclic_two(
    a: Sequence[int],
    b: Sequence[int],
    border_row: Sequence[int] | None = None,
    border_col: Sequence[int] | None = None,
    corner: int = 1,
) -> Matrix:
    """Bordered two-negacirculant of odd order n = 2m + 1."""
    inner = two_negacirculant(a, b).entries
    m = len(a)
    inner_n = 2 * m
    n = inner_n + 1
    if border_row is None:
        border_row = [1] * inner_n
    if border_col is None:
        border_col = [1] * inner_n
    if len(border_row) != inner_n or len(border_col) != inner_n:
        raise ValueError("border_row/col must have length 2m")
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


def sa_bordered_negacyclic_two(
    *,
    n: int,
    iterations: int = 30_000,
    t_start: float = 1.0,
    t_end: float = 0.005,
    seed: int = 0,
    full_border: bool = True,
    initial: tuple[list[int], list[int]] | None = None,
) -> tuple[int, dict, Matrix]:
    """SA over generator + (optional) border bits of bordered_negacyclic_two."""
    if n % 2 == 0:
        raise ValueError(f"requires odd n; got {n}")
    m = (n - 1) // 2
    inner_n = 2 * m
    rng = random.Random(seed)

    if initial is None:
        a = [1 if rng.random() < 0.5 else -1 for _ in range(m)]
        b = [1 if rng.random() < 0.5 else -1 for _ in range(m)]
    else:
        a, b = (list(x) for x in initial)

    if full_border:
        br = [1 if rng.random() < 0.5 else -1 for _ in range(inner_n)]
        bc = [1 if rng.random() < 0.5 else -1 for _ in range(inner_n)]
        corner = 1 if rng.random() < 0.5 else -1
    else:
        br = [1] * inner_n
        bc = [1] * inner_n
        corner = 1

    def build() -> Matrix:
        return bordered_negacyclic_two(a, b, br, bc, corner)

    cur_M = build()
    cur_det = abs(integer_determinant(cur_M.entries))
    tries = 0
    while cur_det == 0 and tries < 16:
        a = [1 if rng.random() < 0.5 else -1 for _ in range(m)]
        b = [1 if rng.random() < 0.5 else -1 for _ in range(m)]
        cur_M = build()
        cur_det = abs(integer_determinant(cur_M.entries))
        tries += 1
    cur_e = -math.log(cur_det) if cur_det > 0 else math.inf

    best_det = cur_det
    best_M = cur_M

    n_bits = 2 * m + (2 * inner_n + 1 if full_border else 0)

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
        elif k < 2 * m + inner_n:
            i2 = k - 2 * m; br[i2] = -br[i2]; reset = lambda: br.__setitem__(i2, -br[i2])
        elif k < 2 * m + 2 * inner_n:
            i2 = k - 2 * m - inner_n; bc[i2] = -bc[i2]; reset = lambda: bc.__setitem__(i2, -bc[i2])
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
    "negacirculant",
    "two_negacirculant",
    "bordered_negacyclic_two",
    "sa_bordered_negacyclic_two",
]
