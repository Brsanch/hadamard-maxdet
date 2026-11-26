"""Williamson construction of Hadamard matrices, specialised at m = 5.

A *Williamson quadruple* of order m is four ±1 m×m matrices ``(A, B, C, D)``,
each *symmetric* and *circulant*, satisfying

    A·Aᵀ + B·Bᵀ + C·Cᵀ + D·Dᵀ = 4m · I_m.

When such a quadruple exists, the **Williamson array**

    W = [[ A,  B,  C,  D],
         [-B,  A, -D,  C],
         [-C,  D,  A, -B],
         [-D, -C,  B,  A]]

is a Hadamard matrix of order ``4m``. For ``m = 5`` existence is classical
(Williamson 1944; Baumert–Hall tables); the search space is tiny because a
length-5 *symmetric* circulant has only 3 free bits (positions 0, 1, 2 with
``a[3] = a[2]`` and ``a[4] = a[1]``), so all four-tuples can be enumerated in
``8^4 = 4096`` candidates in microseconds.

This module exposes:

* :func:`symmetric_circulant_first_row` — assemble a length-5 symmetric
  circulant generator from 3 free bits (a₀, a₁, a₂).
* :func:`circulant_matrix` — build the m×m circulant matrix from a generator.
* :func:`verify_williamson_constraint` — exact-integer check
  ``A·Aᵀ + B·Bᵀ + C·Cᵀ + D·Dᵀ == 4m · I``.
* :func:`enumerate_symmetric_circulant_generators` — enumerate the 8 length-5
  symmetric circulant generators.
* :func:`williamson_quadruples_m5` — return all length-5 Williamson
  quadruples (exhaustive search; verified by the constraint).
* :func:`williamson_array` — build the 4m × 4m Williamson Hadamard matrix
  from a quadruple.
* :func:`bordered_williamson_h21` — extend H₂₀ to an order-21 candidate by
  adding one border row + column.
* :func:`williamson_h20_deletion_h19` — return all 400 H₁₉ candidates from
  single-row × single-column deletions of an H₂₀ Williamson matrix.
* :func:`gs_array_williamson_h23` — order-23 candidate built by a
  Goethals–Seidel-like array seeded by the Williamson quadruple plus a
  length-3 base sequence (multiple variants; we expose the simplest two).

All matrix entries are ±1. All determinants reported by callers must be
exact-Bareiss verified via :mod:`hadamard_maxdet.verifier`.
"""

from __future__ import annotations

from collections.abc import Sequence

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Length-5 symmetric circulant generators
# ---------------------------------------------------------------------------


def symmetric_circulant_first_row(a0: int, a1: int, a2: int) -> list[int]:
    """Return a length-5 generator (a0, a1, a2, a2, a1) for a symmetric circulant.

    A circulant ``C`` with first row ``g`` (i.e. ``C[i][j] = g[(j-i) mod m]``) is
    symmetric iff ``g[k] = g[(-k) mod m]``. For ``m = 5`` this fixes
    ``g[3] = g[2]`` and ``g[4] = g[1]``, leaving 3 free entries.
    """
    for v in (a0, a1, a2):
        if v not in (-1, 1):
            raise ValueError(f"entries must be ±1; got {v!r}")
    return [a0, a1, a2, a2, a1]


def enumerate_symmetric_circulant_generators(m: int = 5) -> list[list[int]]:
    """Enumerate all symmetric length-m circulant generators (±1).

    Currently specialised to ``m = 5`` where there are exactly ``2^3 = 8``
    generators. For other m the function raises ``NotImplementedError``.
    """
    if m != 5:
        raise NotImplementedError(
            f"enumerate_symmetric_circulant_generators only supports m=5; got {m}"
        )
    out: list[list[int]] = []
    for k in range(8):
        a0 = 1 if (k >> 0) & 1 else -1
        a1 = 1 if (k >> 1) & 1 else -1
        a2 = 1 if (k >> 2) & 1 else -1
        out.append(symmetric_circulant_first_row(a0, a1, a2))
    return out


# ---------------------------------------------------------------------------
# Circulant constructor + symmetry / Williamson constraint
# ---------------------------------------------------------------------------


def circulant_matrix(generator: Sequence[int]) -> list[list[int]]:
    """Return the m×m circulant ``C[i][j] = g[(j - i) mod m]``."""
    m = len(generator)
    g = list(generator)
    return [[g[(j - i) % m] for j in range(m)] for i in range(m)]


def _matmul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    n = len(A)
    p = len(B[0])
    k = len(B)
    out = [[0] * p for _ in range(n)]
    for i in range(n):
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
    n = len(A)
    return [[A[j][i] for j in range(n)] for i in range(n)]


def is_symmetric(A: list[list[int]]) -> bool:
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] != A[j][i]:
                return False
    return True


def verify_williamson_constraint(
    A: list[list[int]],
    B: list[list[int]],
    C: list[list[int]],
    D: list[list[int]],
) -> bool:
    """Return True iff ``A·Aᵀ + B·Bᵀ + C·Cᵀ + D·Dᵀ == 4m · I_m`` exactly.

    Each block must be a square m × m ±1 matrix. The check is exact-integer.
    """
    m = len(A)
    if not (len(B) == len(C) == len(D) == m):
        return False
    for X in (A, B, C, D):
        if len(X) != m or any(len(r) != m for r in X):
            return False
        for r in X:
            for v in r:
                if v not in (-1, 1):
                    return False
    AAt = _matmul(A, _transpose(A))
    BBt = _matmul(B, _transpose(B))
    CCt = _matmul(C, _transpose(C))
    DDt = _matmul(D, _transpose(D))
    target_diag = 4 * m
    for i in range(m):
        for j in range(m):
            s = AAt[i][j] + BBt[i][j] + CCt[i][j] + DDt[i][j]
            if i == j:
                if s != target_diag:
                    return False
            else:
                if s != 0:
                    return False
    return True


# ---------------------------------------------------------------------------
# Enumerate Williamson quadruples at m = 5
# ---------------------------------------------------------------------------


def williamson_quadruples_m5() -> list[
    tuple[list[int], list[int], list[int], list[int]]
]:
    """Return every length-5 Williamson generator quadruple ``(a, b, c, d)``.

    Each entry is the *first row* of the corresponding symmetric circulant
    block. The full block matrices are recovered via :func:`circulant_matrix`.

    Search is exhaustive over the ``8^4 = 4096`` symmetric-circulant
    combinations; verification uses :func:`verify_williamson_constraint`
    exactly (no floating point). Returns the list in lexicographic order on
    the generator tuples.
    """
    gens = enumerate_symmetric_circulant_generators(5)
    quads: list[tuple[list[int], list[int], list[int], list[int]]] = []
    for ga in gens:
        A = circulant_matrix(ga)
        for gb in gens:
            B = circulant_matrix(gb)
            for gc in gens:
                C = circulant_matrix(gc)
                for gd in gens:
                    D = circulant_matrix(gd)
                    if verify_williamson_constraint(A, B, C, D):
                        quads.append((list(ga), list(gb), list(gc), list(gd)))
    return quads


# ---------------------------------------------------------------------------
# Williamson array → H_{4m}
# ---------------------------------------------------------------------------


def williamson_array(
    A: list[list[int]],
    B: list[list[int]],
    C: list[list[int]],
    D: list[list[int]],
) -> Matrix:
    """Build the 4m × 4m Williamson matrix from blocks (A, B, C, D).

    Layout::

        W = [[ A,  B,  C,  D],
             [-B,  A, -D,  C],
             [-C,  D,  A, -B],
             [-D, -C,  B,  A]]

    No verification is performed here — call :func:`verify_williamson_constraint`
    upstream. Returns a :class:`Matrix` with entries in {-1, +1} regardless of
    whether the constraint holds.
    """
    m = len(A)
    if not (len(B) == len(C) == len(D) == m):
        raise ValueError("A, B, C, D must all be m × m with the same m")
    n = 4 * m

    def neg(X: list[list[int]]) -> list[list[int]]:
        return [[-v for v in row] for row in X]

    nB, nC, nD = neg(B), neg(C), neg(D)
    blocks = [
        [A,  B,  C,  D],
        [nB, A,  nD, C],
        [nC, D,  A,  nB],
        [nD, nC, B,  A],
    ]
    entries = [[0] * n for _ in range(n)]
    for bi in range(4):
        for bj in range(4):
            blk = blocks[bi][bj]
            for i in range(m):
                for j in range(m):
                    entries[bi * m + i][bj * m + j] = blk[i][j]
    return Matrix(n=n, entries=entries)


def williamson_h20_from_generators(
    a: Sequence[int],
    b: Sequence[int],
    c: Sequence[int],
    d: Sequence[int],
) -> Matrix:
    """Convenience: build H_20 directly from four length-5 generators."""
    return williamson_array(
        circulant_matrix(a),
        circulant_matrix(b),
        circulant_matrix(c),
        circulant_matrix(d),
    )


# ---------------------------------------------------------------------------
# Borderings to order 21
# ---------------------------------------------------------------------------


# A small zoo of border patterns. Each value is a callable(n_inner) -> (
#   border_row, border_col, corner) where border_row/col have length n_inner
# and corner is ±1.
def _all_plus(n_inner: int) -> tuple[list[int], list[int], int]:
    return [1] * n_inner, [1] * n_inner, 1


def _all_minus_with_plus_corner(n_inner: int) -> tuple[list[int], list[int], int]:
    return [-1] * n_inner, [-1] * n_inner, 1


def _alt_signs(n_inner: int) -> tuple[list[int], list[int], int]:
    row = [1 if k % 2 == 0 else -1 for k in range(n_inner)]
    col = [1 if k % 2 == 0 else -1 for k in range(n_inner)]
    return row, col, 1


def _half_split(n_inner: int) -> tuple[list[int], list[int], int]:
    half = n_inner // 2
    row = [1] * half + [-1] * (n_inner - half)
    col = [1] * half + [-1] * (n_inner - half)
    return row, col, 1


def _block_signs(n_inner: int) -> tuple[list[int], list[int], int]:
    # Sign pattern that respects the four-block layout of W: +,+,-,-.
    m = n_inner // 4
    row = [1] * m + [1] * m + [-1] * m + [-1] * m
    col = [1] * m + [1] * m + [-1] * m + [-1] * m
    return row, col, 1


BORDER_PATTERNS: dict[str, callable] = {
    "all_plus": _all_plus,
    "all_minus_plus_corner": _all_minus_with_plus_corner,
    "alt_signs": _alt_signs,
    "half_split": _half_split,
    "block_signs": _block_signs,
}


def bordered_williamson_h21(
    A: list[list[int]],
    B: list[list[int]],
    C: list[list[int]],
    D: list[list[int]],
    border_pattern: str = "all_plus",
    border_row: Sequence[int] | None = None,
    border_col: Sequence[int] | None = None,
    corner: int | None = None,
) -> Matrix:
    """Return an order 21 ±1 matrix obtained by bordering the Williamson H_20.

    Either pass an explicit ``border_row``/``border_col``/``corner`` (each entry
    ±1; row and col of length 20) or use one of the named patterns in
    :data:`BORDER_PATTERNS`.

    No claim of optimality is made; callers should verify with Bareiss.
    """
    inner = williamson_array(A, B, C, D).entries
    n_inner = len(inner)
    n = n_inner + 1

    if border_row is None or border_col is None or corner is None:
        if border_pattern not in BORDER_PATTERNS:
            raise ValueError(
                f"unknown border_pattern {border_pattern!r}; "
                f"known: {sorted(BORDER_PATTERNS)}"
            )
        br, bc, cn = BORDER_PATTERNS[border_pattern](n_inner)
        if border_row is None:
            border_row = br
        if border_col is None:
            border_col = bc
        if corner is None:
            corner = cn

    if len(border_row) != n_inner or len(border_col) != n_inner:
        raise ValueError(f"border_row/col must have length {n_inner}")
    for v in list(border_row) + list(border_col) + [corner]:
        if v not in (-1, 1):
            raise ValueError(f"border entries must be ±1; got {v!r}")

    entries = [[0] * n for _ in range(n)]
    entries[0][0] = corner
    for j in range(n_inner):
        entries[0][j + 1] = border_row[j]
    for i in range(n_inner):
        entries[i + 1][0] = border_col[i]
        for j in range(n_inner):
            entries[i + 1][j + 1] = inner[i][j]
    return Matrix(n=n, entries=entries)


# ---------------------------------------------------------------------------
# Row + column deletions: H_20 → 400 H_19 candidates
# ---------------------------------------------------------------------------


def williamson_h20_deletion_h19(
    A: list[list[int]],
    B: list[list[int]],
    C: list[list[int]],
    D: list[list[int]],
) -> list[tuple[tuple[int, int], Matrix]]:
    """Return all 20 × 20 = 400 single row + column deletions of W(A,B,C,D).

    Each entry is ``((i_del, j_del), Matrix(n=19))``. Rows/cols are 0-indexed.
    No orbit reduction is applied — the four block-symmetry orbits typically
    produce ~25 distinct ratios after Hadamard equivalence, but we leave
    that to downstream tooling.
    """
    inner = williamson_array(A, B, C, D).entries
    n_inner = len(inner)
    out: list[tuple[tuple[int, int], Matrix]] = []
    for i_del in range(n_inner):
        for j_del in range(n_inner):
            sub = [
                [inner[i][j] for j in range(n_inner) if j != j_del]
                for i in range(n_inner)
                if i != i_del
            ]
            out.append(((i_del, j_del), Matrix(n=n_inner - 1, entries=sub)))
    return out


# ---------------------------------------------------------------------------
# Goethals–Seidel-style order-23 from Williamson seeds
# ---------------------------------------------------------------------------


def _back_perm(m: int) -> list[list[int]]:
    return [[1 if (i + j) == (m - 1) else 0 for j in range(m)] for i in range(m)]


def _gs_block_layout(
    A: list[list[int]],
    B: list[list[int]],
    C: list[list[int]],
    D: list[list[int]],
) -> Matrix:
    """Goethals–Seidel array with R = back-perm. Order 4m.

    Identical layout to :func:`hadamard_maxdet.n21_gs_array.gs_array` but
    accepts pre-built blocks (not generators). Kept local to avoid a hard
    cycle through that module.
    """
    m = len(A)
    R = _back_perm(m)
    BR = _matmul(B, R)
    CR = _matmul(C, R)
    DR = _matmul(D, R)
    BTR = _matmul(_transpose(B), R)
    CTR = _matmul(_transpose(C), R)
    DTR = _matmul(_transpose(D), R)

    def neg(X: list[list[int]]) -> list[list[int]]:
        return [[-v for v in row] for row in X]

    n = 4 * m
    entries = [[0] * n for _ in range(n)]
    blocks = [
        [A,        BR,       CR,       DR],
        [neg(BR),  A,        DTR,      neg(CTR)],
        [neg(CR),  neg(DTR), A,        BTR],
        [neg(DR),  CTR,      neg(BTR), A],
    ]
    for bi in range(4):
        for bj in range(4):
            blk = blocks[bi][bj]
            for i in range(m):
                for j in range(m):
                    entries[bi * m + i][bj * m + j] = blk[i][j]
    return Matrix(n=n, entries=entries)


def gs_array_williamson_h23(
    A: list[list[int]],
    B: list[list[int]],
    C: list[list[int]],
    D: list[list[int]],
    e: Sequence[int] | None = None,
    *,
    variant: str = "border3",
) -> Matrix:
    """Build a 23 × 23 ±1 matrix using a GS-like array seeded by Williamson
    blocks plus a length-3 border ``e``.

    The cleanest way to land at n = 23 is to take the order-20 Williamson H_20
    (via the *standard* Williamson layout, which differs from GS only in the
    placement of transposes), border by 3 extra rows + 3 extra columns, and
    fill the 3 × 3 corner block from ``e``. Two variants are provided:

    ``"border3"``  — H_20 in the top-left corner; rightmost 3 columns and
                     bottom 3 rows filled with ±1 from ``e`` plus a constant
                     ``+1`` background.

    ``"border3_gs"`` — same border layout but with GS-style block placement
                     of A, B, C, D (matches :mod:`n21_gs_array.gs_array`).

    Both variants accept ``e`` of length 3 (used to set the diagonal of the
    3 × 3 corner). When ``e is None`` the all-+1 corner is used.

    No claim of optimality. Callers should Bareiss-verify.
    """
    if variant == "border3":
        inner = williamson_array(A, B, C, D)
    elif variant == "border3_gs":
        inner = _gs_block_layout(A, B, C, D)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    n_inner = inner.n
    if n_inner != 20:
        raise ValueError(f"expected inner order 20; got {n_inner}")
    if e is None:
        e = [1, 1, 1]
    if len(e) != 3:
        raise ValueError("e must have length 3")
    for v in e:
        if v not in (-1, 1):
            raise ValueError(f"e entries must be ±1; got {v!r}")

    n = 23
    entries = [[1] * n for _ in range(n)]
    for i in range(n_inner):
        for j in range(n_inner):
            entries[i][j] = inner.entries[i][j]
    # Bottom 3 rows / right 3 cols default to +1 above; set the 3×3 corner
    # diagonal from ``e``.
    for k in range(3):
        entries[n_inner + k][n_inner + k] = e[k]
    return Matrix(n=n, entries=entries)


__all__ = [
    "BORDER_PATTERNS",
    "bordered_williamson_h21",
    "circulant_matrix",
    "enumerate_symmetric_circulant_generators",
    "gs_array_williamson_h23",
    "is_symmetric",
    "symmetric_circulant_first_row",
    "verify_williamson_constraint",
    "williamson_array",
    "williamson_h20_deletion_h19",
    "williamson_h20_from_generators",
    "williamson_quadruples_m5",
]
