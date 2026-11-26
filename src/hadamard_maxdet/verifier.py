"""Verifier for {-1, +1} matrices in the Hadamard maximal determinant problem.

Given a matrix M of size n × n with entries in {-1, +1}, this module:

  1. Validates shape and entries.
  2. Computes det(M) exactly in integer arithmetic (Bareiss algorithm).
  3. Returns a Verdict with the signed determinant and |det|.

Integer arithmetic is essential. For n ≈ 20 a ±1 matrix can have
|det| ≈ 10^16; floating-point evaluation loses enough precision to push
the computed determinant off its true integer value, which corrupts any
search that uses det as the score.

The Bareiss algorithm does Gaussian elimination in the integers with
provably exact intermediate values (the entries after step k are the
(k+1)×(k+1) minors of the original matrix, hence integers).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    """Result of verifying a {-1, +1} matrix."""

    valid: bool
    n: int
    det: int
    abs_det: int
    error: str | None = None

    def __bool__(self) -> bool:
        return self.valid


def verify(entries: Sequence[Sequence[int]]) -> Verdict:
    """Verify a {-1, +1} matrix and compute its determinant exactly.

    Parameters
    ----------
    entries
        Square matrix of values in {-1, +1}, given as an iterable of rows.

    Returns
    -------
    Verdict
        ``valid=True`` with signed integer ``det`` and ``abs_det`` when the
        input is a well-formed ±1 matrix, else ``valid=False`` with an
        ``error`` message describing the first failure.
    """
    rows = [list(r) for r in entries]
    n = len(rows)
    if n == 0:
        return Verdict(False, 0, 0, 0, "matrix is empty")

    for i, r in enumerate(rows):
        if len(r) != n:
            return Verdict(
                False, n, 0, 0,
                f"row {i} has length {len(r)}, expected {n}",
            )
        for j, v in enumerate(r):
            if v not in (-1, 1):
                return Verdict(
                    False, n, 0, 0,
                    f"entry ({i}, {j}) is {v!r}; only -1 and +1 are allowed",
                )

    det = integer_determinant(rows)
    return Verdict(True, n, det, abs(det))


def _to_python_int_rows(entries: Sequence[Sequence[int]]) -> list[list[int]]:
    """Coerce ``entries`` into a list of lists of native Python ``int``.

    This guard exists because numpy ``ndarray`` inputs (and any sequence whose
    scalars are fixed-width integers, e.g. ``numpy.int64``) silently overflow
    during Bareiss elimination once intermediate products exceed ``2**63``.
    The overflow is real and observable: at order ``n = 23`` for ±1 matrices,
    the inner-loop product ``M[i][j] * pivot - M[i][k] * M[k][j]`` exceeds
    ``2**63`` and numpy wraps without raising, so ``integer_determinant``
    returns a wrong integer (typically off by a non-trivial factor).

    Casting every scalar through Python's ``int`` (arbitrary precision)
    eliminates the overflow at the cost of a one-time O(n^2) coercion.
    Cheap relative to the O(n^3) elimination that follows.

    Discovered 2026-04-25; see ``docs/findings.md`` "verifier numpy int64
    overflow" for the audit and regression test.
    """
    return [[int(v) for v in row] for row in entries]


def integer_determinant(entries: Sequence[Sequence[int]]) -> int:
    """Compute det(M) exactly in integer arithmetic via Bareiss elimination.

    Parameters
    ----------
    entries
        Square matrix of integers. Accepts any nested sequence (list of
        lists, tuple of tuples, numpy ``ndarray``, ...). Inputs whose
        scalars are fixed-width integers (e.g. ``numpy.int64``) are coerced
        to native Python ``int`` before elimination to prevent silent
        overflow on intermediate products that exceed ``2**63``.

    Returns
    -------
    int
        The determinant (signed, exact).

    Notes
    -----
    Bareiss (1968) adapts Gaussian elimination so that every intermediate
    entry is a (k+1)×(k+1) minor of the original matrix — hence an integer
    — and uses exact division by the previous pivot. No floating-point
    arithmetic is involved; no precision is lost.

    The only numerical hazard is fixed-width-integer overflow in the
    intermediate products. We avoid it by coercing every scalar to a
    Python ``int`` (arbitrary precision) at the start; see
    ``_to_python_int_rows`` for the bug history.
    """
    n = len(entries)
    if n == 0:
        return 1  # empty determinant convention

    # Always coerce to Python int. Cheap relative to the O(n^3) elimination,
    # and prevents the numpy int64 overflow bug at n >= 23 for ±1 matrices.
    M = _to_python_int_rows(entries)
    prev_pivot = 1
    sign = 1

    for k in range(n - 1):
        if M[k][k] == 0:
            swap = None
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    swap = i
                    break
            if swap is None:
                return 0
            M[k], M[swap] = M[swap], M[k]
            sign = -sign

        pivot = M[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * pivot - M[i][k] * M[k][j]) // prev_pivot
            M[i][k] = 0
        prev_pivot = pivot

    return sign * M[n - 1][n - 1]


def upper_bound(n: int) -> int:
    """Return the best *general* known upper bound on |det M| for order n.

    Uses the class-dependent closed-form bound:
      - n ≡ 0 (mod 4): Hadamard's bound n^(n/2).
      - n ≡ 1 (mod 4): Barba's bound (2n-1)^(1/2) · (n-1)^((n-1)/2).
      - n ≡ 2 (mod 4): Ehlich–Wojtas bound (2n-2) · (n-2)^((n-2)/2).
      - n ≡ 3 (mod 4): Barba's bound (2n-1)^(1/2) · (n-1)^((n-1)/2) (Ehlich 1964b gives a sharper bound for n ≡ 3 (mod 4)).

    Returns the floor of the (possibly irrational) bound. These bounds are
    tight only under side conditions (see module docstrings); for general n
    the true maximum can be strictly smaller.

    Ref: Orrick & Solomon, *The Hadamard maximal determinant problem*.
    """
    import math

    if n < 1:
        raise ValueError(f"order must be positive, got {n}")

    # All branches use exact integer arithmetic — math.sqrt would lose precision
    # against large n^k integers (≥ ~2^53), silently shifting the bound by ~10^37.
    r = n % 4
    if r == 0:
        # Hadamard: n^(n/2). Exact integer since n/2 is integral.
        return math.isqrt(n ** n)
    if r == 1:
        # Barba (1933): floor(sqrt((2n-1) * (n-1)^(n-1)))
        return math.isqrt((2 * n - 1) * (n - 1) ** (n - 1))
    if r == 2:
        # Ehlich (1964a) / Wojtas (1964): (2n-2) * (n-2)^((n-2)/2).
        return (2 * n - 2) * (n - 2) ** ((n - 2) // 2)
    # r == 3: Barba (1933): floor(sqrt((2n-1) * (n-1)^(n-1)))
    return math.isqrt((2 * n - 1) * (n - 1) ** (n - 1))
