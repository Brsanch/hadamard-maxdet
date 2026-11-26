"""Koukouvinos–Mitrouli–Seberry-style direct construction for n ≡ 3 (mod 4).

This module implements **Proposition 28** from the survey

    Browne, Egan, Hegarty, Ó Catháin (2021),
    "Bounds on the maximal determinant problem for ±1 matrices,"
    arXiv:2104.06756, §6.1, p. 21.

The proposition gives an algebraic ±1 matrix M of order n = 4q + 3 from a
Paley quadratic-residue (q × q) core Q, the order-4 Hadamard matrix
``H4 = J4 - 2·I4`` (the unique ±1 Hadamard with constant row-sum 2), and a
fixed 4 × 3 border block

    B = [[ 1,  1,  1],
         [ 1, -1, -1],
         [-1,  1, -1],
         [-1, -1,  1]].

The construction is

    P = Q ⊗ H4 - I_q ⊗ J_4         # (4q × 4q), entries in {-1, +1}
    R = j_q ⊗ B                    # (4q × 3),  entries in {-1, +1}
    M = [[P,    R  ],
         [R^T,  J_3]]              # ((4q+3) × (4q+3))

The proposition asserts

    det(M M^T) = 16 · (4q)^(3q + 3) · (4q + 16)^(q − 1)

so that |det M| = 4 · (4q)^((3q+3)/2) · (4q + 16)^((q-1)/2).

That formula is **valid only when the Paley core Q is skew-symmetric**, i.e.
when q ≡ 3 (mod 4) (or, more generally, when there exists a skew-Hadamard
matrix of order q + 1). At q ≡ 1 (mod 4) the Paley core is symmetric, the
proof's cancellation step ``Q + Q^T = 0`` fails, and the construction still
yields a valid ±1 matrix but with a much smaller |det|.

For our target n = 23 (q = 5), q ≡ 1 (mod 4), so Prop 28 with the Paley core
does **not** match its own formula prediction at this order. See the
``prop28_matrix`` docstring and the test suite for the actual measured |det|
values. This module ships the construction and the formula in parallel; the
caller is expected to interpret ``prop28_det_predicted(q)`` as the formula
value (which agrees with the actual |det| only for q ≡ 3 mod 4), and
``integer_determinant(prop28_matrix(q).entries)`` as the ground truth.
"""

from __future__ import annotations

import math
from typing import Sequence

from hadamard_maxdet.matrix import Matrix


# ---------------------------------------------------------------------------
# Paley quadratic-residue core
# ---------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    r = 3
    while r * r <= n:
        if n % r == 0:
            return False
        r += 2
    return True


def paley_core_matrix(q: int) -> list[list[int]]:
    """Return the q × q Paley quadratic-residue core matrix.

    ``Q[i][j] = chi(i - j)`` over ``F_q``, where ``chi`` is the quadratic
    character: ``chi(0) = 0``, ``chi(x) = +1`` if ``x`` is a non-zero QR, else
    ``-1``. Entries are in ``{-1, 0, +1}`` with zero diagonal.

    Properties:

    - ``Q J = J Q = 0`` (row and column sums are zero).
    - ``Q Q^T = q I_q − J_q`` for any odd prime q.
    - ``Q + Q^T = 0`` (skew-symmetric) iff ``q ≡ 3 (mod 4)``.
    - ``Q − Q^T = 0`` (symmetric) iff ``q ≡ 1 (mod 4)``.

    Parameters
    ----------
    q
        An odd prime.
    """
    if q < 3 or not _is_prime(q):
        raise ValueError(f"paley_core requires an odd prime q ≥ 3; got q={q}")

    # Quadratic-residue indicator chi on F_q; chi(0)=0, chi(x)=±1.
    qr = {pow(x, 2, q) for x in range(1, q)}
    chi = [0] * q
    for x in range(1, q):
        chi[x] = 1 if x in qr else -1

    return [[chi[(i - j) % q] for j in range(q)] for i in range(q)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kron(A: Sequence[Sequence[int]], B: Sequence[Sequence[int]]) -> list[list[int]]:
    ra, ca = len(A), len(A[0])
    rb, cb = len(B), len(B[0])
    C = [[0] * (ca * cb) for _ in range(ra * rb)]
    for i in range(ra):
        for j in range(ca):
            aij = A[i][j]
            if aij == 0:
                continue
            for k in range(rb):
                for l in range(cb):
                    C[i * rb + k][j * cb + l] = aij * B[k][l]
    return C


def _h4_complement_of_identity() -> list[list[int]]:
    """Return ``J4 - 2·I4``: the order-4 Hadamard with constant row-sum 2.

    This is the H4 used by Proposition 28: it satisfies
    ``H4 J4 = J4 H4^T = 2 J4`` (symmetric mixed product), which is what the
    proof's cancellation argument requires. Sylvester's H4 has row-sum
    (4, 0, 0, 0) and does *not* satisfy the symmetry; substituting Sylvester
    into the construction produces a different (smaller) determinant.
    """
    return [
        [-1,  1,  1,  1],
        [ 1, -1,  1,  1],
        [ 1,  1, -1,  1],
        [ 1,  1,  1, -1],
    ]


def _border_block_B() -> list[list[int]]:
    """Return the 4 × 3 block ``B`` from Proposition 28.

    Each row of B sums to (3, -1, -1, -1) — i.e. row 0 sums to 3, every
    other row sums to -1 — so that ``B B^T = 4 I_4 - J_4``.
    """
    return [
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ]


# ---------------------------------------------------------------------------
# Proposition 28 construction
# ---------------------------------------------------------------------------


def prop28_matrix(q: int) -> Matrix:
    """Build the (4q + 3) × (4q + 3) ±1 matrix M from Proposition 28.

    Parameters
    ----------
    q
        Odd prime giving order n = 4q + 3.

    Returns
    -------
    Matrix
        The constructed ±1 matrix wrapped in :class:`hadamard_maxdet.Matrix`.

    Notes
    -----
    *q ≡ 3 (mod 4)*: the Paley core is skew, the proposition's proof goes
    through, and ``|det M|`` matches :func:`prop28_det_predicted` exactly.

    *q ≡ 1 (mod 4)*: the Paley core is symmetric; the construction still
    produces a valid ±1 matrix (correct dimensions, correct ±1 entries,
    correct Gram diagonal n), but ``|det M|`` is **substantially smaller**
    than the formula's prediction. In particular at q = 5 (n = 23), the
    construction yields ``|det| = 32_768_000_000_000`` (≈ 0.0084 · Barba),
    far below the in-repo record. This is the "documented discrepancy" with
    Table 1 of Browne et al., which lists 0.7060 in the Prop-28 column for
    n = 23 — that number must come from a generalisation we have not
    isolated, since order-6 has no skew-Hadamard matrix.
    """
    if q < 3 or not _is_prime(q):
        raise ValueError(f"Prop 28 requires an odd prime q ≥ 3; got q={q}")

    Q = paley_core_matrix(q)
    H4 = _h4_complement_of_identity()
    J4 = [[1] * 4 for _ in range(4)]
    Iq = [[1 if i == j else 0 for j in range(q)] for i in range(q)]
    B = _border_block_B()
    J3 = [[1] * 3 for _ in range(3)]

    QH = _kron(Q, H4)        # (4q × 4q)
    IqJ4 = _kron(Iq, J4)     # (4q × 4q)
    P = [
        [QH[i][j] - IqJ4[i][j] for j in range(4 * q)]
        for i in range(4 * q)
    ]
    # Sanity: every entry of P must be in {-1, +1}. (Q has 0 on the diagonal,
    # and the diagonal blocks of Q⊗H4 are zero, so P's diagonal blocks are
    # exactly -J4. Off-diagonal blocks are ±H4 - 0 = ±H4 with entries ±1.)
    for i, row in enumerate(P):
        for j, v in enumerate(row):
            if v not in (-1, 1):
                raise RuntimeError(
                    f"prop28: P[{i}][{j}] = {v} not in {{-1,+1}}; "
                    "this indicates a logic error in the construction."
                )

    jq = [[1] for _ in range(q)]
    R = _kron(jq, B)         # (4q × 3)
    for i, row in enumerate(R):
        for j, v in enumerate(row):
            if v not in (-1, 1):
                raise RuntimeError(
                    f"prop28: R[{i}][{j}] = {v} not in {{-1,+1}}"
                )

    n = 4 * q + 3
    M: list[list[int]] = [[0] * n for _ in range(n)]
    for i in range(4 * q):
        for j in range(4 * q):
            M[i][j] = P[i][j]
        for j in range(3):
            M[i][4 * q + j] = R[i][j]
    for i in range(3):
        for j in range(4 * q):
            M[4 * q + i][j] = R[j][i]
        for j in range(3):
            M[4 * q + i][4 * q + j] = J3[i][j]

    return Matrix(n=n, entries=M)


def prop28_det_predicted(q: int) -> int:
    """Return the |det M| value predicted by Proposition 28's formula.

    The formula is
        det(M M^T) = 16 · (4q)^(3q + 3) · (4q + 16)^(q − 1),
    so |det M| = sqrt(...). The right-hand side is always a perfect square
    in the integers when ``q`` makes ``(3q + 3)`` and ``(q − 1)`` both even
    (which happens for every odd q since ``3q + 3 = 3(q+1)`` and ``q - 1``
    are both even when q is odd). This function returns the integer square
    root.

    *Validity*: this matches ``|det(prop28_matrix(q))|`` only when
    ``q ≡ 3 (mod 4)`` (so the Paley core is skew). For ``q ≡ 1 (mod 4)``
    (e.g. q = 5 ⇒ n = 23) the *predicted* value is far above the *actual*
    determinant of the construction. See :func:`prop28_matrix` for details.
    """
    if q < 3 or not _is_prime(q):
        raise ValueError(f"prop28_det_predicted requires odd prime q ≥ 3; got q={q}")
    det_sq = 16 * (4 * q) ** (3 * q + 3) * (4 * q + 16) ** (q - 1)
    root = math.isqrt(det_sq)
    if root * root != det_sq:
        # Should be exact for the formula at any odd q.
        raise RuntimeError(
            f"prop28_det_predicted({q}): det_sq = {det_sq} is not a perfect square"
        )
    return root


def is_skew_paley(q: int) -> bool:
    """Return True iff the Paley core of order q is skew-symmetric.

    Equivalently, q ≡ 3 (mod 4). The proposition's formula matches the
    actual |det| of :func:`prop28_matrix` only in this regime.
    """
    if q < 3 or not _is_prime(q):
        return False
    return q % 4 == 3
