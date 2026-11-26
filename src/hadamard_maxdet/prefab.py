"""Prefab block-assembly construction for the Hadamard maximal-det problem.

For an order ``n`` we have not yet beaten, decompose ``n = a + b`` and build

    M = [[ A, B ],
         [ C, D ]]

where ``A`` is a proven-max ±1 matrix at order ``a``, ``D`` is a proven-max
±1 matrix at order ``b``, and ``B`` (a×b) and ``C`` (b×a) are the ±1 "glue"
blocks that we optimise.

Schur complement (with ``A`` invertible — it is, since |det A| > 0 for any
proven-max ±1 matrix):

    det(M) = det(A) · det( D − C · A^{-1} · B )

So we hold ``A`` and ``D`` fixed, and search over the ``a·b + b·a`` ±1
entries of ``B`` and ``C`` to maximise

    |det(M)| = |det(A)| · |det(S)|,    S = D − C A^{-1} B.

This module provides:

* :func:`load_proven_matrix(n)` — return a ±1 matrix at order ``n``
  saturating the proven max ``D(±1, n)`` (cache-backed).
* :func:`schur_det(A, B, C, D, A_inv_num, det_A)` — compute ``det(M)`` of
  the assembled block matrix exactly via the Schur formula above. Uses
  the *integer* representation ``adj(A) = det(A) · A^{-1}`` so that all
  intermediate quantities are exact integers and the final det is exact.
* :func:`prefab_sa(a, b, ...)` — multi-start simulated annealing over the
  glue blocks (B, C), scoring via :func:`schur_det`. Returns the best
  full block matrix and its exact |det|.

Performance note. The hot loop scores via Schur on the b×b complement and
then promotes the winner to a full Bareiss verification, so the cost per
step scales with ``b^3`` (small) rather than ``n^3``. For (a, b) = (17, 6)
the inner det is 6×6 — trivial — and the bulk of cost is the matrix
products ``A^{-1} B`` and ``C (A^{-1} B)``.
"""
from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from hadamard_maxdet.incremental import _compute_adjugate
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.orders import record_pm1
from hadamard_maxdet.verifier import integer_determinant
from hadamard_maxdet.winner_decode import find_winning_matrix


# ---------------------------------------------------------------------------
# Proven-max matrix loader (cache-backed)
# ---------------------------------------------------------------------------

# In-process cache of proven-max matrices keyed by order.
_PROVEN_CACHE: dict[int, list[list[int]]] = {}


def _try_structured(n: int) -> Optional[list[list[int]]]:
    """Cheap closed-form constructions for orders where one exists."""
    # Sylvester (n a power of 2): always Hadamard.
    if n >= 2 and (n & (n - 1)) == 0:
        return Matrix.sylvester(n).entries
    # Paley type I (n = q + 1, q ≡ 3 mod 4 prime): Hadamard.
    try:
        return Matrix.paley_type_i(n).entries
    except ValueError:
        pass
    return None


def load_proven_matrix(
    n: int,
    *,
    sa_budget_seconds: float = 20.0,
    base_seed: int = 12345,
) -> list[list[int]]:
    """Return a ±1 matrix of order ``n`` whose |det| equals the proven max.

    Uses (in priority order):
      1. In-process cache.
      2. Structured constructions (Sylvester, Paley I) when applicable.
      3. ``find_winning_matrix`` from :mod:`hadamard_maxdet.winner_decode`.

    Raises ``RuntimeError`` if no proven-max-saturating matrix is found
    within the budget. ``n`` must be one of the orders for which the
    proven max is tabulated (1 ≤ n ≤ 21).
    """
    if n in _PROVEN_CACHE:
        return [row[:] for row in _PROVEN_CACHE[n]]

    target = record_pm1(n)

    # 1) Try a structured construction; verify it hits the proven max.
    cand = _try_structured(n)
    if cand is not None and abs(integer_determinant(cand)) == target:
        _PROVEN_CACHE[n] = [row[:] for row in cand]
        return [row[:] for row in cand]

    # 2) Fall back to SA multi-start (with bordered-two-circulant for odd n).
    abs_det, ent, _src = find_winning_matrix(
        n,
        target=target,
        n_starts=8,
        iterations=80_000,
        structured_seconds=sa_budget_seconds,
        base_seed=base_seed,
    )
    if abs_det != target:
        raise RuntimeError(
            f"could not reach proven max {target} at order {n}; got {abs_det}"
        )
    _PROVEN_CACHE[n] = [row[:] for row in ent]
    return [row[:] for row in ent]


# ---------------------------------------------------------------------------
# Schur-complement determinant of a block matrix
# ---------------------------------------------------------------------------


def _assemble(
    A: Sequence[Sequence[int]],
    B: Sequence[Sequence[int]],
    C: Sequence[Sequence[int]],
    D: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Glue four blocks into a single (a+b)×(a+b) matrix."""
    a = len(A)
    b = len(D)
    if any(len(r) != a for r in A):
        raise ValueError("A not square a×a")
    if any(len(r) != b for r in D):
        raise ValueError("D not square b×b")
    if len(B) != a or any(len(r) != b for r in B):
        raise ValueError("B must be a×b")
    if len(C) != b or any(len(r) != a for r in C):
        raise ValueError("C must be b×a")
    M = []
    for i in range(a):
        M.append(list(A[i]) + list(B[i]))
    for i in range(b):
        M.append(list(C[i]) + list(D[i]))
    return M


def schur_det(
    A: Sequence[Sequence[int]],
    B: Sequence[Sequence[int]],
    C: Sequence[Sequence[int]],
    D: Sequence[Sequence[int]],
    *,
    adj_A: Optional[Sequence[Sequence[int]]] = None,
    det_A: Optional[int] = None,
) -> int:
    """Return ``det(M)`` exactly for the assembled block matrix M = [[A,B],[C,D]].

    Uses the Schur identity (valid because ``det A ≠ 0`` for our proven-max
    blocks):

        det(M) = det(A) · det( D − C · A^{-1} · B )

    All intermediate quantities are kept as exact integers by representing
    ``A^{-1}`` through the integer adjugate ``adj(A) = det(A) · A^{-1}``:

        det(M) = det(A) · det(D − C · adj(A) · B / det(A))
               = det(A)^{1 - b} · det( det(A) · D − C · adj(A) · B )

    where ``b = order(D)``. The right-hand side is an exact-integer
    expression; for our (a, b) sizes the dividend is small enough to live in
    Python's arbitrary-precision ints.

    Parameters
    ----------
    A, B, C, D
        Block sub-matrices. ``A`` must be a×a (invertible), ``D`` b×b,
        ``B`` a×b, ``C`` b×a. Entries integer (typically ±1).
    adj_A, det_A
        Optional cached integer adjugate and determinant of ``A``. When
        omitted they are recomputed via :func:`integer_determinant` and
        :func:`_compute_adjugate`.
    """
    a = len(A)
    b = len(D)
    if det_A is None:
        det_A = integer_determinant(A)
    if det_A == 0:
        # Schur path is invalid; assemble and call Bareiss directly.
        return integer_determinant(_assemble(A, B, C, D))
    if adj_A is None:
        adj_A = _compute_adjugate([list(r) for r in A], det_A)

    A_arr = np.array(A, dtype=object)
    B_arr = np.array(B, dtype=object)
    C_arr = np.array(C, dtype=object)
    D_arr = np.array(D, dtype=object)
    adj_arr = np.array(adj_A, dtype=object)

    # X = adj(A) · B  (a×b, integer entries).
    X = adj_arr @ B_arr
    # Y = C · X       (b×b, integer entries).
    Y = C_arr @ X
    # Z = det(A) · D − Y  (b×b, integer entries).
    Z = det_A * D_arr - Y
    Z_list = [[int(Z[i, j]) for j in range(b)] for i in range(b)]
    detZ = integer_determinant(Z_list)
    # det(M) = det(A)^{1-b} · detZ. Since b ≥ 1, the exponent is ≤ 0:
    # det(M) · det(A)^{b-1} = detZ, so det(M) = detZ // det(A)^{b-1}.
    pow_factor = det_A ** (b - 1) if b >= 1 else 1
    if pow_factor == 0:
        return integer_determinant(_assemble(A, B, C, D))
    q, r = divmod(detZ, pow_factor)
    if r != 0:
        # Algebraic invariant violated → fall back to direct Bareiss.
        return integer_determinant(_assemble(A, B, C, D))
    return q


def assemble(
    A: Sequence[Sequence[int]],
    B: Sequence[Sequence[int]],
    C: Sequence[Sequence[int]],
    D: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Public wrapper around :func:`_assemble`."""
    return _assemble(A, B, C, D)


# ---------------------------------------------------------------------------
# Simulated annealing over glue blocks
# ---------------------------------------------------------------------------


@dataclass
class PrefabSAResult:
    """Outcome of :func:`prefab_sa`."""

    a: int
    b: int
    abs_det: int
    full_matrix: list[list[int]]
    A: list[list[int]]
    D: list[list[int]]
    B: list[list[int]]
    C: list[list[int]]
    iterations: int
    accepted: int
    walltime_seconds: float
    history: list[int] = field(default_factory=list)


def _glue_random(a: int, b: int, rng: random.Random) -> tuple[
    list[list[int]], list[list[int]],
]:
    B = [[1 if rng.random() < 0.5 else -1 for _ in range(b)] for _ in range(a)]
    C = [[1 if rng.random() < 0.5 else -1 for _ in range(a)] for _ in range(b)]
    return B, C


def prefab_sa(
    a: int,
    b: int,
    *,
    A: Optional[list[list[int]]] = None,
    D: Optional[list[list[int]]] = None,
    iterations: int = 50_000,
    t_start: float = 1.0,
    t_end: float = 0.005,
    seed: int = 0,
    log_every: int = 0,
) -> PrefabSAResult:
    """SA over the (B, C) glue blocks holding A and D fixed.

    Parameters
    ----------
    a, b
        Sizes of the diagonal blocks. ``a + b`` is the assembled order.
    A, D
        Diagonal blocks; if omitted, loaded via :func:`load_proven_matrix`.
    iterations
        Total SA steps.
    t_start, t_end
        Geometric temperature schedule in ``log|det|`` units.
    seed
        PRNG seed (also used to initialise B, C if not provided).
    log_every
        If > 0, append best |det| every ``log_every`` steps.

    Returns
    -------
    PrefabSAResult
        Best assembled matrix and its exact-Bareiss-verified |det|.
    """
    if A is None:
        A = load_proven_matrix(a)
    if D is None:
        D = load_proven_matrix(b)
    if len(A) != a or any(len(r) != a for r in A):
        raise ValueError(f"A must be {a}×{a}")
    if len(D) != b or any(len(r) != b for r in D):
        raise ValueError(f"D must be {b}×{b}")

    rng = random.Random(seed)

    det_A = integer_determinant(A)
    if det_A == 0:
        raise ValueError("diagonal block A must be invertible")
    adj_A = _compute_adjugate([list(r) for r in A], det_A)

    B, C = _glue_random(a, b, rng)
    cur_det = schur_det(A, B, C, D, adj_A=adj_A, det_A=det_A)
    if cur_det == 0:
        # Try a few resamples to dodge a measure-zero singular start.
        for _ in range(8):
            B, C = _glue_random(a, b, rng)
            cur_det = schur_det(A, B, C, D, adj_A=adj_A, det_A=det_A)
            if cur_det != 0:
                break
    cur_e = -math.log(abs(cur_det)) if cur_det != 0 else math.inf

    best_det = cur_det
    best_B = [row[:] for row in B]
    best_C = [row[:] for row in C]
    best_e = cur_e

    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0
    accepted = 0
    history: list[int] = []
    start = time.monotonic()

    # Total free entries = a·b (B) + b·a (C). Index by (which, i, j).
    n_B = a * b
    n_total = 2 * a * b

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)

        idx = rng.randrange(n_total)
        if idx < n_B:
            i, j = divmod(idx, b)
            block = "B"
            old = B[i][j]
            B[i][j] = -old
        else:
            idx2 = idx - n_B
            i, j = divmod(idx2, a)
            block = "C"
            old = C[i][j]
            C[i][j] = -old

        new_det = schur_det(A, B, C, D, adj_A=adj_A, det_A=det_A)
        if new_det == 0:
            # Reject (infinite energy).
            if block == "B":
                B[i][j] = old
            else:
                C[i][j] = old
            if log_every and step % log_every == 0:
                history.append(abs(best_det))
            continue
        new_e = -math.log(abs(new_det))
        delta = new_e - cur_e
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            cur_det = new_det
            cur_e = new_e
            accepted += 1
            if abs(cur_det) > abs(best_det):
                best_det = cur_det
                best_B = [row[:] for row in B]
                best_C = [row[:] for row in C]
                best_e = cur_e
        else:
            if block == "B":
                B[i][j] = old
            else:
                C[i][j] = old

        if log_every and step % log_every == 0:
            history.append(abs(best_det))

    walltime = time.monotonic() - start

    full = _assemble(A, best_B, best_C, D)
    # Authoritative re-verification via Bareiss.
    abs_det = abs(integer_determinant(full))
    return PrefabSAResult(
        a=a,
        b=b,
        abs_det=abs_det,
        full_matrix=full,
        A=[row[:] for row in A],
        D=[row[:] for row in D],
        B=best_B,
        C=best_C,
        iterations=iterations,
        accepted=accepted,
        walltime_seconds=walltime,
        history=history,
    )


def prefab_sa_multistart(
    a: int,
    b: int,
    *,
    A: Optional[list[list[int]]] = None,
    D: Optional[list[list[int]]] = None,
    n_starts: int = 4,
    iterations: int = 50_000,
    t_start: float = 1.0,
    t_end: float = 0.005,
    base_seed: int = 1000,
) -> PrefabSAResult:
    """Multi-start wrapper around :func:`prefab_sa`. Returns best run."""
    if A is None:
        A = load_proven_matrix(a)
    if D is None:
        D = load_proven_matrix(b)
    best: Optional[PrefabSAResult] = None
    for s in range(n_starts):
        r = prefab_sa(
            a, b, A=A, D=D,
            iterations=iterations,
            t_start=t_start, t_end=t_end,
            seed=base_seed + s * 7919,
        )
        if best is None or r.abs_det > best.abs_det:
            best = r
    assert best is not None
    return best


__all__ = [
    "load_proven_matrix",
    "schur_det",
    "assemble",
    "prefab_sa",
    "prefab_sa_multistart",
    "PrefabSAResult",
]
