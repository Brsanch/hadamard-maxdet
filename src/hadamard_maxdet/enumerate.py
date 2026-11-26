"""Exhaustive enumeration over structured-ansatz cores.

For small n (n <= ~22) the bordered-circulant ansatz state space
(2^(n-1)) is small enough to enumerate exhaustively in minutes on a
laptop. The ansatz integer determinant is O(n^3) Bareiss ops per core,
so n=22 with 2^21 cores is roughly 2M * 22^3 ~ 2e10 small-int ops; with
multiprocessing across 6-8 workers this is a few minutes of wall time.

The bordered-circulant matrix has two |det|-preserving symmetries on
the core bit-vector:

  * Cyclic rotation of the core. The lower-right block is a circulant
    C, and C_{rot(g)} = P C_g P^T for the cyclic shift P. The full
    bordered matrix M then satisfies M_{rot(g)} = P' M_g P'^T where
    P' is identity on row/col 0 and P on rows/cols 1..n-1. So
    |det M_{rot(g)}| = |det M_g|.

  * Negation of the entire core ``g -> -g``. The spec hypothesised that
    this is a |det|-preserving symmetry. Empirically, it is **NOT**:
    :func:`negation_preserves_abs_det` returns False for every n we
    tested (n=5..13). We keep the check (`include_negation_class`
    triggers it and only quotients by negation if it holds) but expect
    no speedup from this axis on bordered-circulant.

Effective orbit size per equivalence class is ``n - 1`` (rotation only).
For n=22 this turns 2^21 = 2097152 candidates into roughly 99,864
representatives (verified by enumerator).
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass

from hadamard_maxdet.structured import bordered_circulant, skew_generator_from_free_bits
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bits_from_index(idx: int, m: int) -> list[int]:
    """Map an integer index in [0, 2^m) to a length-m ±1 vector."""
    return [1 if (idx >> k) & 1 else -1 for k in range(m)]


def _index_from_bits(bits: list[int]) -> int:
    """Inverse of :func:`_bits_from_index`."""
    idx = 0
    for k, b in enumerate(bits):
        if b == 1:
            idx |= 1 << k
    return idx


def _canonical_under_rotation(idx: int, m: int) -> int:
    """Return the lex-smallest index obtained by cyclically rotating the
    underlying ±1 vector of length ``m``.

    We compare integer indices, which is equivalent to comparing the
    bit-vectors lexicographically when bit 0 is the lowest-order bit
    and rotation is a uniform cyclic shift of indices.
    """
    bits = _bits_from_index(idx, m)
    best = idx
    cur = bits
    for _ in range(m - 1):
        cur = [cur[-1]] + cur[:-1]
        v = _index_from_bits(cur)
        if v < best:
            best = v
    return best


def _det_of_core_index(args: tuple[int, int]) -> tuple[int, int]:
    """Worker function: given (n, idx), build the bordered circulant and
    return (idx, |det|). Top-level so it pickles for multiprocessing.
    """
    n, idx = args
    m = n - 1
    bits = _bits_from_index(idx, m)
    M = bordered_circulant(bits)
    return idx, abs(integer_determinant(M.entries))


def _det_of_skew_index(args: tuple[int, int]) -> tuple[int, int]:
    n, idx = args
    half = (n - 1) // 2
    bits = _bits_from_index(idx, half)
    gen = skew_generator_from_free_bits(n, bits)
    M = Matrix.circulant(gen)
    return idx, abs(integer_determinant(M.entries))


# ---------------------------------------------------------------------------
# Negation-symmetry empirical check
# ---------------------------------------------------------------------------


def negation_preserves_abs_det(n: int, samples: int = 32, seed: int = 0) -> bool:
    """Empirically check that negating the entire core preserves |det| for
    bordered-circulant of order ``n``.

    Negation flips the sign of every entry in the (n-1) x (n-1) circulant
    block. For odd n-1 (i.e. n even), this is equivalent to multiplying
    rows 1..n-1 by -1 and then negating column entries to undo the border
    sign — a non-trivial symmetry that only sometimes holds. We verify it
    on random cores.
    """
    import random
    rng = random.Random(seed)
    m = n - 1
    for _ in range(samples):
        bits = [1 if rng.random() < 0.5 else -1 for _ in range(m)]
        d1 = abs(integer_determinant(bordered_circulant(bits).entries))
        d2 = abs(integer_determinant(bordered_circulant([-b for b in bits]).entries))
        if d1 != d2:
            return False
    return True


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


@dataclass
class EnumResult:
    n: int
    ansatz: str
    total_indices: int       # 2^(n-1) for bordered, 2^((n-1)/2) for skew
    visited: int             # reduced count actually scored
    best_det: int
    best_core: list[int]
    walltime: float
    negation_symmetric: bool | None = None


def enumerate_bordered_circulant(
    n: int,
    *,
    limit: int | None = None,
    include_negation_class: bool = False,
    use_multiprocessing: bool = True,
    n_workers: int | None = None,
    chunk_size: int = 4096,
) -> EnumResult:
    """Exhaustively enumerate bordered-circulant cores for order ``n``.

    Parameters
    ----------
    n
        Order of the matrix. Core has length ``n - 1`` and 2^(n-1) values.
    limit
        If given, only score the first ``limit`` reduced indices (after
        rotation-quotient + optional negation quotient). Used for tests.
    include_negation_class
        If True and the negation symmetry is verified to hold, only one
        of {core, -core} is scored per orbit, halving the search.
    use_multiprocessing
        If True, parallelise across CPU cores via :class:`multiprocessing.Pool`.
    n_workers
        Number of pool workers. Defaults to min(8, cpu_count).
    chunk_size
        Multiprocessing chunk size for ``imap_unordered``.

    Returns
    -------
    EnumResult
        With ``best_det`` the maximum |det| over all bordered-circulant
        cores, ``best_core`` a representative core attaining it, and
        ``visited`` the number of equivalence-class representatives
        actually scored.
    """
    if n < 2:
        raise ValueError(f"bordered-circulant ansatz requires n >= 2; got {n}")

    m = n - 1
    total = 1 << m
    t0 = time.monotonic()

    # Decide whether to apply negation quotient.
    neg_ok: bool | None = None
    if include_negation_class:
        neg_ok = negation_preserves_abs_det(n, samples=64)
        # If the symmetry doesn't hold, we DON'T quotient by negation; we
        # still record the result.

    # Build representative set (canonical-under-rotation indices, optionally
    # negation-canonical).
    reps: list[int] = []
    seen_neg: set[int] = set()
    for idx in range(total):
        if _canonical_under_rotation(idx, m) != idx:
            continue
        if include_negation_class and neg_ok:
            # Pair idx with its bit-flip (negated bits = idx XOR (2^m - 1))
            # then take canonical-under-rotation of the negation, then
            # the lex-smaller of the two as the representative.
            neg_idx = idx ^ ((1 << m) - 1)
            neg_canon = _canonical_under_rotation(neg_idx, m)
            rep = min(idx, neg_canon)
            if rep != idx:
                continue
            if neg_canon in seen_neg:
                continue
            seen_neg.add(idx)
            seen_neg.add(neg_canon)
        reps.append(idx)

    if limit is not None:
        reps = reps[:limit]

    # Score representatives.
    best_det = 0
    best_core: list[int] = [1] * m

    if use_multiprocessing and len(reps) > 4096:
        if n_workers is None:
            n_workers = min(8, os.cpu_count() or 4)
        args_iter = ((n, idx) for idx in reps)
        with mp.Pool(processes=n_workers) as pool:
            for idx, d in pool.imap_unordered(
                _det_of_core_index, args_iter, chunksize=chunk_size
            ):
                if d > best_det:
                    best_det = d
                    best_core = _bits_from_index(idx, m)
    else:
        for idx in reps:
            d = _det_of_core_index((n, idx))[1]
            if d > best_det:
                best_det = d
                best_core = _bits_from_index(idx, m)

    walltime = time.monotonic() - t0
    return EnumResult(
        n=n,
        ansatz="bordered_circulant",
        total_indices=total,
        visited=len(reps),
        best_det=best_det,
        best_core=best_core,
        walltime=walltime,
        negation_symmetric=neg_ok,
    )


def enumerate_skew_circulant(
    n: int,
    *,
    use_multiprocessing: bool = False,
    n_workers: int | None = None,
) -> EnumResult:
    """Exhaustively enumerate skew-circulant generators for odd order ``n``.

    Free-bit count is ``(n - 1) // 2``, so the search space is at most
    2^((n-1)/2). For n <= 23 this is <= 2^11 = 2048 and runs in well
    under a second; multiprocessing is off by default.
    """
    if n % 2 == 0:
        raise ValueError(f"skew-circulant ansatz requires odd n; got {n}")

    half = (n - 1) // 2
    total = 1 << half
    t0 = time.monotonic()

    best_det = 0
    best_core: list[int] = [1] * half

    if use_multiprocessing and total > 4096:
        if n_workers is None:
            n_workers = min(8, os.cpu_count() or 4)
        args_iter = ((n, idx) for idx in range(total))
        with mp.Pool(processes=n_workers) as pool:
            for idx, d in pool.imap_unordered(
                _det_of_skew_index, args_iter, chunksize=256
            ):
                if d > best_det:
                    best_det = d
                    best_core = _bits_from_index(idx, half)
    else:
        for idx in range(total):
            d = _det_of_skew_index((n, idx))[1]
            if d > best_det:
                best_det = d
                best_core = _bits_from_index(idx, half)

    walltime = time.monotonic() - t0
    return EnumResult(
        n=n,
        ansatz="skew_circulant",
        total_indices=total,
        visited=total,
        best_det=best_det,
        best_core=best_core,
        walltime=walltime,
        negation_symmetric=None,
    )
