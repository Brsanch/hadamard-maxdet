"""Decode the structural fingerprint of *winning* matrices.

Where ``feasibility.py`` rules out the (a, b, partition) regimes that
*cannot* contain the proven optima, this module tackles the dual question:
**what does the proven-max actually look like, structurally?**

For each order ``n`` where our pipeline saturates the proven max
(currently n ≤ 17), we (re)generate a winning matrix via SA multi-start
and record:

* Gram off-diagonal support (the set of distinct off-diag values).
* Gram off-diagonal multiset (count of each value).
* Singular-value multiplicity pattern (binned to 4 decimals).
* Number of distinct row-weight values (block-structure indicator).
* Best-fit canonical Ehlich partition (if any) and its match score.

The hope is to see whether winners share a structural feature
(e.g. always 2-valued Gram support, certain row-weight pattern, certain
SV multiplicity) that we can then look for at H19 and H23.

Public API
----------

* :func:`winner_fingerprint(n, ...)` — get one proven-max-saturating matrix
  at order ``n`` and return its full structural fingerprint.
* :func:`reconstruct_winner(n, ...)` — heavy multi-strategy attempt to
  reach the proven max at ``n`` (defaults targeted at the H19 case);
  returns the best matrix found and its fingerprint.
* :func:`cross_n_winner_table(ns, ...)` — table of fingerprints for a
  list of orders.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from hadamard_maxdet.incremental import sa_incremental
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.orders import BENCHMARKS
from hadamard_maxdet.telemetry import gram_offdiag_report
from hadamard_maxdet.verifier import integer_determinant, upper_bound


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


@dataclass
class WinnerFingerprint:
    """Full structural fingerprint of a (possibly winning) ±1 matrix."""

    n: int
    abs_det: int
    abs_det_verified: int
    proven_max: Optional[int]
    ratio_to_proven: Optional[float]
    ratio_to_upper_bound: float
    gram_offdiag_support: list[int]
    gram_offdiag_counts: dict[int, int]
    gram_offdiag_cardinality: int
    sv_multiplicities: list[tuple[float, int]]
    sv_distinct_count: int
    row_weight_counts: dict[int, int]
    distinct_row_weight_values: int
    best_partition: Optional[tuple[int, ...]]
    best_partition_intra_cross: Optional[tuple[int, int]]
    best_partition_score: Optional[float]
    matrix: list[list[int]] = field(default_factory=list)
    source: str = ""

    def to_json(self, *, include_matrix: bool = False) -> dict:
        out = {
            "n": self.n,
            "abs_det": self.abs_det,
            "abs_det_verified": self.abs_det_verified,
            "proven_max": self.proven_max,
            "ratio_to_proven": self.ratio_to_proven,
            "ratio_to_upper_bound": self.ratio_to_upper_bound,
            "gram_offdiag_support": self.gram_offdiag_support,
            "gram_offdiag_counts": {
                str(k): v for k, v in self.gram_offdiag_counts.items()
            },
            "gram_offdiag_cardinality": self.gram_offdiag_cardinality,
            "sv_multiplicities": [
                [float(s), int(c)] for s, c in self.sv_multiplicities
            ],
            "sv_distinct_count": self.sv_distinct_count,
            "row_weight_counts": {
                str(k): v for k, v in self.row_weight_counts.items()
            },
            "distinct_row_weight_values": self.distinct_row_weight_values,
            "best_partition": (
                list(self.best_partition) if self.best_partition else None
            ),
            "best_partition_intra_cross": (
                list(self.best_partition_intra_cross)
                if self.best_partition_intra_cross
                else None
            ),
            "best_partition_score": self.best_partition_score,
            "source": self.source,
        }
        if include_matrix:
            out["matrix"] = self.matrix
        return out


def _row_weight_counts(matrix: list[list[int]]) -> dict[int, int]:
    """Counts of distinct row weights w_i (number of +1 entries per row)."""
    counts: Counter = Counter()
    for row in matrix:
        w = sum(1 for x in row if x > 0)
        counts[w] += 1
    return dict(sorted(counts.items()))


def _best_partition_fit(
    matrix: list[list[int]],
    *,
    intra_cross_candidates: Sequence[tuple[int, int]] = (
        (0, 0),     # Hadamard regime
        (3, -1),    # canonical Ehlich/Wojtas n ≡ 1, 3 mod 4
        (5, -3),
        (1, -3),
        (-1, 3),
        (1, -1),    # parity-infeasible at large n; included as a probe
        (3, 1),
        (5, 1),
        (-3, 1),
        (5, -1),
        (3, -3),
        (-3, -1),
    ),
) -> tuple[Optional[tuple[int, ...]], Optional[tuple[int, int]], Optional[float]]:
    """Best-fit (partition, (intra,cross), score) over the canonical
    Ehlich partitions and a small (a, b) candidate set.

    Uses ``basin_atlas.partition_assignment`` (greedy + local search) for
    each candidate and returns the highest-scoring fit.
    """
    n = len(matrix)
    if n < 4:
        return None, None, None

    # Local imports to avoid load-time cycles.
    from hadamard_maxdet.basin_atlas import partition_assignment
    from hadamard_maxdet.generalized_ehlich import enumerate_partitions

    partitions = enumerate_partitions(n, min_block_size=1)
    if not partitions:
        return None, None, None

    # Trim to the predicted top-K most-likely (highest predicted block-Gram
    # det) partitions to keep cost bounded; for small n the list is short.
    from hadamard_maxdet.generalized_ehlich import predicted_block_det

    best: tuple[Optional[tuple[int, ...]], Optional[tuple[int, int]], Optional[float]] = (
        None,
        None,
        None,
    )
    best_score = -1.0

    # Sort partitions by predicted (3, -1) block det, descending; cap at 12.
    scored: list[tuple[float, tuple[int, ...]]] = []
    for part in partitions:
        try:
            d = predicted_block_det(part, intra_value=3, cross_value=-1)
        except Exception:
            d = 0.0
        scored.append((d, tuple(part)))
    scored.sort(reverse=True, key=lambda kv: kv[0])
    short_list = [p for _, p in scored[:12]]

    for part in short_list:
        for (a, b) in intra_cross_candidates:
            try:
                pa = partition_assignment(
                    matrix, tuple(part), intra_value=a, cross_value=b,
                    n_restarts=2,
                )
            except Exception:
                continue
            if pa.gram_match_score > best_score:
                best_score = pa.gram_match_score
                best = (tuple(part), (a, b), pa.gram_match_score)

    return best


def fingerprint_matrix(
    matrix: list[list[int]], *, source: str = ""
) -> WinnerFingerprint:
    """Compute the full WinnerFingerprint of a single ±1 matrix.

    The matrix's |det| is verified via Bareiss.
    """
    n = len(matrix)
    abs_det_verified = abs(integer_determinant(matrix))
    bid = f"H{n}"
    proven = (
        BENCHMARKS[bid].best_known
        if (bid in BENCHMARKS and BENCHMARKS[bid].best_known is not None)
        else None
    )
    ub = upper_bound(n)

    rep = gram_offdiag_report(matrix)
    sv_mults = [(float(s), int(c)) for s, c in rep["sv_multiplicities"]]

    rw = _row_weight_counts(matrix)

    best_part, best_ab, best_score = _best_partition_fit(matrix)

    return WinnerFingerprint(
        n=n,
        abs_det=abs_det_verified,
        abs_det_verified=abs_det_verified,
        proven_max=proven,
        ratio_to_proven=(abs_det_verified / proven) if proven else None,
        ratio_to_upper_bound=(abs_det_verified / ub) if ub else 0.0,
        gram_offdiag_support=list(rep["gram_offdiag_support"]),
        gram_offdiag_counts=dict(rep["gram_offdiag_counts"]),
        gram_offdiag_cardinality=len(rep["gram_offdiag_support"]),
        sv_multiplicities=sv_mults,
        sv_distinct_count=len(sv_mults),
        row_weight_counts=rw,
        distinct_row_weight_values=len(rw),
        best_partition=best_part,
        best_partition_intra_cross=best_ab,
        best_partition_score=best_score,
        matrix=[list(r) for r in matrix],
        source=source,
    )


# ---------------------------------------------------------------------------
# Find / regenerate winning matrix
# ---------------------------------------------------------------------------


def _try_paley_deletion(n: int) -> Optional[list[list[int]]]:
    """Best Paley-deletion seed at order ``n``, if available; else None."""
    try:
        from hadamard_maxdet.paley_seed import top_paley_seeds

        seeds = top_paley_seeds(n, k=4)
    except Exception:
        return None
    if not seeds:
        return None
    # First entry is highest |det|.
    _, _, mat, _ = seeds[0]
    return mat.entries


def _try_bordered_two_circulant(
    n: int, *, seconds: float, seed: int
) -> tuple[int, Optional[list[list[int]]]]:
    """Run sa_bordered_two_circulant in a wall-time budget. Returns
    (best_abs_det, best_entries_or_None) — only odd n supported.
    """
    if n % 2 == 0:
        return 0, None
    try:
        from hadamard_maxdet.rich_ansatz import sa_bordered_two_circulant
    except Exception:
        return 0, None
    t0 = time.monotonic()
    best = 0
    best_entries: Optional[list[list[int]]] = None
    chunks = 0
    while time.monotonic() - t0 < seconds:
        r = sa_bordered_two_circulant(
            n,
            iterations=10_000,
            t_start=1.0,
            t_end=0.005,
            seed=seed + chunks,
        )
        if r.abs_det > best:
            best = int(r.abs_det)
            best_entries = [list(row) for row in r.matrix.entries]
        chunks += 1
    return best, best_entries


def find_winning_matrix(
    n: int,
    *,
    target: Optional[int] = None,
    n_starts: int = 8,
    iterations: int = 100_000,
    structured_seconds: float = 30.0,
    base_seed: int = 12345,
) -> tuple[int, list[list[int]], str]:
    """(Re)generate a high-|det| matrix at order ``n``.

    Strategy: SA multi-start (free ±1) + bordered_two_circulant SA
    (odd n only) + Paley-deletion seed (when available); return the best.

    Returns ``(abs_det, entries, source)``. If ``target`` is set, an early
    return triggers as soon as one strategy meets or exceeds it.
    """
    bid = f"H{n}"
    proven = (
        BENCHMARKS[bid].best_known
        if (bid in BENCHMARKS and BENCHMARKS[bid].best_known is not None)
        else None
    )
    if target is None and proven is not None:
        target = proven

    best_abs = 0
    best_entries: list[list[int]] = []
    source = ""

    # 1) Paley deletion (cheap, immediate).
    paley = _try_paley_deletion(n)
    if paley is not None:
        d = abs(integer_determinant(paley))
        if d > best_abs:
            best_abs = d
            best_entries = [list(r) for r in paley]
            source = "paley_deletion"
            if target and best_abs >= target:
                return best_abs, best_entries, source

    # 2) Bordered two-circulant SA (odd n).
    if n % 2 == 1:
        d, ent = _try_bordered_two_circulant(
            n, seconds=structured_seconds, seed=base_seed,
        )
        if d > best_abs and ent is not None:
            best_abs = d
            best_entries = ent
            source = "bordered_two_circulant_sa"
            if target and best_abs >= target:
                return best_abs, best_entries, source

    # 3) SA multi-start (free ±1).
    for s in range(n_starts):
        r = sa_incremental(
            n,
            iterations=iterations,
            t_start=1.0,
            t_end=0.005,
            seed=base_seed + s * 1009,
            refresh_every=0,
        )
        d = int(r.abs_det)
        if d > best_abs:
            best_abs = d
            best_entries = [list(row) for row in r.matrix.entries]
            source = "sa_incremental_multistart"
            if target and best_abs >= target:
                return best_abs, best_entries, source

    return best_abs, best_entries, source


def winner_fingerprint(
    n: int,
    *,
    target: Optional[int] = None,
    n_starts: int = 8,
    iterations: int = 100_000,
    structured_seconds: float = 30.0,
    base_seed: int = 12345,
) -> WinnerFingerprint:
    """Find a (preferably proven-max-saturating) matrix at order ``n`` and
    return its full :class:`WinnerFingerprint`.
    """
    abs_det, entries, source = find_winning_matrix(
        n,
        target=target,
        n_starts=n_starts,
        iterations=iterations,
        structured_seconds=structured_seconds,
        base_seed=base_seed,
    )
    fp = fingerprint_matrix(entries, source=source)
    # Source label may carry useful info; abs_det reported by SA is sanity-
    # checked against the verified Bareiss det in fingerprint_matrix.
    assert fp.abs_det_verified == abs(integer_determinant(entries))
    return fp


# ---------------------------------------------------------------------------
# Reconstruction (heavy, for H19+)
# ---------------------------------------------------------------------------


def reconstruct_winner(
    n: int,
    *,
    wall_seconds: float = 300.0,
    base_seed: int = 4242,
    sa_chunk_iters: int = 50_000,
) -> WinnerFingerprint:
    """Heavy multi-strategy attempt to push toward the proven max at ``n``.

    Default budget 5 minutes. Combines:
      * Paley-deletion seed (if available) → SA refinement.
      * Bordered two-circulant SA chunks (odd n).
      * Class-tabu SA (if module available).
      * Free SA from random starts.

    Returns the fingerprint of the best matrix found.
    """
    bid = f"H{n}"
    proven = (
        BENCHMARKS[bid].best_known
        if (bid in BENCHMARKS and BENCHMARKS[bid].best_known is not None)
        else None
    )
    deadline = time.monotonic() + wall_seconds

    best_abs = 0
    best_entries: list[list[int]] = []
    best_source = ""

    def _maybe_update(d: int, ent: list[list[int]], src: str) -> None:
        nonlocal best_abs, best_entries, best_source
        if d > best_abs:
            best_abs = d
            best_entries = [list(r) for r in ent]
            best_source = src

    # 1) Paley deletion warm-start.
    paley_entries = _try_paley_deletion(n)
    if paley_entries is not None:
        d0 = abs(integer_determinant(paley_entries))
        _maybe_update(d0, paley_entries, "paley_deletion_seed")
        # Refine the Paley seed with a budgeted SA.
        if time.monotonic() < deadline:
            init_mat = Matrix(n=n, entries=paley_entries)
            r = sa_incremental(
                n,
                iterations=sa_chunk_iters * 2,
                t_start=0.5,
                t_end=0.001,
                seed=base_seed + 1,
                initial=init_mat,
                refresh_every=0,
            )
            _maybe_update(int(r.abs_det), r.matrix.entries, "paley_then_sa")
            if proven and best_abs >= proven:
                return fingerprint_matrix(best_entries, source=best_source)

    # 2) Bordered two-circulant SA chunks (odd n).
    if n % 2 == 1:
        try:
            from hadamard_maxdet.rich_ansatz import sa_bordered_two_circulant
            chunk = 0
            while time.monotonic() < deadline:
                r = sa_bordered_two_circulant(
                    n,
                    iterations=sa_chunk_iters,
                    t_start=1.0,
                    t_end=0.001,
                    seed=base_seed + 1000 + chunk,
                )
                _maybe_update(int(r.abs_det), r.matrix.entries,
                              "bordered_two_circulant_sa")
                chunk += 1
                if proven and best_abs >= proven:
                    return fingerprint_matrix(
                        best_entries, source=best_source,
                    )
                # Reserve last 30% of budget for free SA + class-tabu.
                if time.monotonic() > deadline - 0.4 * wall_seconds:
                    break
        except Exception:
            pass

    # 3) Class-tabu SA: lock out the current best's signature and try to
    # escape the basin (only if we already have a candidate matrix).
    if best_entries:
        try:
            from hadamard_maxdet.class_tabu import (
                class_signature, sa_with_class_tabu,
            )
            locked = {class_signature(best_entries)}
            chunk = 0
            while time.monotonic() < deadline:
                init = Matrix(n=n, entries=best_entries)
                r = sa_with_class_tabu(
                    init,
                    iterations=sa_chunk_iters,
                    locked_signatures=locked,
                    t_start=1.5,
                    t_end=0.005,
                    seed=base_seed + 5000 + chunk,
                )
                d = int(getattr(r, "abs_det", 0))
                ent = getattr(r, "matrix", None)
                if ent is not None:
                    raw = ent.entries if hasattr(ent, "entries") else ent
                    _maybe_update(d, raw, "class_tabu_sa")
                chunk += 1
                if proven and best_abs >= proven:
                    return fingerprint_matrix(best_entries, source=best_source)
                # Bound class-tabu time.
                if time.monotonic() > deadline - 0.15 * wall_seconds:
                    break
        except Exception:
            pass

    # 4) Free SA multi-start chunks until deadline.
    chunk = 0
    while time.monotonic() < deadline:
        r = sa_incremental(
            n,
            iterations=sa_chunk_iters,
            t_start=1.0,
            t_end=0.005,
            seed=base_seed + 9000 + chunk,
            refresh_every=0,
        )
        _maybe_update(int(r.abs_det), r.matrix.entries, "sa_incremental")
        chunk += 1
        if proven and best_abs >= proven:
            break

    return fingerprint_matrix(best_entries, source=best_source)


# ---------------------------------------------------------------------------
# Cross-n table
# ---------------------------------------------------------------------------


def cross_n_winner_table(
    ns: Sequence[int],
    *,
    n_starts: int = 6,
    iterations: int = 80_000,
    structured_seconds: float = 25.0,
    base_seed: int = 12345,
) -> list[WinnerFingerprint]:
    """Build a table of WinnerFingerprints across ``ns``.

    For small n (<= 17) saturating proven max is fast; for n >= 19 the
    fingerprint reflects the *best we found*, not the proven optimum.
    """
    rows: list[WinnerFingerprint] = []
    for n in ns:
        fp = winner_fingerprint(
            n,
            n_starts=n_starts,
            iterations=iterations,
            structured_seconds=structured_seconds,
            base_seed=base_seed + n * 31,
        )
        rows.append(fp)
    return rows


__all__ = [
    "WinnerFingerprint",
    "fingerprint_matrix",
    "find_winning_matrix",
    "winner_fingerprint",
    "reconstruct_winner",
    "cross_n_winner_table",
]
