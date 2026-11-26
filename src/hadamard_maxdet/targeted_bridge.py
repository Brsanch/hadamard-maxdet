"""Targeted bridge from near-2-val matrices to the |G| ⊆ {1, 3} submanifold.

Background
----------
The smart 3-valued search round produced 4,245 distinct (count_1, count_3, count_5)
multisets at H23 (see ``smart_3val_search``). Among them, "near-2-val" classes
have ``count_5 ∈ {1, 2}``: only one or two off-diagonal Gram pairs sit at |G|=5,
the rest at |G| ∈ {1, 3}. Such matrices are *one* or *two* |G|=5 pairs away
from being 2-valued.

The targeted-bridge experiment asks: can a small set of single-entry flips
("targeted" because we only need to fix the |G|=5 pairs) snap a near-2-val
matrix back onto the |G| ⊆ {1, 3} submanifold without introducing any new
|G| ∉ {1, 3} entries? If yes, we land on a 2-valued matrix whose
(count_1, count_3) signature might be:

* The same as the source seed's 2-val ancestor (boring),
* One of {Orrick, BTC, KMS} or one of the 104 prior 2-val classes (already
  catalogued),
* Or a NEW 2-val class — potentially with |det| > Orrick's record at
  ``2,779,447,296,000,000``.

This module implements:

* :func:`load_near_2val_matrices_from_3val_run` — extract concrete matrices
  with ``count_5 ≤ K`` from ``results/smart_3val_search.json``.
* :func:`find_snap_to_2val_moves` — enumerate flip-sets of size ``≤ max_k``
  that bring ``count_5`` to 0 while keeping ``|G| ⊆ {1, 3}``.
* :func:`enumerate_bridges` — apply :func:`find_snap_to_2val_moves` over a
  set of near-2-val matrices, deduplicate destination 2-val signatures.
* :func:`cross_reference` — split discovered destinations into
  ``known`` vs ``new`` (where "known" = the 3 canonical {Orrick, BTC, KMS}
  plus the 104 prior 2-val multisets from
  ``results/smart_2val_search.json``).
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.smart_2val_search import (
    AbsClassMultiset,
    enumerate_known_2val_classes,
    gram_abs_class_signature,
    is_on_manifold,
)
from hadamard_maxdet.smart_3val_search import (
    AbsClass3State,
    is_on_manifold_3val,
)
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass
class NearTwoValMatrix:
    """A concrete matrix with ``count_5 ∈ {1, ..., K}`` from the 3-val run."""

    job_name: str
    signature: AbsClassMultiset  # ((1, c1), (3, c3), (5, c5))
    abs_det: int
    count_5: int
    matrix: list[list[int]]


@dataclass
class BridgeMove:
    """A single discovered snap-to-2-val flip-set on a near-2-val matrix."""

    source_job: str
    source_signature: AbsClassMultiset
    source_abs_det: int
    flip_positions: tuple[tuple[int, int], ...]  # (i, c) pairs, in (i, c) order
    destination_signature: AbsClassMultiset  # ((1, c1), (3, c3))
    destination_abs_det: int
    # Note: matrix re-derivable from source + flips, kept slim.


@dataclass
class BridgeAggregate:
    """Per-destination aggregation of all bridges that landed on it."""

    destination_signature: AbsClassMultiset
    best_abs_det: int  # max over all bridges to this destination
    best_matrix: list[list[int]]  # the matrix achieving best_abs_det
    n_bridges: int
    sample_source_jobs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loader — extract near-2-val matrices from the 3-val run
# ---------------------------------------------------------------------------


def load_near_2val_matrices_from_3val_run(
    json_path: str | Path,
    *,
    count_5_max: int = 2,
) -> list[NearTwoValMatrix]:
    """Load matrices from ``results/smart_3val_search.json`` whose own
    ``best_feasible_signature`` has ``count_5 ∈ {1, ..., count_5_max}``.

    The 3-val run records, per job, exactly one concrete matrix
    (``best_feasible_entries`` at ``best_feasible_signature``). Even though
    ``visited_signatures`` is much richer, only the per-job "best" matrix
    is materialised. We pick out those whose own signature is near-2-val.

    Parameters
    ----------
    json_path
        Path to the 3-val run JSON.
    count_5_max
        Upper inclusive bound on ``count_5``.

    Returns
    -------
    list[NearTwoValMatrix]
    """
    path = Path(json_path)
    with path.open() as f:
        data = json.load(f)
    out: list[NearTwoValMatrix] = []
    for rec in data.get("per_job", []):
        if "result" not in rec:
            continue
        r = rec["result"]
        # T1/T3 jobs serialise the WalkResult directly; T2 wraps it in 'walk'.
        walk = r if "best_feasible_signature" in r else r.get("walk")
        if walk is None:
            continue
        sig_raw = walk.get("best_feasible_signature")
        if sig_raw is None:
            continue
        entries = walk.get("best_feasible_entries")
        if entries is None:
            continue
        sig = tuple(tuple(p) for p in sig_raw)
        sig_dict = {p[0]: p[1] for p in sig}
        c5 = int(sig_dict.get(5, 0))
        if c5 < 1 or c5 > count_5_max:
            continue
        out.append(
            NearTwoValMatrix(
                job_name=rec["name"],
                signature=sig,
                abs_det=int(walk["best_feasible_abs_det"]),
                count_5=c5,
                matrix=[list(row) for row in entries],
            )
        )
    return out


# ---------------------------------------------------------------------------
# Snap-to-2-val flip-set enumeration
# ---------------------------------------------------------------------------


def _gram_counts(M: Sequence[Sequence[int]]) -> tuple[int, int, int, int]:
    """Return ``(c1, c3, c5, c_other)`` with ``c_*`` = #pairs of |G_offdiag| equal to *."""
    A = np.asarray(M, dtype=np.int64)
    n = A.shape[0]
    if n < 2:
        return 0, 0, 0, 0
    G = A @ A.T
    iu = np.triu_indices(n, k=1)
    vals = np.abs(G[iu])
    c1 = int(np.sum(vals == 1))
    c3 = int(np.sum(vals == 3))
    c5 = int(np.sum(vals == 5))
    total = n * (n - 1) // 2
    other = total - c1 - c3 - c5
    return c1, c3, c5, other


def _flip(M: list[list[int]], pos: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Return a new matrix with entries at each ``(i, c)`` in ``pos`` negated."""
    out = [row[:] for row in M]
    for i, c in pos:
        out[i][c] = -out[i][c]
    return out


def _is_2val_after_flips(
    M: Sequence[Sequence[int]],
    state: AbsClass3State,
    flips: Sequence[tuple[int, int]],
) -> tuple[bool, int, int, int]:
    """Test whether applying ``flips`` to ``M`` yields a |G| ⊆ {1, 3} matrix.

    Returns ``(ok, c1_after, c3_after, c5_after)``.

    Implementation: applies flips in-place to a copy of ``M``, recomputes the
    Gram support cardinalities exactly. Cost O(n^2) per call.

    NOTE: ``state`` is unused here but kept in the signature for symmetry
    with potentially-faster incremental versions in future revisions.
    """
    Mp = [row[:] for row in M]
    for i, c in flips:
        Mp[i][c] = -Mp[i][c]
    c1, c3, c5, other = _gram_counts(Mp)
    if c5 == 0 and other == 0:
        return True, c1, c3, c5
    return False, c1, c3, c5


def find_snap_to_2val_moves(
    M: Sequence[Sequence[int]],
    *,
    max_k: int = 2,
    n_max_solutions: int = 50,
) -> list[tuple[tuple[tuple[int, int], ...], AbsClassMultiset]]:
    """Find flip-sets of size ``≤ max_k`` that snap ``M`` onto |G| ⊆ {1, 3}.

    Strategy: brute-force enumeration over choose(n^2, k) for k = 1, 2, ...,
    ``max_k``. For each candidate flip-set, recompute the Gram support and
    accept if (a) ``count_5 == 0`` and (b) ``count_outside_{1,3} == 0``.

    Parameters
    ----------
    M
        Source matrix (must already have small ``count_5``; otherwise the
        search wastes time on hopeless candidates).
    max_k
        Maximum flip-set size to enumerate.
    n_max_solutions
        Stop early once this many distinct flip-sets snap to 2-val.

    Returns
    -------
    list of ``(flip_positions, destination_signature)`` pairs, where
    ``flip_positions`` is a sorted tuple of ``(i, c)`` and
    ``destination_signature`` is the resulting 2-val multiset.
    """
    A = np.asarray(M, dtype=np.int64)
    n = A.shape[0]
    if n < 2:
        return []
    state = AbsClass3State.from_matrix(M)
    if state.count_out != 0:
        # Source isn't even on the 3-val manifold. Snap is undefined.
        return []
    # If already 2-val, return the empty flip-set as a trivial solution.
    if state.count_abs5 == 0:
        sig = gram_abs_class_signature(M)
        return [((), sig)]

    positions = [(i, c) for i in range(n) for c in range(n)]
    found: list[tuple[tuple[tuple[int, int], ...], AbsClassMultiset]] = []
    seen_flip_sets: set[tuple[tuple[int, int], ...]] = set()

    for k in range(1, max_k + 1):
        for combo in combinations(positions, k):
            ok, c1, c3, c5 = _is_2val_after_flips(M, state, combo)
            if not ok:
                continue
            key = tuple(sorted(combo))
            if key in seen_flip_sets:
                continue
            seen_flip_sets.add(key)
            sig_parts: list[tuple[int, int]] = []
            if c1 > 0:
                sig_parts.append((1, c1))
            if c3 > 0:
                sig_parts.append((3, c3))
            sig: AbsClassMultiset = tuple(sig_parts)
            found.append((key, sig))
            if len(found) >= n_max_solutions:
                return found
    return found


# ---------------------------------------------------------------------------
# Bridge enumeration over a list of near-2-val matrices
# ---------------------------------------------------------------------------


def enumerate_bridges(
    near_2val: Sequence[NearTwoValMatrix],
    *,
    max_k: int = 2,
    n_max_solutions_per_source: int = 50,
    progress_every: int = 25,
    verbose: bool = False,
) -> tuple[list[BridgeMove], dict[AbsClassMultiset, BridgeAggregate]]:
    """Enumerate flip-bridges from each near-2-val matrix to the 2-val manifold.

    Parameters
    ----------
    near_2val
        Source near-2-val matrices.
    max_k
        Per-source ``max_k`` passed to :func:`find_snap_to_2val_moves`.
    n_max_solutions_per_source
        Cap per source on how many flip-sets to keep.
    progress_every
        Print a progress line every this many sources processed (when
        ``verbose=True``).
    verbose
        Print progress lines.

    Returns
    -------
    bridges : list[BridgeMove]
        Every successful bridge found.
    by_destination : dict[AbsClassMultiset, BridgeAggregate]
        Per-destination aggregation: ``best_abs_det``, ``best_matrix``,
        ``n_bridges``, sample source jobs.
    """
    bridges: list[BridgeMove] = []
    by_dest: dict[AbsClassMultiset, BridgeAggregate] = {}

    for idx, src in enumerate(near_2val):
        moves = find_snap_to_2val_moves(
            src.matrix,
            max_k=max_k,
            n_max_solutions=n_max_solutions_per_source,
        )
        for flip_positions, dest_sig in moves:
            # Empty flip-set means the source was *already* 2-val (shouldn't
            # happen for count_5 >= 1 but defensive).
            if len(flip_positions) == 0:
                continue
            dest_M = _flip(src.matrix, flip_positions)
            dest_det = abs(integer_determinant(dest_M))
            bridge = BridgeMove(
                source_job=src.job_name,
                source_signature=src.signature,
                source_abs_det=src.abs_det,
                flip_positions=flip_positions,
                destination_signature=dest_sig,
                destination_abs_det=dest_det,
            )
            bridges.append(bridge)
            agg = by_dest.get(dest_sig)
            if agg is None:
                by_dest[dest_sig] = BridgeAggregate(
                    destination_signature=dest_sig,
                    best_abs_det=dest_det,
                    best_matrix=dest_M,
                    n_bridges=1,
                    sample_source_jobs=[src.job_name],
                )
            else:
                agg.n_bridges += 1
                if dest_det > agg.best_abs_det:
                    agg.best_abs_det = dest_det
                    agg.best_matrix = dest_M
                if len(agg.sample_source_jobs) < 5 and src.job_name not in agg.sample_source_jobs:
                    agg.sample_source_jobs.append(src.job_name)
        if verbose and (idx + 1) % progress_every == 0:
            print(
                f"  [bridge] {idx + 1}/{len(near_2val)} sources processed; "
                f"bridges={len(bridges)} distinct dests={len(by_dest)}"
            )
    return bridges, by_dest


# ---------------------------------------------------------------------------
# Cross-reference against known 2-val classes
# ---------------------------------------------------------------------------


def load_known_2val_signatures(
    *,
    results_dir: str | Path = "results",
    smart_2val_path: str | Path = "results/smart_2val_search.json",
) -> set[AbsClassMultiset]:
    """Return the set of 2-val signatures we already know about: the three
    canonical {Orrick, BTC, KMS} classes plus the 104 prior multisets from
    the 2-val smart-search round.
    """
    known: set[AbsClassMultiset] = set()
    canonical = enumerate_known_2val_classes(results_dir)
    for kc in canonical:
        known.add(kc.abs_class_multiset)
    p2 = Path(smart_2val_path)
    if p2.exists():
        with p2.open() as f:
            data = json.load(f)
        for rec in data.get("new_multisets", []):
            sig = tuple(tuple(p) for p in rec["signature"])
            known.add(sig)
    return known


def cross_reference(
    by_destination: dict[AbsClassMultiset, BridgeAggregate],
    known_2val: set[AbsClassMultiset],
) -> tuple[dict[AbsClassMultiset, BridgeAggregate],
           dict[AbsClassMultiset, BridgeAggregate]]:
    """Split destinations into ``(already_known, new)`` 2-val classes."""
    already: dict[AbsClassMultiset, BridgeAggregate] = {}
    new: dict[AbsClassMultiset, BridgeAggregate] = {}
    for sig, agg in by_destination.items():
        if sig in known_2val:
            already[sig] = agg
        else:
            new[sig] = agg
    return already, new


# ---------------------------------------------------------------------------
# Triple verification on any > Orrick candidate
# ---------------------------------------------------------------------------


ORRICK_DET = 2_779_447_296_000_000


def triple_verify_abs_det(entries: Sequence[Sequence[int]]) -> int:
    """Twice on M, once on Mᵀ; assert agreement; return |det|."""
    M = [[int(v) for v in row] for row in entries]
    Mt = [[M[j][i] for j in range(len(M))] for i in range(len(M))]
    d1 = abs(integer_determinant(M))
    d2 = abs(integer_determinant(M))
    d3 = abs(integer_determinant(Mt))
    if not (d1 == d2 == d3):
        raise RuntimeError(
            f"triple-verify disagreement: d1={d1} d2={d2} d3(transpose)={d3}"
        )
    return d1


__all__ = [
    "ORRICK_DET",
    "NearTwoValMatrix",
    "BridgeMove",
    "BridgeAggregate",
    "load_near_2val_matrices_from_3val_run",
    "find_snap_to_2val_moves",
    "enumerate_bridges",
    "load_known_2val_signatures",
    "cross_reference",
    "triple_verify_abs_det",
]
