"""Basin atlas — comprehensive map of H23 (and general n) local-max basins.

Six diagnostics over a curated set of distinct local-max ±1 matrices:

1. Pairwise distance matrices (Hamming / Gram-EMD / spectral L2).
2. 2D MDS embedding from the Hamming distance matrix.
3. Saddle-height search between two basins via interpolated waypoints
   with constrained SA relaxation.
4. Ehlich-partition coverage: try to fit each basin into the predicted
   high-|det| 4-block partitions enumerated by :mod:`orrick_hunt`.
5. Distance-from-anchor profile: (Hamming(M, anchor), |det|/upper_bound)
   trend.
6. Connected-components under bounded random walks: which basins
   reach which others under a 2-flip walk constrained to keep |det|
   above a fraction of the anchor's |det|.

All six are pure offline analytics on existing matrices: no full SA
campaigns are launched. The saddle search runs O(K · sa_iters) brief
constrained SA passes where K ≈ 15 and sa_iters defaults to 5000.

Public API
----------

* :func:`load_all_h23_basins` — read every H23 matrix on disk plus
  Williamson H23, KMS Prop 28 H23, and the top Paley-deletion H23,
  deduplicate by Gram class signature, and return a list of
  :class:`BasinEntry` records sorted by descending ``|det|``.
* :func:`pairwise_distances` — three N×N distance matrices.
* :func:`mds_embed` — 2D MDS embedding of a distance matrix.
* :func:`saddle_search_between` — interpolated path with SA-relaxed
  waypoints; reports the minimum ``|det|`` along the path.
* :func:`partition_assignment` — best row-to-block assignment fitting
  a target intra/cross Gram pattern; returns Gram-loss + assignment.
* :func:`bounded_walk_reachability` — per-basin random walk reaching
  other basins' Gram class signatures.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hadamard_maxdet.class_tabu import ClassSignature, class_signature
from hadamard_maxdet.incremental import (
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    refresh,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant, upper_bound


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BasinEntry:
    """A single local-max basin representative."""

    label: str
    matrix: list[list[int]]
    abs_det: int
    signature: ClassSignature
    source: str

    @property
    def n(self) -> int:
        return len(self.matrix)

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "abs_det": self.abs_det,
            "signature": [list(p) for p in self.signature],
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Loading + deduplication
# ---------------------------------------------------------------------------


def _matrix_from_dict(d: dict) -> list[list[int]] | None:
    """Extract a matrix payload from one of the H23 result-file shapes."""
    for key in (
        "matrix_entries",
        "entries",
        "overall_best_matrix",
        "matrix",
        "final_matrix",
    ):
        if key in d and isinstance(d[key], list) and d[key] and isinstance(d[key][0], list):
            return [[int(v) for v in row] for row in d[key]]
    return None


def _safe_williamson_h23() -> list[list[int]] | None:
    """Build the default GS-array Williamson H23 if the helper is available."""
    try:
        from hadamard_maxdet.williamson import (
            gs_array_williamson_h23,
            williamson_quadruples_m5,
        )
    except Exception:
        return None
    quads = williamson_quadruples_m5()
    if not quads:
        return None
    A, B, C, D = quads[0]
    try:
        M = gs_array_williamson_h23(A, B, C, D, e=[1, 1, 1], variant="border3")
    except Exception:
        return None
    return [list(row) for row in M.entries]


def _safe_kms_h23() -> list[list[int]] | None:
    try:
        from hadamard_maxdet.kms_construction import prop28_matrix
    except Exception:
        return None
    try:
        M = prop28_matrix(5)
    except Exception:
        return None
    return [list(row) for row in M.entries]


def _safe_top_paley_h23() -> list[list[int]] | None:
    try:
        from hadamard_maxdet.paley_seed import top_paley_seeds
    except Exception:
        return None
    try:
        seeds = top_paley_seeds(23, k=1)
    except Exception:
        return None
    if not seeds:
        return None
    _, _, M, _ = seeds[0]
    return [list(row) for row in M.entries]


def load_all_h23_basins(
    results_dir: str | os.PathLike[str] = "results",
    *,
    n: int = 23,
    include_constructed: bool = True,
) -> list[BasinEntry]:
    """Load every H23 ±1 matrix found in ``results_dir`` plus the three
    construction families (Williamson, KMS Prop 28, top Paley-deletion).

    Deduplicates by Gram class signature, keeping the highest-``|det|``
    representative for each signature. Returns a list sorted by descending
    ``|det|``.
    """
    rd = Path(results_dir)
    candidates: list[BasinEntry] = []

    # Pull every JSON in results/ matching H{n}_*.json that carries a matrix.
    if rd.is_dir():
        prefix = f"H{n}_"
        for fp in sorted(rd.glob(f"{prefix}*.json")):
            try:
                with open(fp) as f:
                    d = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            mat = _matrix_from_dict(d)
            if mat is None or len(mat) != n:
                continue
            try:
                det_actual = integer_determinant(mat)
            except Exception:
                continue
            if det_actual == 0:
                continue
            label = d.get("label") or fp.stem
            sig = class_signature(mat)
            candidates.append(
                BasinEntry(
                    label=label,
                    matrix=mat,
                    abs_det=abs(det_actual),
                    signature=sig,
                    source=str(fp.name),
                )
            )

    if include_constructed and n == 23:
        for label, source, builder in (
            ("williamson_gs_h23", "williamson.gs_array", _safe_williamson_h23),
            ("kms_prop28_q5", "kms_construction.prop28_matrix(5)", _safe_kms_h23),
            ("paley_deletion_top1", "paley_seed.top_paley_seeds(23,1)", _safe_top_paley_h23),
        ):
            mat = builder()
            if mat is None or len(mat) != n:
                continue
            try:
                det_actual = integer_determinant(mat)
            except Exception:
                continue
            if det_actual == 0:
                continue
            sig = class_signature(mat)
            candidates.append(
                BasinEntry(
                    label=label,
                    matrix=mat,
                    abs_det=abs(det_actual),
                    signature=sig,
                    source=source,
                )
            )

    # Dedup by signature, keep highest |det|.
    by_sig: dict[ClassSignature, BasinEntry] = {}
    for e in candidates:
        prev = by_sig.get(e.signature)
        if prev is None or e.abs_det > prev.abs_det:
            by_sig[e.signature] = e

    out = sorted(by_sig.values(), key=lambda e: -e.abs_det)
    return out


# ---------------------------------------------------------------------------
# Diagnostic 1: pairwise distances
# ---------------------------------------------------------------------------


def hamming_distance(A: Sequence[Sequence[int]], B: Sequence[Sequence[int]]) -> int:
    """Number of (r, c) where ``A[r][c] != B[r][c]``."""
    n = len(A)
    A_arr = np.asarray(A, dtype=np.int8)
    B_arr = np.asarray(B, dtype=np.int8)
    return int(np.sum(A_arr != B_arr))


def _gram_offdiag_multiset(M: Sequence[Sequence[int]]) -> Counter:
    n = len(M)
    arr = np.asarray(M, dtype=np.int64)
    G = arr @ arr.T
    cnt: Counter = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            cnt[int(G[i, j])] += 1
    return cnt


def gram_emd(c1: Counter, c2: Counter) -> float:
    """Earth-mover (1-Wasserstein) distance between two integer-valued count
    multisets of the same total mass.

    Both ``c1`` and ``c2`` are tallies over integers; total counts must
    match. EMD between two 1-D distributions equals the L1 distance of
    their cumulative distribution functions, which is the closed form used
    here.
    """
    n1 = sum(c1.values())
    n2 = sum(c2.values())
    if n1 != n2:
        raise ValueError(f"EMD requires equal mass; got {n1} vs {n2}")
    if n1 == 0:
        return 0.0
    keys = sorted(set(c1) | set(c2))
    cum1 = 0.0
    cum2 = 0.0
    total = 0.0
    for k in keys:
        cum1 += c1.get(k, 0) / n1
        cum2 += c2.get(k, 0) / n1
        # gap between adjacent keys (Earth-mover on integer support).
        # We use successive differences: the L1 over CDF approximated by
        # treating keys as a sorted lattice.
        total += abs(cum1 - cum2)
    return total


def spectral_distance(M_a: Sequence[Sequence[int]], M_b: Sequence[Sequence[int]]) -> float:
    """L2 norm between sorted singular value vectors of ``M_a`` and ``M_b``."""
    sa = np.linalg.svd(np.asarray(M_a, dtype=np.float64), compute_uv=False)
    sb = np.linalg.svd(np.asarray(M_b, dtype=np.float64), compute_uv=False)
    sa_sorted = np.sort(sa)
    sb_sorted = np.sort(sb)
    return float(np.linalg.norm(sa_sorted - sb_sorted))


def pairwise_distances(
    basins: Sequence[BasinEntry],
) -> dict[str, np.ndarray]:
    """Return three N×N distance matrices keyed by ``hamming``, ``gram_emd``,
    ``spectral``. All are symmetric with zeros on the diagonal.
    """
    N = len(basins)
    H = np.zeros((N, N), dtype=np.int64)
    G = np.zeros((N, N), dtype=np.float64)
    S = np.zeros((N, N), dtype=np.float64)

    multisets = [_gram_offdiag_multiset(b.matrix) for b in basins]
    svs = [
        np.sort(np.linalg.svd(np.asarray(b.matrix, dtype=np.float64), compute_uv=False))
        for b in basins
    ]

    for i in range(N):
        for j in range(i + 1, N):
            h = hamming_distance(basins[i].matrix, basins[j].matrix)
            H[i, j] = H[j, i] = h
            g = gram_emd(multisets[i], multisets[j])
            G[i, j] = G[j, i] = g
            s = float(np.linalg.norm(svs[i] - svs[j]))
            S[i, j] = S[j, i] = s
    return {"hamming": H, "gram_emd": G, "spectral": S}


# ---------------------------------------------------------------------------
# Diagnostic 2: MDS embedding
# ---------------------------------------------------------------------------


def mds_embed(distance_matrix: np.ndarray, k: int = 2) -> np.ndarray:
    """Classical (Torgerson) multidimensional scaling.

    Returns an ``N × k`` array of embedding coordinates. Falls back gracefully
    when the distance matrix is degenerate (returns the top-k principal
    components of a centred distance² matrix even when negative eigenvalues
    appear, by clamping to zero).
    """
    D = np.asarray(distance_matrix, dtype=np.float64)
    N = D.shape[0]
    if N == 0:
        return np.zeros((0, k))
    if N == 1:
        return np.zeros((1, k))
    D2 = D ** 2
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ D2 @ J
    # Symmetrise to absorb floating noise.
    B = 0.5 * (B + B.T)
    eigvals, eigvecs = np.linalg.eigh(B)
    # Largest k eigenvalues / vectors.
    order = np.argsort(eigvals)[::-1][:k]
    top_vals = np.maximum(eigvals[order], 0.0)
    top_vecs = eigvecs[:, order]
    coords = top_vecs * np.sqrt(top_vals)[None, :]
    return coords


# ---------------------------------------------------------------------------
# Diagnostic 3: saddle search between two basins
# ---------------------------------------------------------------------------


@dataclass
class SaddleStep:
    """One waypoint in the saddle search."""

    fraction_b: float
    abs_det_initial: int
    abs_det_relaxed: int
    n_relaxed_flips: int


@dataclass
class SaddleResult:
    label_a: str
    label_b: str
    hamming_a_b: int
    abs_det_a: int
    abs_det_b: int
    saddle_height: int
    n_intermediate: int
    sa_iters: int
    walltime_seconds: float
    path: list[SaddleStep]

    def to_json(self) -> dict:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "hamming_a_b": self.hamming_a_b,
            "abs_det_a": self.abs_det_a,
            "abs_det_b": self.abs_det_b,
            "saddle_height": self.saddle_height,
            "n_intermediate": self.n_intermediate,
            "sa_iters": self.sa_iters,
            "walltime_seconds": self.walltime_seconds,
            "path": [
                {
                    "fraction_b": s.fraction_b,
                    "abs_det_initial": s.abs_det_initial,
                    "abs_det_relaxed": s.abs_det_relaxed,
                    "n_relaxed_flips": s.n_relaxed_flips,
                }
                for s in self.path
            ],
        }


def _interpolate_anchored(
    M_a: list[list[int]], M_b: list[list[int]], frac_b: float, rng: random.Random
) -> tuple[list[list[int]], set[tuple[int, int]]]:
    """Build a waypoint between ``M_a`` and ``M_b`` with a fraction ``frac_b``
    of the *differing* entries forced to ``M_b``'s value (the rest forced to
    ``M_a``'s value). Return ``(matrix, anchored_positions)``.

    Anchored positions are exactly the differing positions: at every flipped
    position we have *fixed* it to one side or the other, so SA must not
    touch it. Positions where ``M_a == M_b`` are unconstrained (free).
    """
    n = len(M_a)
    diffs: list[tuple[int, int]] = [
        (i, j) for i in range(n) for j in range(n) if M_a[i][j] != M_b[i][j]
    ]
    rng.shuffle(diffs)
    k_b = int(round(frac_b * len(diffs)))
    to_b = set(diffs[:k_b])
    anchored = set(diffs)  # all differing positions are pinned at this waypoint
    M_w = [row[:] for row in M_a]
    for (i, j) in to_b:
        M_w[i][j] = M_b[i][j]
    return M_w, anchored


def _sa_constrained(
    M_init: list[list[int]],
    fixed_positions: set[tuple[int, int]],
    *,
    iterations: int,
    rng: random.Random,
    t_start: float = 0.5,
    t_end: float = 0.005,
    refresh_every: int = 1000,
) -> tuple[list[list[int]], int, int]:
    """Light SA on entries *not in* ``fixed_positions``, maximising ``|det|``.

    Returns ``(best_matrix, best_abs_det, n_accepted_flips)``.
    """
    n = len(M_init)
    state = IncrementalState.from_matrix(M_init)
    if state.det == 0:
        # try a quick refresh then bail with the initial matrix unchanged.
        refresh(state)
    if state.det == 0:
        return [row[:] for row in M_init], 0, 0
    free_positions = [
        (i, j) for i in range(n) for j in range(n) if (i, j) not in fixed_positions
    ]
    if not free_positions:
        return [row[:] for row in M_init], abs(state.det), 0

    best_det = abs(state.det)
    best_M = [row[:] for row in state.M]
    cur_e = -math.log(max(best_det, 1))
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0
    accepted = 0

    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        t = t_start * math.exp(log_ratio * frac)
        i, j = free_positions[rng.randrange(len(free_positions))]
        new_det = propose_entry_flip(state, i, j)
        if new_det == 0:
            continue
        prop_e = -math.log(abs(new_det))
        delta = prop_e - cur_e
        if delta > 0 and rng.random() >= math.exp(-delta / max(t, 1e-12)):
            continue
        apply_entry_flip(state, i, j)
        cur_e = prop_e
        accepted += 1
        if abs(state.det) > best_det:
            best_det = abs(state.det)
            best_M = [row[:] for row in state.M]
        if refresh_every > 0 and accepted % refresh_every == 0:
            refresh(state)
            if state.det == 0:
                # restart from best.
                state = IncrementalState.from_matrix(best_M)
                cur_e = -math.log(max(best_det, 1))
    return best_M, best_det, accepted


def saddle_search_between(
    M_a: list[list[int]],
    M_b: list[list[int]],
    *,
    n_intermediate: int = 15,
    sa_iters: int = 5000,
    seed: int = 0,
    label_a: str = "a",
    label_b: str = "b",
) -> SaddleResult:
    """Linearly interpolate between two ±1 matrices, then SA-relax each
    waypoint with the differing-entry assignments held fixed. Reports the
    minimum ``|det|`` along the relaxed path = saddle height.

    The endpoints are *also* relaxed (with frac=0 and frac=1, all differing
    positions pinned to one side), so they'd return their original |det|
    if they were local maxima — useful sanity check.
    """
    if len(M_a) != len(M_b):
        raise ValueError("matrices must have the same order")
    n = len(M_a)
    rng = random.Random(seed)
    h = hamming_distance(M_a, M_b)
    abs_det_a = abs(integer_determinant(M_a))
    abs_det_b = abs(integer_determinant(M_b))
    fractions = np.linspace(0.0, 1.0, n_intermediate + 2)  # endpoints + interior
    path: list[SaddleStep] = []
    t0 = time.monotonic()
    for f in fractions:
        wp_rng = random.Random(seed + int(f * 10_000))
        wp_init, anchored = _interpolate_anchored(M_a, M_b, float(f), wp_rng)
        det0 = abs(integer_determinant(wp_init))
        # SA-relax the FREE positions (those equal in M_a and M_b).
        M_relax, det_relaxed, accepted = _sa_constrained(
            wp_init,
            anchored,
            iterations=sa_iters,
            rng=random.Random(seed + 977 * int(f * 1000) + 1),
        )
        path.append(
            SaddleStep(
                fraction_b=float(f),
                abs_det_initial=det0,
                abs_det_relaxed=det_relaxed,
                n_relaxed_flips=accepted,
            )
        )
    walltime = time.monotonic() - t0
    saddle = min(s.abs_det_relaxed for s in path[1:-1]) if len(path) > 2 else min(
        s.abs_det_relaxed for s in path
    )
    return SaddleResult(
        label_a=label_a,
        label_b=label_b,
        hamming_a_b=h,
        abs_det_a=abs_det_a,
        abs_det_b=abs_det_b,
        saddle_height=saddle,
        n_intermediate=n_intermediate,
        sa_iters=sa_iters,
        walltime_seconds=walltime,
        path=path,
    )


# ---------------------------------------------------------------------------
# Diagnostic 4: Ehlich-partition coverage
# ---------------------------------------------------------------------------


@dataclass
class PartitionAssignment:
    """Best fit of a matrix's rows into a target 4-block Ehlich partition."""

    sizes: tuple[int, int, int, int]
    intra_value: int
    cross_value: int
    assignment: list[int]  # row -> block index 0..3
    gram_match_score: float  # fraction of off-diagonal entries matching target
    intra_match_count: int
    intra_total: int
    cross_match_count: int
    cross_total: int

    def to_json(self) -> dict:
        return {
            "sizes": list(self.sizes),
            "intra_value": self.intra_value,
            "cross_value": self.cross_value,
            "assignment": self.assignment,
            "gram_match_score": self.gram_match_score,
            "intra_match_count": self.intra_match_count,
            "intra_total": self.intra_total,
            "cross_match_count": self.cross_match_count,
            "cross_total": self.cross_total,
        }


def _greedy_block_assignment(
    G_off: np.ndarray,
    sizes: tuple[int, int, int, int],
    intra: int,
    cross: int,
    *,
    rng: random.Random,
    n_restarts: int = 8,
    swap_passes: int = 2,
) -> tuple[list[int], int, int, int, int]:
    """Greedy + local-search assignment of rows to 4 blocks of given sizes.

    Maximises the number of off-diagonal Gram entries that match the target
    pattern (intra-block = ``intra``, cross-block = ``cross``).

    Returns ``(assignment, intra_match, intra_total, cross_match, cross_total)``.
    """
    n = G_off.shape[0]
    best_score = -1
    best_state: tuple[list[int], int, int, int, int] | None = None
    target_sizes = list(sizes)

    def evaluate(assign: list[int]) -> tuple[int, int, int, int]:
        intra_match = 0
        intra_total = 0
        cross_match = 0
        cross_total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if assign[i] == assign[j]:
                    intra_total += 1
                    if G_off[i, j] == intra:
                        intra_match += 1
                else:
                    cross_total += 1
                    if G_off[i, j] == cross:
                        cross_match += 1
        return intra_match, intra_total, cross_match, cross_total

    for restart in range(n_restarts):
        # Initial random assignment with the right block sizes.
        rows = list(range(n))
        rng.shuffle(rows)
        assign = [0] * n
        idx = 0
        for k, sz in enumerate(target_sizes):
            for _ in range(sz):
                assign[rows[idx]] = k
                idx += 1
        # Local search: pairwise row-swap if it improves (matches).
        for _ in range(swap_passes):
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if assign[i] == assign[j]:
                        continue
                    # Swap; recompute deltas locally only for rows i and j.
                    bi_old, bj_old = assign[i], assign[j]
                    score_before = _local_score(G_off, assign, i, intra, cross) + \
                        _local_score(G_off, assign, j, intra, cross)
                    assign[i], assign[j] = bj_old, bi_old
                    score_after = _local_score(G_off, assign, i, intra, cross) + \
                        _local_score(G_off, assign, j, intra, cross)
                    if score_after > score_before:
                        improved = True
                    else:
                        assign[i], assign[j] = bi_old, bj_old
            if not improved:
                break
        intra_match, intra_total, cross_match, cross_total = evaluate(assign)
        score = intra_match + cross_match
        if score > best_score:
            best_score = score
            best_state = (list(assign), intra_match, intra_total, cross_match, cross_total)

    assert best_state is not None
    return best_state


def _local_score(
    G_off: np.ndarray, assign: list[int], i: int, intra: int, cross: int
) -> int:
    """Number of pairs (i, k) in G_off whose value matches the target given
    ``assign``. Used as a delta-evaluator during local search.
    """
    n = G_off.shape[0]
    s = 0
    bi = assign[i]
    for k in range(n):
        if k == i:
            continue
        v = int(G_off[i, k])
        if assign[k] == bi:
            if v == intra:
                s += 1
        else:
            if v == cross:
                s += 1
    return s


def partition_assignment(
    M: Sequence[Sequence[int]],
    partition_sizes: tuple[int, int, int, int],
    *,
    intra_value: int = 3,
    cross_value: int = -1,
    rng: random.Random | None = None,
    n_restarts: int = 8,
) -> PartitionAssignment:
    """Best assignment of M's rows to a 4-block Ehlich partition with the
    given sizes. Reports the fraction of off-diagonal Gram entries that match
    the target intra/cross pattern.

    A score of 1.0 means M *already* realises the partition exactly. Lower
    scores measure the structural distance to the target block-Gram.
    """
    if sum(partition_sizes) != len(M):
        raise ValueError(
            f"partition sizes {partition_sizes} sum to "
            f"{sum(partition_sizes)} != n = {len(M)}"
        )
    if rng is None:
        rng = random.Random(0)
    arr = np.asarray(M, dtype=np.int64)
    G = arr @ arr.T
    assignment, intra_match, intra_total, cross_match, cross_total = (
        _greedy_block_assignment(
            G, tuple(partition_sizes), intra_value, cross_value,
            rng=rng, n_restarts=n_restarts,
        )
    )
    total_off = intra_total + cross_total
    score = (intra_match + cross_match) / total_off if total_off else 0.0
    return PartitionAssignment(
        sizes=tuple(partition_sizes),
        intra_value=intra_value,
        cross_value=cross_value,
        assignment=assignment,
        gram_match_score=score,
        intra_match_count=intra_match,
        intra_total=intra_total,
        cross_match_count=cross_match,
        cross_total=cross_total,
    )


# ---------------------------------------------------------------------------
# Diagnostic 6: bounded random-walk reachability
# ---------------------------------------------------------------------------


def bounded_walk_reachability(
    basins: Sequence[BasinEntry],
    *,
    walk_budget: int = 1000,
    det_floor_fraction: float = 0.5,
    seed: int = 0,
) -> dict:
    """From each basin, run a 2-flip random walk and track which OTHER basins'
    Gram class signatures the walk encounters.

    A 2-flip step proposes flipping two entries; the move is accepted iff the
    resulting matrix has ``|det| >= det_floor_fraction * basin.abs_det``
    (not the global anchor — each walk uses its own seed basin's |det| as
    the floor).

    Returns ``{"edges": [{"from": i, "to": j, "first_step": k}], "stats": ...}``.
    """
    rng = random.Random(seed)
    sig_to_idx = {b.signature: i for i, b in enumerate(basins)}
    edges: list[dict] = []
    stats: list[dict] = []
    for i, b in enumerate(basins):
        floor = int(det_floor_fraction * b.abs_det)
        state = IncrementalState.from_matrix([row[:] for row in b.matrix])
        if state.det == 0:
            stats.append({"basin": i, "label": b.label, "reached": [], "floor": floor})
            continue
        n = b.n
        seen: dict[int, int] = {}  # other-basin-idx -> first-step it was reached
        cur_det = abs(state.det)
        accepted = 0
        for step in range(walk_budget):
            i1 = rng.randrange(n); j1 = rng.randrange(n)
            i2 = rng.randrange(n); j2 = rng.randrange(n)
            if (i1, j1) == (i2, j2):
                continue
            # Apply both flips, check det, possibly revert.
            d_after_first = propose_entry_flip(state, i1, j1)
            if d_after_first == 0:
                continue
            apply_entry_flip(state, i1, j1)
            d_after_second = propose_entry_flip(state, i2, j2)
            if d_after_second == 0 or abs(d_after_second) < floor:
                # Revert i1.
                # Use refresh for safety to keep adjugate consistent.
                # (Singular case for either step would otherwise corrupt state.)
                state.M[i1][j1] = -state.M[i1][j1]
                refresh(state)
                continue
            apply_entry_flip(state, i2, j2)
            cur_det = abs(state.det)
            accepted += 1
            sig = class_signature(state.M)
            if sig in sig_to_idx:
                idx = sig_to_idx[sig]
                if idx != i and idx not in seen:
                    seen[idx] = step
                    edges.append({"from": i, "to": idx, "first_step": step})
            # Periodic refresh to keep adjugate stable.
            if accepted % 50 == 0:
                refresh(state)
                if state.det == 0:
                    break
        stats.append(
            {
                "basin": i,
                "label": b.label,
                "abs_det": b.abs_det,
                "floor": floor,
                "accepted_2flips": accepted,
                "n_other_basins_reached": len(seen),
                "reached_basin_indices": sorted(seen.keys()),
            }
        )
    return {"edges": edges, "stats": stats}


# ---------------------------------------------------------------------------
# Diagnostic 5 helper: distance-from-anchor profile (pure utility)
# ---------------------------------------------------------------------------


def distance_from_anchor_profile(
    basins: Sequence[BasinEntry], anchor_idx: int = 0, n_for_bound: int | None = None,
) -> list[dict]:
    """For each basin (other than the anchor), return ``(label, hamming, |det|/Barba)``."""
    if not basins:
        return []
    anchor = basins[anchor_idx]
    n = anchor.n
    bound = upper_bound(n_for_bound or n)
    out: list[dict] = []
    for i, b in enumerate(basins):
        if i == anchor_idx:
            continue
        h = hamming_distance(anchor.matrix, b.matrix)
        out.append(
            {
                "label": b.label,
                "abs_det": b.abs_det,
                "ratio_to_bound": b.abs_det / bound,
                "hamming_to_anchor": h,
            }
        )
    return out


__all__ = [
    "BasinEntry",
    "load_all_h23_basins",
    "hamming_distance",
    "gram_emd",
    "spectral_distance",
    "pairwise_distances",
    "mds_embed",
    "SaddleStep",
    "SaddleResult",
    "saddle_search_between",
    "PartitionAssignment",
    "partition_assignment",
    "bounded_walk_reachability",
    "distance_from_anchor_profile",
]
