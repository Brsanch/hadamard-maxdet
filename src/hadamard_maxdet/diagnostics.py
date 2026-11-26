"""Diagnostic deep-dive instrumentation for the Hadamard maxdet search.

Purpose: characterise the search problem (basin structure, local
ruggedness, equivalence-class orbits, move effectiveness, cross-n
calibration) without trying to improve the H23 best.

Five diagnostics, all deterministic given a seed, all using the exact
integer Bareiss determinant for any reported |det|.

  * :func:`basin_map` — many-seed SA at small n, cluster final matrices
    by Gram + singular-value fingerprint.
  * :func:`energy_profile` — Hamming-1/2/3 neighbour |det| histogram
    around a given matrix (e.g. the H23 BTC seed).
  * :func:`orbit_size_estimate` — Monte-Carlo birthday estimate of the
    Hadamard-equivalence class size of a matrix under
    ±row × ±col × row-perm × col-perm.
  * :func:`move_profile` — per-temperature accepted-move profiling on
    a single SA run. Reports useful work rate and the wasted-tail length.
  * :func:`cross_n_validation` — calibration sweep against proven maxima.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from hadamard_maxdet.incremental import (
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    sa_incremental,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.orders import BENCHMARKS
from hadamard_maxdet.telemetry import gram_offdiag_report
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Diagnostic 1: basin map
# ---------------------------------------------------------------------------


def _fingerprint(entries: Sequence[Sequence[int]]) -> tuple:
    """Return a hashable fingerprint of ``entries`` invariant under nothing
    except small numerical noise in the SVD (rounded to 6 decimal places).

    Uses Gram off-diagonal multiset + sorted singular values rounded to
    6 dp. Two matrices with the same Gram off-diagonal multiset and the
    same SV spectrum are *highly* likely to be Hadamard-equivalent (this is
    not a proof — full Hadamard equivalence requires nauty-style isomorph
    checking on the bipartite double cover — but it's enough to cluster
    SA outcomes for the purpose of basin counting).
    """
    n = len(entries)
    M = np.array(entries, dtype=np.int64)
    G = M @ M.T
    off = []
    for i in range(n):
        for j in range(i + 1, n):
            off.append(int(G[i, j]))
    off_sorted = tuple(sorted(off))
    sv = np.linalg.svd(M.astype(float), compute_uv=False)
    sv_rounded = tuple(round(float(s), 6) for s in sorted(sv, reverse=True))
    return (off_sorted, sv_rounded)


@dataclass
class BasinCluster:
    """A cluster of seeds whose final matrices share a fingerprint."""

    abs_det: int
    fingerprint_key: str
    sv_multiplicity_pattern: list[tuple[float, int]]
    gram_offdiag_support: list[int]
    cluster_size: int
    representative_seeds: list[int]

    def to_json(self) -> dict:
        return {
            "abs_det": self.abs_det,
            "fingerprint_key": self.fingerprint_key,
            "sv_multiplicity_pattern": self.sv_multiplicity_pattern,
            "gram_offdiag_support": self.gram_offdiag_support,
            "cluster_size": self.cluster_size,
            "representative_seeds": self.representative_seeds[:5],
        }


@dataclass
class BasinMapResult:
    n: int
    n_seeds: int
    iters_per_seed: int
    proven_max: int | None
    clusters: list[BasinCluster]
    walltime_seconds: float

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "n_seeds": self.n_seeds,
            "iters_per_seed": self.iters_per_seed,
            "proven_max": self.proven_max,
            "n_clusters": len(self.clusters),
            "largest_cluster_size": (
                self.clusters[0].cluster_size if self.clusters else 0
            ),
            "largest_cluster_fraction": (
                self.clusters[0].cluster_size / self.n_seeds if self.clusters else 0.0
            ),
            "best_abs_det": (
                max(c.abs_det for c in self.clusters) if self.clusters else 0
            ),
            "proven_max_cluster_size": _proven_max_cluster_size(self),
            "clusters": [c.to_json() for c in self.clusters],
            "walltime_seconds": self.walltime_seconds,
        }


def _proven_max_cluster_size(result: BasinMapResult) -> int:
    if result.proven_max is None:
        return 0
    for c in result.clusters:
        if c.abs_det == result.proven_max:
            return c.cluster_size
    return 0


def basin_map(
    n: int,
    *,
    n_seeds: int,
    iters_per_seed: int = 30_000,
    t_start: float = 1.0,
    t_end: float = 0.005,
    base_seed: int = 0,
) -> BasinMapResult:
    """Run sa_incremental from ``n_seeds`` random starts and cluster outcomes.

    Each seed runs an independent SA chain. Final matrices are clustered
    by ``_fingerprint``. Clusters are sorted by descending size.
    """
    start = time.monotonic()
    bucket: dict[tuple, list[int]] = {}
    bucket_meta: dict[tuple, tuple[int, list[tuple[float, int]], list[int]]] = {}

    for s in range(n_seeds):
        seed = base_seed + s
        result = sa_incremental(
            n,
            iterations=iters_per_seed,
            t_start=t_start,
            t_end=t_end,
            seed=seed,
            refresh_every=0,
        )
        # Use the *best-ever* matrix (already integer-Bareiss verified).
        entries = result.matrix.entries
        abs_det = result.abs_det
        fp = _fingerprint(entries)
        if fp not in bucket:
            bucket[fp] = []
            # Record meta once per cluster.
            rep = gram_offdiag_report(entries)
            bucket_meta[fp] = (
                abs_det,
                [(float(s_), int(c_)) for s_, c_ in rep["sv_multiplicities"]],
                list(rep["gram_offdiag_support"]),
            )
        bucket[fp].append(seed)

    clusters: list[BasinCluster] = []
    for fp, seeds in bucket.items():
        abs_det, sv_mult, off_supp = bucket_meta[fp]
        key = hashlib.md5(repr(fp).encode("utf-8")).hexdigest()[:12]
        clusters.append(
            BasinCluster(
                abs_det=abs_det,
                fingerprint_key=key,
                sv_multiplicity_pattern=sv_mult,
                gram_offdiag_support=off_supp,
                cluster_size=len(seeds),
                representative_seeds=sorted(seeds)[:10],
            )
        )
    clusters.sort(key=lambda c: (-c.cluster_size, -c.abs_det))

    proven = None
    bid = f"H{n}"
    if bid in BENCHMARKS and BENCHMARKS[bid].best_known is not None:
        proven = BENCHMARKS[bid].best_known

    return BasinMapResult(
        n=n,
        n_seeds=n_seeds,
        iters_per_seed=iters_per_seed,
        proven_max=proven,
        clusters=clusters,
        walltime_seconds=time.monotonic() - start,
    )


# ---------------------------------------------------------------------------
# Diagnostic 2: energy profile
# ---------------------------------------------------------------------------


@dataclass
class EnergyProfileBin:
    distance: int
    n_samples: int
    n_above: int
    n_equal: int
    n_below: int
    n_singular: int
    max_abs_det: int
    min_abs_det: int
    mean_abs_det: float
    std_abs_det: float
    top_neighbours: list[dict]  # for distance 1 only; (i, j) flips and |det|


@dataclass
class EnergyProfileResult:
    n: int
    seed_abs_det: int
    bins: list[EnergyProfileBin]

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "seed_abs_det": self.seed_abs_det,
            "bins": [
                {
                    "distance": b.distance,
                    "n_samples": b.n_samples,
                    "n_above": b.n_above,
                    "n_equal": b.n_equal,
                    "n_below": b.n_below,
                    "n_singular": b.n_singular,
                    "max_abs_det": b.max_abs_det,
                    "min_abs_det": b.min_abs_det,
                    "mean_abs_det": b.mean_abs_det,
                    "std_abs_det": b.std_abs_det,
                    "top_neighbours": b.top_neighbours,
                }
                for b in self.bins
            ],
        }


def _abs_det(entries: Sequence[Sequence[int]]) -> int:
    return abs(integer_determinant([list(r) for r in entries]))


def _flip_in_place(entries: list[list[int]], flips: list[tuple[int, int]]) -> None:
    for i, j in flips:
        entries[i][j] = -entries[i][j]


def energy_profile(
    matrix: Sequence[Sequence[int]],
    *,
    distances: tuple[int, ...] = (1, 2, 3),
    n_samples_per_distance: int = 5_000,
    seed: int = 0,
    top_k_d1: int = 5,
) -> EnergyProfileResult:
    """Histogram |det| of Hamming-d neighbours of ``matrix`` for d in distances.

    For d=1 we enumerate all n^2 single-flip neighbours. For d>=2 we
    sample ``n_samples_per_distance`` random distinct-flip combinations.
    All |det| values are exact-integer.
    """
    rng = random.Random(seed)
    base = [list(r) for r in matrix]
    n = len(base)
    seed_det = _abs_det(base)

    # Use incremental adjugate for d=1 fast scoring.
    state = IncrementalState.from_matrix(base)

    bins: list[EnergyProfileBin] = []

    for d in distances:
        if d == 1:
            samples_dets: list[int] = []
            top: list[tuple[int, tuple[int, int]]] = []  # (|det|, (i,j))
            for i in range(n):
                for j in range(n):
                    new_det = propose_entry_flip(state, i, j)
                    ad = abs(new_det)
                    samples_dets.append(ad)
                    top.append((ad, (i, j)))
            top.sort(reverse=True)
            top_neighbours = [
                {"i": int(p[0]), "j": int(p[1]), "abs_det": int(ad)}
                for ad, p in top[:top_k_d1]
            ]
        else:
            samples_dets = []
            top_neighbours = []
            n_total = n * n
            tried: set[tuple[int, ...]] = set()
            attempts = 0
            max_attempts = n_samples_per_distance * 4
            while len(samples_dets) < n_samples_per_distance and attempts < max_attempts:
                attempts += 1
                # Choose d distinct positions.
                idx = rng.sample(range(n_total), d)
                key = tuple(sorted(idx))
                if key in tried:
                    continue
                tried.add(key)
                flips = [(p // n, p % n) for p in idx]
                m = [row[:] for row in base]
                _flip_in_place(m, flips)
                ad = abs(integer_determinant(m))
                samples_dets.append(ad)

        n_above = sum(1 for x in samples_dets if x > seed_det)
        n_equal = sum(1 for x in samples_dets if x == seed_det)
        n_below = sum(1 for x in samples_dets if 0 < x < seed_det)
        n_singular = sum(1 for x in samples_dets if x == 0)
        arr = np.array(samples_dets, dtype=np.float64)
        bins.append(
            EnergyProfileBin(
                distance=d,
                n_samples=len(samples_dets),
                n_above=n_above,
                n_equal=n_equal,
                n_below=n_below,
                n_singular=n_singular,
                max_abs_det=int(max(samples_dets)) if samples_dets else 0,
                min_abs_det=int(min(samples_dets)) if samples_dets else 0,
                mean_abs_det=float(arr.mean()) if arr.size else 0.0,
                std_abs_det=float(arr.std()) if arr.size else 0.0,
                top_neighbours=top_neighbours,
            )
        )

    return EnergyProfileResult(n=n, seed_abs_det=seed_det, bins=bins)


# ---------------------------------------------------------------------------
# Diagnostic 3: equivalence-class orbit size estimate
# ---------------------------------------------------------------------------


@dataclass
class OrbitSizeResult:
    n: int
    n_samples: int
    n_unique: int
    n_collisions: int
    estimated_orbit_size: float
    lower_bound_95: float
    interpretation: str

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "n_samples": self.n_samples,
            "n_unique": self.n_unique,
            "n_collisions": self.n_collisions,
            "estimated_orbit_size": self.estimated_orbit_size,
            "lower_bound_95": self.lower_bound_95,
            "interpretation": self.interpretation,
        }


def _canonical_hash(entries: list[list[int]]) -> bytes:
    """Lex-sort rows, then lex-sort columns, then md5 the byte string.

    Not a true Hadamard canonical form — does not absorb sign equivalence.
    But: if we also (i) absorb sign by negating any row whose first entry
    is -1 and any column whose first entry is -1, we get a unique
    representative for matrices that only differ by row/col permutations
    and sign flips (the orbit we sample from).
    """
    n = len(entries)
    M = np.array(entries, dtype=np.int8)
    # Negate rows whose first entry is -1.
    for i in range(n):
        if M[i, 0] == -1:
            M[i, :] = -M[i, :]
    # Sort rows lex.
    row_order = sorted(range(n), key=lambda r: tuple(M[r].tolist()))
    M = M[row_order, :]
    # Negate cols whose first entry (after row sort) is -1.
    for j in range(n):
        if M[0, j] == -1:
            M[:, j] = -M[:, j]
    # Sort cols lex.
    col_order = sorted(range(n), key=lambda c: tuple(M[:, c].tolist()))
    M = M[:, col_order]
    return hashlib.sha256(M.tobytes()).digest()


def orbit_size_estimate(
    matrix: Sequence[Sequence[int]],
    *,
    n_samples: int = 10_000,
    seed: int = 0,
) -> OrbitSizeResult:
    """Birthday-style estimate of the orbit of ``matrix`` under
    ±row × ±col × row-perm × col-perm.

    Sample ``n_samples`` random transformations and hash to a canonical
    form. If we see ``K`` unique hashes in ``N`` samples with ``C``
    collisions, the orbit size estimate is ``N(N-1) / (2 C)`` (birthday
    formula). If ``C == 0`` (no collisions) the orbit size is at least
    on the order of ``N^2`` with high probability.
    """
    rng = random.Random(seed)
    n = len(matrix)
    base = np.array(matrix, dtype=np.int8)

    seen: set[bytes] = set()
    collisions = 0

    for _ in range(n_samples):
        M = base.copy()
        # Random row signs.
        row_signs = np.array(
            [1 if rng.random() < 0.5 else -1 for _ in range(n)], dtype=np.int8
        )
        col_signs = np.array(
            [1 if rng.random() < 0.5 else -1 for _ in range(n)], dtype=np.int8
        )
        M = (M * row_signs[:, None]) * col_signs[None, :]
        # Random row perm.
        rp = list(range(n))
        rng.shuffle(rp)
        M = M[rp, :]
        cp = list(range(n))
        rng.shuffle(cp)
        M = M[:, cp]
        h = _canonical_hash(M.tolist())
        if h in seen:
            collisions += 1
        else:
            seen.add(h)

    n_unique = len(seen)
    if collisions == 0:
        # Lower bound: if orbit had size <= K, expected collisions in
        # n_samples draws is ~ n_samples^2 / (2K). Setting expected
        # collisions = 3 (Poisson 95% upper for "saw 0") gives
        # K >= n_samples^2 / 6.
        est = float("inf")
        lb = max(n_unique, n_samples * n_samples / 6.0)
        interp = (
            f"No collisions in {n_samples} samples → orbit > "
            f"~{int(lb):,} (95% CI lower bound)."
        )
    else:
        est = n_samples * (n_samples - 1) / (2.0 * collisions)
        # Crude 95% Poisson CI for number of collisions: c ± 2 sqrt(c)
        c_high = collisions + 2.0 * math.sqrt(collisions)
        lb = n_samples * (n_samples - 1) / (2.0 * c_high)
        interp = (
            f"{collisions} collisions in {n_samples} samples; "
            f"orbit size estimate {est:,.0f} (lower 95% ≈ {lb:,.0f})."
        )

    return OrbitSizeResult(
        n=n,
        n_samples=n_samples,
        n_unique=n_unique,
        n_collisions=collisions,
        estimated_orbit_size=est,
        lower_bound_95=lb,
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# Diagnostic 4: move-by-move profiling
# ---------------------------------------------------------------------------


@dataclass
class MoveProfileResult:
    n: int
    iterations: int
    accepted: int
    decile_stats: list[dict]  # per temperature decile
    n_increase: int
    n_decrease: int
    n_equal: int
    first_best_step: int
    first_best_temperature: float
    best_abs_det: int
    wasted_tail_steps: int
    walltime_seconds: float

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "iterations": self.iterations,
            "accepted": self.accepted,
            "n_increase_accepted": self.n_increase,
            "n_decrease_accepted": self.n_decrease,
            "n_equal_accepted": self.n_equal,
            "first_best_step": self.first_best_step,
            "first_best_temperature": self.first_best_temperature,
            "best_abs_det": self.best_abs_det,
            "wasted_tail_steps": self.wasted_tail_steps,
            "decile_stats": self.decile_stats,
            "walltime_seconds": self.walltime_seconds,
        }


def move_profile(
    n: int,
    *,
    iterations: int = 200_000,
    t_start: float = 1.0,
    t_end: float = 0.005,
    seed: int = 0,
) -> MoveProfileResult:
    """Profile every accepted move during a single sa_incremental run.

    Records Δ log|det| per accepted move bucketed into 10 temperature
    deciles. Returns useful-work statistics.
    """
    rng = random.Random(seed)
    mat = Matrix.random(n, rng)
    state = IncrementalState.from_matrix(mat.entries)
    tries = 0
    while state.det == 0 and tries < 16:
        mat = Matrix.random(n, rng)
        state = IncrementalState.from_matrix(mat.entries)
        tries += 1
    if state.det == 0:
        raise RuntimeError("could not find non-singular start in 16 tries")

    current_e = -math.log(abs(state.det))
    best_entries = [row[:] for row in state.M]
    best_e = current_e
    best_abs_det = abs(state.det)
    first_best_step = 0
    first_best_temperature = t_start
    log_ratio = math.log(t_end / t_start)

    # Per-decile accumulators.
    decile_dlog: list[list[float]] = [[] for _ in range(10)]
    decile_nprop: list[int] = [0] * 10
    decile_naccept: list[int] = [0] * 10

    n_increase = 0
    n_decrease = 0
    n_equal = 0
    accepted = 0

    start = time.monotonic()
    for step in range(iterations):
        frac = step / max(1, iterations - 1)
        decile = min(9, int(frac * 10))
        t = t_start * math.exp(log_ratio * frac)
        decile_nprop[decile] += 1

        i = rng.randrange(n)
        j = rng.randrange(n)
        new_det = propose_entry_flip(state, i, j)
        if new_det == 0:
            continue
        proposal_e = -math.log(abs(new_det))
        delta = proposal_e - current_e
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            apply_entry_flip(state, i, j)
            d_log = -delta  # Δ log|det| = -Δ E
            decile_dlog[decile].append(d_log)
            decile_naccept[decile] += 1
            accepted += 1
            if d_log > 0:
                n_increase += 1
            elif d_log < 0:
                n_decrease += 1
            else:
                n_equal += 1
            current_e = proposal_e
            if current_e < best_e:
                best_e = current_e
                best_entries = [row[:] for row in state.M]
                best_abs_det = abs(state.det)
                first_best_step = step
                first_best_temperature = t
    walltime = time.monotonic() - start

    # Re-verify the best matrix exactly.
    best_abs_det = abs(integer_determinant(best_entries))

    decile_stats = []
    for d in range(10):
        arr = np.array(decile_dlog[d], dtype=np.float64)
        positives = arr[arr > 0]
        decile_stats.append(
            {
                "decile": d,
                "temperature_low": t_start * math.exp(log_ratio * (d / 10.0)),
                "temperature_high": t_start * math.exp(log_ratio * ((d + 1) / 10.0)),
                "n_proposed": decile_nprop[d],
                "n_accepted": decile_naccept[d],
                "accept_rate": (
                    decile_naccept[d] / decile_nprop[d] if decile_nprop[d] else 0.0
                ),
                "n_useful": int(positives.size),
                "useful_fraction": float(positives.size) / max(1, decile_naccept[d]),
                "mean_useful_dlogdet": (
                    float(positives.mean()) if positives.size else 0.0
                ),
                "mean_dlogdet": float(arr.mean()) if arr.size else 0.0,
            }
        )

    wasted = iterations - first_best_step

    return MoveProfileResult(
        n=n,
        iterations=iterations,
        accepted=accepted,
        decile_stats=decile_stats,
        n_increase=n_increase,
        n_decrease=n_decrease,
        n_equal=n_equal,
        first_best_step=first_best_step,
        first_best_temperature=first_best_temperature,
        best_abs_det=best_abs_det,
        wasted_tail_steps=wasted,
        walltime_seconds=walltime,
    )


# ---------------------------------------------------------------------------
# Diagnostic 5: cross-n calibration
# ---------------------------------------------------------------------------


@dataclass
class CrossNRow:
    n: int
    proven_max: int | None
    sa_best: int
    sa_walltime: float
    structured_best: int | None
    structured_walltime: float | None
    overall_best: int
    overall_ratio: float | None
    method: str

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "proven_max": self.proven_max,
            "sa_best": self.sa_best,
            "sa_walltime": self.sa_walltime,
            "structured_best": self.structured_best,
            "structured_walltime": self.structured_walltime,
            "overall_best": self.overall_best,
            "overall_ratio": self.overall_ratio,
            "method": self.method,
        }


def _run_sa_multistart(
    n: int,
    *,
    n_starts: int,
    iterations: int,
    base_seed: int = 0,
) -> tuple[int, float, list[list[int]]]:
    start = time.monotonic()
    best_det = 0
    best_entries: list[list[int]] = []
    for s in range(n_starts):
        r = sa_incremental(
            n,
            iterations=iterations,
            t_start=1.0,
            t_end=0.005,
            seed=base_seed + s,
            refresh_every=0,
        )
        if r.abs_det > best_det:
            best_det = r.abs_det
            best_entries = r.matrix.entries
    return best_det, time.monotonic() - start, best_entries


def cross_n_validation(
    orders_to_test: Sequence[int] = (11, 13, 15, 17, 19, 21),
    *,
    n_starts: int = 8,
    iterations: int = 100_000,
    structured_seconds: float = 60.0,
    base_seed: int = 0,
    use_structured: bool = True,
    iteration_caps: dict[int, int] | None = None,
) -> list[CrossNRow]:
    """Run SA multi-start + optional bordered_two_circulant_sa per order,
    report best vs proven max.

    ``iteration_caps`` lets the caller throttle higher orders.
    """
    rows: list[CrossNRow] = []
    iteration_caps = iteration_caps or {}

    # Local import to avoid a hard cycle if rich_ansatz changes.
    try:
        from hadamard_maxdet.rich_ansatz import sa_bordered_two_circulant
    except Exception:
        sa_bordered_two_circulant = None  # type: ignore

    for n in orders_to_test:
        iters = iteration_caps.get(n, iterations)
        sa_best, sa_wall, _entries = _run_sa_multistart(
            n,
            n_starts=n_starts,
            iterations=iters,
            base_seed=base_seed + n * 1009,
        )

        struct_best: int | None = None
        struct_wall: float | None = None
        method = "sa_incremental_multistart"
        if use_structured and sa_bordered_two_circulant is not None and n % 2 == 1:
            t0 = time.monotonic()
            best_struct = 0
            chunks = 0
            while time.monotonic() - t0 < structured_seconds:
                r = sa_bordered_two_circulant(
                    n,
                    iterations=10_000,
                    t_start=1.0,
                    t_end=0.005,
                    seed=base_seed + n * 7919 + chunks,
                )
                if r.abs_det > best_struct:
                    best_struct = r.abs_det
                chunks += 1
            struct_wall = time.monotonic() - t0
            struct_best = best_struct
            if struct_best > sa_best:
                method = "bordered_two_circulant_sa"

        overall = max(sa_best, struct_best or 0)
        bid = f"H{n}"
        proven = (
            BENCHMARKS[bid].best_known
            if (bid in BENCHMARKS and BENCHMARKS[bid].best_known is not None)
            else None
        )
        ratio = (overall / proven) if (proven and proven > 0) else None
        rows.append(
            CrossNRow(
                n=n,
                proven_max=proven,
                sa_best=sa_best,
                sa_walltime=sa_wall,
                structured_best=struct_best,
                structured_walltime=struct_wall,
                overall_best=overall,
                overall_ratio=ratio,
                method=method,
            )
        )
    return rows


__all__ = [
    "BasinCluster",
    "BasinMapResult",
    "basin_map",
    "EnergyProfileBin",
    "EnergyProfileResult",
    "energy_profile",
    "OrbitSizeResult",
    "orbit_size_estimate",
    "MoveProfileResult",
    "move_profile",
    "CrossNRow",
    "cross_n_validation",
]
