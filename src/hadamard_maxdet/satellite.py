"""Satellite view of the Hadamard maxdet landscape at order n.

Where :mod:`orrick_hunt` enumerates the *top* Ehlich-block partitions for SA
seeding and :mod:`basin_atlas` reports diagnostics for the basins we have
*reached*, this module zooms out to the **full theoretical landscape**: every
4-part partition of n, ordered by its closed-form Ehlich/Wojtas predicted
upper bound on |det|, with our concrete basins overlaid.

Three deliverables, each a thin wrapper over existing primitives:

1. :func:`enumerate_all_ehlich_partitions` — full enumeration of ordered
   4-partitions ``(s_1, s_2, s_3, s_4)`` of ``n`` with ``s_1 ≤ s_2 ≤ s_3 ≤ s_4``
   (deduplicated by multiset) and the closed-form predicted ``|det|`` for
   the canonical 4-valued ansatz (intra-block Gram = ``+3``, cross-block
   Gram = ``-1``). For n ≡ 3 (mod 4) this matches the conjectured high-|det|
   block geometry; the predicted bound is ``sqrt(det(G_block))``.

2. :func:`overlay_basins_on_partitions` — for each basin, run
   :func:`basin_atlas.partition_assignment` against ALL partitions (not just
   the top-K) and record the best fit. Returns a per-basin "home partition"
   plus a per-partition "reached / unreached" flag (a partition is *reached*
   iff some basin scores ≥ ``reach_threshold``, default 0.9).

3. :func:`cross_n_partition_summary` — count, per n, how many partitions
   have predicted ratio above various thresholds (proven max, in-repo best,
   Barba × 0.95). Lets us see whether the "tons of unreached high-prediction
   partitions" pattern at H23 generalises to H19, H21.

The Ehlich predicted bound formula here matches `orrick_hunt`'s exactly
(both use the same :func:`_block_gram_matrix` helper); this module differs
only in (a) returning the full enumeration sorted by predicted |det| and
(b) layering the basin overlay + the cross-n summary on top.

Notes
-----
* "Predicted bound" is upper on what a block-Gram of that exact shape can
  achieve; it is **not** an upper bound on the true |det| of any matrix in
  that partition family (the matrix may have a different intra/cross
  pattern). It is the best-case |det| for the canonical 4-valued ansatz.
* A partition can be unreached for two reasons: (i) no matrix realises it
  exactly (the planted Gram pattern has no ±1 realisation), or (ii) we
  simply have not searched there. Distinguishing the two requires
  construction effort (`orrick_hunt.construct_from_partition`); the
  satellite view does not attempt the construction.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from hadamard_maxdet.basin_atlas import BasinEntry, partition_assignment
from hadamard_maxdet.orrick_hunt import EhlichPartition, _block_gram_matrix
from hadamard_maxdet.verifier import upper_bound


# ---------------------------------------------------------------------------
# 1. Full Ehlich enumeration
# ---------------------------------------------------------------------------


def predicted_ehlich_det(
    partition: tuple[int, int, int, int],
    *,
    intra_value: int = 3,
    cross_value: int = -1,
) -> float:
    """Closed-form predicted ``|det|`` for a 4-block ±1 matrix Gram.

    Builds the block-structured Gram with diagonal ``n``, intra-block
    off-diagonals = ``intra_value``, cross-block off-diagonals = ``cross_value``,
    and returns ``sqrt(det(G))``. Returns ``0.0`` if G is not positive
    semi-definite (the partition cannot be realised by any real matrix).
    """
    n = sum(partition)
    G = _block_gram_matrix(tuple(partition), n, intra_value, cross_value)
    sign, logdet = np.linalg.slogdet(G)
    if sign <= 0:
        return 0.0
    return math.exp(0.5 * logdet)


def enumerate_all_ehlich_partitions(
    n: int,
    *,
    intra_value: int = 3,
    cross_value: int = -1,
    min_block_size: int = 1,
) -> list[EhlichPartition]:
    """Enumerate **every** 4-part partition of ``n`` (with ``s_1 ≤ … ≤ s_4``)
    and its predicted block-Gram |det|, sorted by predicted |det| descending.

    Parameters
    ----------
    n
        Matrix order.
    intra_value, cross_value
        Gram pattern for the canonical 4-valued ansatz. Defaults +3, −1
        (the Ehlich/Wojtas conjecture for n ≡ 3 mod 4).
    min_block_size
        Minimum block size; ``1`` allows degenerate single-row blocks.

    Returns
    -------
    list[EhlichPartition]
        All multiset-distinct partitions whose block-Gram is positive
        semi-definite. Sorted by predicted ``|det|`` descending.

    Notes
    -----
    Partitions are returned in canonical sorted order (s_1 ≤ s_2 ≤ s_3 ≤ s_4),
    so each multiset appears exactly once. At n=23 with defaults this gives
    94 partitions; at n=19, 54; at n=21, 72.
    """
    if n < 4 * min_block_size:
        return []
    seen: set[tuple[int, int, int, int]] = set()
    out: list[EhlichPartition] = []
    for s1 in range(min_block_size, n + 1):
        for s2 in range(s1, n + 1):
            for s3 in range(s2, n + 1):
                s4 = n - s1 - s2 - s3
                if s4 < s3:
                    continue
                if s4 < min_block_size:
                    continue
                key = (s1, s2, s3, s4)
                if key in seen:
                    continue
                seen.add(key)
                G = _block_gram_matrix(key, n, intra_value, cross_value)
                sign, logdet = np.linalg.slogdet(G)
                if sign <= 0:
                    continue
                predicted = math.exp(0.5 * logdet)
                out.append(
                    EhlichPartition(
                        sizes=key,
                        intra_value=intra_value,
                        cross_value=cross_value,
                        n=n,
                        predicted_abs_det=predicted,
                    )
                )
    out.sort(key=lambda p: -p.predicted_abs_det)
    return out


# ---------------------------------------------------------------------------
# Partition geometry: complexity scalar for x-axis
# ---------------------------------------------------------------------------


def partition_complexity(partition: tuple[int, int, int, int]) -> float:
    """Scalar describing how unequal the parts are.

    Defined as ``Σ |s_i − n/4|`` (L1 distance from the perfectly-balanced
    partition). Equals 0 iff all four parts equal n/4 (only possible when
    ``n ≡ 0 mod 4``); grows as the partition becomes more skewed. Bounded
    above by ``2·max(s_i) − n/2`` for the most-skewed partitions.
    """
    n = sum(partition)
    mean = n / 4.0
    return float(sum(abs(s - mean) for s in partition))


# ---------------------------------------------------------------------------
# 2. Overlay basins on partitions
# ---------------------------------------------------------------------------


@dataclass
class PartitionOverlay:
    """Best-fit Ehlich partition for one basin, plus the gap to its prediction."""

    basin_label: str
    basin_abs_det: int
    best_partition: tuple[int, int, int, int]
    best_score: float
    predicted_abs_det: float
    gap: float
    """``predicted_abs_det − basin_abs_det`` (positive when prediction is above)."""
    intra_match: int
    intra_total: int
    cross_match: int
    cross_total: int

    def to_json(self) -> dict:
        return {
            "basin_label": self.basin_label,
            "basin_abs_det": self.basin_abs_det,
            "best_partition": list(self.best_partition),
            "best_score": self.best_score,
            "predicted_abs_det": self.predicted_abs_det,
            "gap": self.gap,
            "intra_match": self.intra_match,
            "intra_total": self.intra_total,
            "cross_match": self.cross_match,
            "cross_total": self.cross_total,
        }


@dataclass
class PartitionLandscape:
    """Per-partition reached/unreached status from the basin overlay."""

    partition: tuple[int, int, int, int]
    predicted_abs_det: float
    predicted_ratio: float
    """``predicted_abs_det / Barba(n)``."""
    complexity: float
    reached: bool
    """``True`` iff some basin scores >= reach_threshold on this partition."""
    best_basin_label: str | None
    best_score: float
    best_basin_abs_det: int

    def to_json(self) -> dict:
        return {
            "partition": list(self.partition),
            "predicted_abs_det": self.predicted_abs_det,
            "predicted_ratio": self.predicted_ratio,
            "complexity": self.complexity,
            "reached": self.reached,
            "best_basin_label": self.best_basin_label,
            "best_score": self.best_score,
            "best_basin_abs_det": self.best_basin_abs_det,
        }


def overlay_basins_on_partitions(
    n: int,
    basins: Sequence[BasinEntry],
    *,
    intra_value: int = 3,
    cross_value: int = -1,
    reach_threshold: float = 0.9,
    n_restarts: int = 4,
    seed: int = 0,
) -> tuple[list[PartitionOverlay], list[PartitionLandscape]]:
    """For every basin × every partition, fit and record best partition per basin
    AND best basin per partition.

    Returns ``(per_basin, per_partition)``:

    * ``per_basin`` — one :class:`PartitionOverlay` per basin: its best-fit
      partition and the gap between that partition's prediction and the
      basin's actual |det|.
    * ``per_partition`` — one :class:`PartitionLandscape` per partition:
      whether ANY basin scores ≥ ``reach_threshold``, plus the best-scoring
      basin (if any).

    Parameters
    ----------
    reach_threshold
        Score above which we declare a partition "reached" by the basin.
        Default 0.9 (basin's row-permutation matches ≥ 90 % of the planted
        Gram pattern). Partition fits at the basins we have land in the
        0.65–0.78 band, so this threshold is *strict* by design.
    n_restarts
        Number of random restarts in the greedy block assignment per
        (basin, partition) pair. Lower than the basin-atlas default of 8
        to keep this enumeration tractable; the trade-off is that an
        occasional partition fit may underestimate the true score by a few
        percent.

    Notes
    -----
    Cost: ``O(|basins| × |partitions| × n_restarts × n²)`` per partition,
    plus the local-search swap passes inside `partition_assignment`. At
    n=23 with 29 basins × 94 partitions × 4 restarts ≈ 11k assignments;
    each is ~5 ms ⇒ ~1 min total.
    """
    partitions = enumerate_all_ehlich_partitions(
        n, intra_value=intra_value, cross_value=cross_value
    )
    barba = float(upper_bound(n))

    # Per-basin best partition.
    per_basin: list[PartitionOverlay] = []
    # Per-partition best basin.
    best_basin_per_partition: dict[
        tuple[int, int, int, int],
        tuple[float, str | None, int]  # (score, basin_label, basin_abs_det)
    ] = {p.sizes: (-1.0, None, 0) for p in partitions}

    for bi, basin in enumerate(basins):
        if basin.n != n:
            continue
        best_score = -1.0
        best_part: tuple[int, int, int, int] | None = None
        best_pred = 0.0
        best_im = 0
        best_it = 0
        best_cm = 0
        best_ct = 0
        for p in partitions:
            rng = random.Random(seed + 1009 * bi + hash(p.sizes) % 9973)
            res = partition_assignment(
                basin.matrix,
                p.sizes,
                intra_value=intra_value,
                cross_value=cross_value,
                rng=rng,
                n_restarts=n_restarts,
            )
            score = res.gram_match_score
            # Per-partition update.
            cur_best, _, _ = best_basin_per_partition[p.sizes]
            if score > cur_best:
                best_basin_per_partition[p.sizes] = (
                    score, basin.label, basin.abs_det
                )
            # Per-basin update.
            if score > best_score:
                best_score = score
                best_part = p.sizes
                best_pred = p.predicted_abs_det
                best_im = res.intra_match_count
                best_it = res.intra_total
                best_cm = res.cross_match_count
                best_ct = res.cross_total
        assert best_part is not None
        per_basin.append(
            PartitionOverlay(
                basin_label=basin.label,
                basin_abs_det=basin.abs_det,
                best_partition=best_part,
                best_score=best_score,
                predicted_abs_det=best_pred,
                gap=best_pred - basin.abs_det,
                intra_match=best_im,
                intra_total=best_it,
                cross_match=best_cm,
                cross_total=best_ct,
            )
        )

    # Build per-partition landscape.
    per_partition: list[PartitionLandscape] = []
    for p in partitions:
        score, label, det = best_basin_per_partition[p.sizes]
        per_partition.append(
            PartitionLandscape(
                partition=p.sizes,
                predicted_abs_det=p.predicted_abs_det,
                predicted_ratio=p.predicted_abs_det / barba if barba > 0 else 0.0,
                complexity=partition_complexity(p.sizes),
                reached=score >= reach_threshold,
                best_basin_label=label,
                best_score=max(score, 0.0),
                best_basin_abs_det=det,
            )
        )
    # Sort per_partition by predicted |det| descending (matches enumerate order).
    per_partition.sort(key=lambda r: -r.predicted_abs_det)
    return per_basin, per_partition


# ---------------------------------------------------------------------------
# 3. Cross-n summary
# ---------------------------------------------------------------------------


@dataclass
class CrossNRow:
    """One row of the cross-n satellite summary."""

    n: int
    n_partitions: int
    barba: int
    proven_max: int | None
    proven_max_ratio: float | None
    in_repo_best: int | None
    in_repo_best_ratio: float | None
    n_above_proven: int | None
    n_above_in_repo: int | None
    n_above_barba_95: int

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "n_partitions": self.n_partitions,
            "barba": self.barba,
            "proven_max": self.proven_max,
            "proven_max_ratio": self.proven_max_ratio,
            "in_repo_best": self.in_repo_best,
            "in_repo_best_ratio": self.in_repo_best_ratio,
            "n_above_proven": self.n_above_proven,
            "n_above_in_repo": self.n_above_in_repo,
            "n_above_barba_95": self.n_above_barba_95,
        }


def cross_n_partition_summary(
    ns: Sequence[int],
    *,
    in_repo_bests: dict[int, int] | None = None,
    proven_maxes: dict[int, int | None] | None = None,
    intra_value: int = 3,
    cross_value: int = -1,
) -> list[CrossNRow]:
    """For each ``n`` in ``ns``, enumerate Ehlich partitions and count how
    many beat various thresholds.

    Parameters
    ----------
    ns
        Orders to summarise (e.g. ``[19, 21, 23]``).
    in_repo_bests
        Map ``n -> our best |det|`` for the open cases. If ``None``, falls
        back to ``proven_maxes`` (proven optimal IS our in-repo best when
        the order is solved).
    proven_maxes
        Map ``n -> proven max |det|`` (or ``None`` if unknown). If not
        supplied, looked up from :mod:`hadamard_maxdet.orders`.
    """
    from hadamard_maxdet.orders import BENCHMARKS

    rows: list[CrossNRow] = []
    for n in ns:
        partitions = enumerate_all_ehlich_partitions(
            n, intra_value=intra_value, cross_value=cross_value
        )
        barba = upper_bound(n)
        oid = f"H{n}"
        order = BENCHMARKS.get(oid)
        proven = order.best_known if order else None
        if proven_maxes is not None and n in proven_maxes:
            proven = proven_maxes[n]
        in_repo = proven  # default
        if in_repo_bests is not None and n in in_repo_bests:
            in_repo = in_repo_bests[n]

        n_above_barba_95 = sum(
            1 for p in partitions if p.predicted_abs_det >= 0.95 * barba
        )
        n_above_proven: int | None
        n_above_in_repo: int | None
        if proven is not None:
            n_above_proven = sum(
                1 for p in partitions if p.predicted_abs_det >= proven
            )
        else:
            n_above_proven = None
        if in_repo is not None:
            n_above_in_repo = sum(
                1 for p in partitions if p.predicted_abs_det >= in_repo
            )
        else:
            n_above_in_repo = None

        rows.append(
            CrossNRow(
                n=n,
                n_partitions=len(partitions),
                barba=barba,
                proven_max=proven,
                proven_max_ratio=(proven / barba) if proven else None,
                in_repo_best=in_repo,
                in_repo_best_ratio=(in_repo / barba) if in_repo else None,
                n_above_proven=n_above_proven,
                n_above_in_repo=n_above_in_repo,
                n_above_barba_95=n_above_barba_95,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Plotting (matplotlib optional)
# ---------------------------------------------------------------------------


def plot_satellite_view(
    n: int,
    per_partition: Sequence[PartitionLandscape],
    out_path: str,
    *,
    orrick_ratio: float = 0.7091,
    btc_ratio: float = 0.6702,
    title: str | None = None,
) -> bool:
    """2D scatter of partition complexity vs predicted ratio.

    Reached partitions get filled markers sized by best score; unreached
    get open circles. Horizontal lines at Barba (1.0), Orrick, BTC.

    Returns ``True`` if the plot was written, ``False`` if matplotlib is
    not available (silent fail; caller can decide).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    xs_r = [r.complexity for r in per_partition if r.reached]
    ys_r = [r.predicted_ratio for r in per_partition if r.reached]
    sz_r = [50 + 200 * r.best_score for r in per_partition if r.reached]
    xs_u = [r.complexity for r in per_partition if not r.reached]
    ys_u = [r.predicted_ratio for r in per_partition if not r.reached]
    # marker size scaled by best partial score for unreached too (lets us
    # see *closeness* to reached among the unreached)
    sz_u = [20 + 100 * r.best_score for r in per_partition if not r.reached]

    fig, ax = plt.subplots(figsize=(9, 6))
    if xs_u:
        ax.scatter(
            xs_u, ys_u, s=sz_u, facecolors="none", edgecolors="#888",
            linewidths=1.0, label=f"unreached ({len(xs_u)})",
        )
    if xs_r:
        ax.scatter(
            xs_r, ys_r, s=sz_r, c="C3", alpha=0.85, edgecolors="black",
            label=f"reached ({len(xs_r)})",
        )
    ax.axhline(1.0, color="C0", linestyle="--", linewidth=1.0, label="Barba")
    ax.axhline(orrick_ratio, color="C2", linestyle="--",
               linewidth=1.0, label=f"Orrick = {orrick_ratio:.4f}")
    ax.axhline(btc_ratio, color="C1", linestyle=":",
               linewidth=1.0, label=f"BTC = {btc_ratio:.4f}")
    ax.set_xlabel("Partition complexity   $\\sum_i |s_i - n/4|$")
    ax.set_ylabel("Predicted |det| ratio to Barba")
    ax.set_title(title or f"H{n} satellite view: Ehlich-partition landscape")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


__all__ = [
    "predicted_ehlich_det",
    "enumerate_all_ehlich_partitions",
    "partition_complexity",
    "PartitionOverlay",
    "PartitionLandscape",
    "overlay_basins_on_partitions",
    "CrossNRow",
    "cross_n_partition_summary",
    "plot_satellite_view",
]
