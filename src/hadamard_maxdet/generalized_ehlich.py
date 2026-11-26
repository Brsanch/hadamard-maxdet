"""Generalized satellite view of the Hadamard maxdet landscape.

Where :mod:`satellite` enumerates 4-part partitions of n under the *canonical*
Ehlich/Wojtas ansatz (intra-block Gram = +3, cross-block Gram = -1), this
module sweeps the **(intra, cross) Gram-value plane** to find which (a, b)
regime the proven H19 / H21 maxima actually live in. The previous
satellite-view round revealed that the canonical (+3, -1) ansatz is provably
suboptimal at H19 and H21 (proven max sits 23-28 percentage points above the
canonical 4-block ceiling). The proven optima must live in a non-canonical
(a, b) regime — this module finds it.

Closed-form determinant of the block Gram
------------------------------------------

Let G be n×n with diagonal d = n, intra-block off-diagonals = a, cross-block
off-diagonals = b, partition (s_1, s_2, s_3, s_4). Write

    G = (d-a) I + (a-b) B + b J

where B = block-diagonal sum_i v_i v_i^T (v_i = indicator of block i), and
J = e e^T is the all-ones matrix. The eigen-decomposition splits naturally:

* On the (n-4)-dimensional subspace orthogonal to all v_i, G acts as (d-a) I,
  contributing eigenvalue (d-a) with multiplicity (n-4).
* On the 4-dimensional span{v_1, ..., v_4}, G is represented in the v-basis
  by the 4×4 matrix

      M = K + b · 1 · s^T,   K = diag((d-a) + (a-b) s_i),  s = (s_1,...,s_4).

  By the matrix determinant lemma,

      det(M) = det(K) · (1 + b · sum_i s_i / k_i),
            where k_i = (d-a) + (a-b) s_i.

Therefore:

      det(G) = (d-a)^(n-4) · prod_i k_i · (1 + b · sum_i s_i / k_i).

Predicted |det| of a ±1 matrix realising this Gram is sqrt(det(G)) (positive
only when det(G) > 0; otherwise the partition is ruled out for that (a, b)).

Three deliverables wrap this:

* :func:`predicted_block_det` — closed-form det.
* :func:`enumerate_ab_grid` — sweep all (a, b) pairs and partitions of n.
* :func:`top_regimes_above` — return all (a, b, partition) triples whose
  predicted ratio exceeds a threshold.

The closed-form result agrees exactly with the numerical determinant from
:func:`hadamard_maxdet.orrick_hunt._block_gram_matrix` (verified in tests).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from hadamard_maxdet.verifier import upper_bound


# ---------------------------------------------------------------------------
# Closed-form determinant
# ---------------------------------------------------------------------------


def _block_gram_logdet(
    n: int,
    partition: Sequence[int],
    intra: float,
    cross: float,
) -> tuple[int, float]:
    """Return (sign, log|det|) of the block Gram matrix.

    Uses the closed-form decomposition described in the module docstring.
    Avoids constructing the n×n matrix; runs in O(1) arithmetic in the
    block count (4) plus O(n) for the multiplicity-(n-4) eigenvalue
    contribution (a single power).
    """
    d = float(n)
    a = float(intra)
    b = float(cross)
    s = [float(x) for x in partition]

    # Eigenvalue (d-a) with multiplicity (n-4) — its log contribution.
    diag_eig = d - a
    sign_diag = 0
    log_diag = 0.0
    n_minor_4 = n - 4
    if n_minor_4 < 0:
        # Partition has more parts than n allows — degenerate.
        return 0, float("-inf")
    if n_minor_4 > 0:
        if diag_eig == 0.0:
            return 0, float("-inf")
        sign_diag = 1 if diag_eig > 0 else (-1) ** n_minor_4
        log_diag = n_minor_4 * math.log(abs(diag_eig))
    else:
        sign_diag = 1
        log_diag = 0.0

    # 4×4 block: M = K + b · 1 · s^T, K = diag(k_i), k_i = (d-a) + (a-b) s_i.
    k = [(d - a) + (a - b) * si for si in s]
    if any(ki == 0.0 for ki in k):
        return 0, float("-inf")
    # det(K) = prod k_i.
    sign_K = 1
    log_K = 0.0
    for ki in k:
        if ki < 0:
            sign_K = -sign_K
        log_K += math.log(abs(ki))
    # 1 + b · sum s_i / k_i.
    bracket = 1.0 + b * sum(si / ki for si, ki in zip(s, k))
    if bracket == 0.0:
        return 0, float("-inf")
    sign_bracket = 1 if bracket > 0 else -1
    log_bracket = math.log(abs(bracket))

    sign = sign_diag * sign_K * sign_bracket
    log_det = log_diag + log_K + log_bracket
    return sign, log_det


def predicted_block_det(
    n: int,
    partition: Sequence[int],
    intra: float,
    cross: float,
) -> float:
    """Return sqrt(det(G_block)) for the (intra, cross) block Gram.

    Returns 0.0 if det(G) <= 0 (partition not realisable as M M^T for that
    (intra, cross) — the planted Gram is not PSD, so no real ±1 matrix can
    have this Gram).
    """
    if sum(partition) != n:
        raise ValueError(
            f"partition {tuple(partition)} sums to {sum(partition)}, expected {n}"
        )
    sign, log_det = _block_gram_logdet(n, partition, intra, cross)
    if sign <= 0:
        return 0.0
    return math.exp(0.5 * log_det)


def predicted_block_ratio(
    n: int,
    partition: Sequence[int],
    intra: float,
    cross: float,
    base_bound: float | None = None,
) -> float:
    """Predicted |det| as a fraction of the relevant upper bound.

    ``base_bound`` defaults to :func:`hadamard_maxdet.verifier.upper_bound`
    (Barba/Ehlich-Wojtas/Ehlich/Hadamard depending on n mod 4).
    """
    if base_bound is None:
        base_bound = float(upper_bound(n))
    if base_bound <= 0:
        return 0.0
    return predicted_block_det(n, partition, intra, cross) / base_bound


# ---------------------------------------------------------------------------
# Enumeration helpers
# ---------------------------------------------------------------------------


def enumerate_partitions(n: int, *, min_block_size: int = 1) -> list[tuple[int, int, int, int]]:
    """All multiset-distinct ordered 4-partitions ``s_1 ≤ s_2 ≤ s_3 ≤ s_4`` of n."""
    out: list[tuple[int, int, int, int]] = []
    for s1 in range(min_block_size, n + 1):
        for s2 in range(s1, n + 1):
            for s3 in range(s2, n + 1):
                s4 = n - s1 - s2 - s3
                if s4 < s3 or s4 < min_block_size:
                    continue
                out.append((s1, s2, s3, s4))
    return out


# ---------------------------------------------------------------------------
# (a, b) sweep
# ---------------------------------------------------------------------------


DEFAULT_AB_VALUES: tuple[int, ...] = (-5, -3, -1, 1, 3, 5)


@dataclass(frozen=True)
class GeneralizedRegime:
    """One (n, intra, cross, partition) combination with its prediction."""

    n: int
    intra: int
    cross: int
    partition: tuple[int, int, int, int]
    predicted_abs_det: float
    predicted_ratio: float

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "intra": self.intra,
            "cross": self.cross,
            "partition": list(self.partition),
            "predicted_abs_det": self.predicted_abs_det,
            "predicted_ratio": self.predicted_ratio,
        }


def hadamard_ceiling(n: int) -> float:
    """``n^(n/2)`` — Hadamard's universal upper bound on |det| of a ±1 matrix."""
    return float(n) ** (n / 2.0)


def enumerate_ab_grid(
    n: int,
    *,
    ab_values: Sequence[int] = DEFAULT_AB_VALUES,
    require_distinct_ab: bool = True,
    min_block_size: int = 1,
    enforce_hadamard: bool = True,
    enforce_parity: bool = True,
) -> list[GeneralizedRegime]:
    """Sweep all (intra, cross) ∈ ab_values² × all 4-part partitions of n.

    Parameters
    ----------
    n
        Matrix order.
    ab_values
        Candidate Gram values to sweep over. Default ``{-5,-3,-1,+1,+3,+5}``
        gives 30 distinct (intra, cross) pairs (with require_distinct_ab=True).
    require_distinct_ab
        If True (default), skip pairs with intra == cross (they collapse the
        block structure to a single uniform-value off-diagonal regime — still
        valid block-Grams but trivial).
    min_block_size
        Minimum part size in the partition; ``1`` allows degenerate single-row
        blocks.
    enforce_hadamard
        If True (default), drop triples whose predicted |det| exceeds the
        universal Hadamard bound n^(n/2). Such Grams cannot be realised by
        any real ±1 matrix; their large predicted determinant is a
        mathematical artefact of the planted block structure exceeding what
        ±1 entries can produce.
    enforce_parity
        If True (default), drop triples where (a, b) violate the parity
        constraint G_ij ≡ n (mod 2). For odd n, both intra and cross must be
        odd; for even n, both must be even (incl. 0). Without this, the
        Gram has no ±1 realiser at all.

    Returns
    -------
    list[GeneralizedRegime]
        All (intra, cross, partition) triples whose block Gram is positive
        definite (and, by default, Hadamard-realisable + parity-consistent),
        sorted by predicted ratio descending.
    """
    base = float(upper_bound(n))
    hadamard = hadamard_ceiling(n)
    n_parity = n % 2
    partitions = enumerate_partitions(n, min_block_size=min_block_size)
    out: list[GeneralizedRegime] = []
    for a in ab_values:
        if enforce_parity and (a % 2) != n_parity:
            continue
        for b in ab_values:
            if require_distinct_ab and a == b:
                continue
            if enforce_parity and (b % 2) != n_parity:
                continue
            for p in partitions:
                pred = predicted_block_det(n, p, a, b)
                if pred <= 0.0:
                    continue
                if enforce_hadamard and pred > hadamard:
                    continue
                ratio = pred / base if base > 0 else 0.0
                out.append(
                    GeneralizedRegime(
                        n=n,
                        intra=int(a),
                        cross=int(b),
                        partition=p,
                        predicted_abs_det=pred,
                        predicted_ratio=ratio,
                    )
                )
    out.sort(key=lambda r: -r.predicted_ratio)
    return out


def top_regimes_above(
    n: int,
    threshold_ratio: float,
    *,
    ab_values: Sequence[int] = DEFAULT_AB_VALUES,
    min_block_size: int = 1,
    enforce_hadamard: bool = True,
    enforce_parity: bool = True,
) -> list[GeneralizedRegime]:
    """All (intra, cross, partition) triples with predicted ratio > threshold."""
    grid = enumerate_ab_grid(
        n,
        ab_values=ab_values,
        min_block_size=min_block_size,
        enforce_hadamard=enforce_hadamard,
        enforce_parity=enforce_parity,
    )
    return [r for r in grid if r.predicted_ratio > threshold_ratio]


def best_per_ab_pair(
    n: int,
    *,
    ab_values: Sequence[int] = DEFAULT_AB_VALUES,
    min_block_size: int = 1,
    enforce_hadamard: bool = True,
    enforce_parity: bool = True,
) -> dict[tuple[int, int], GeneralizedRegime]:
    """For each (a, b) pair, return the top-predicted partition.

    Useful for "which (a, b) regime gives the highest predicted ceiling at n?"
    """
    grid = enumerate_ab_grid(
        n,
        ab_values=ab_values,
        min_block_size=min_block_size,
        enforce_hadamard=enforce_hadamard,
        enforce_parity=enforce_parity,
    )
    best: dict[tuple[int, int], GeneralizedRegime] = {}
    for r in grid:
        key = (r.intra, r.cross)
        cur = best.get(key)
        if cur is None or r.predicted_ratio > cur.predicted_ratio:
            best[key] = r
    return best


# ---------------------------------------------------------------------------
# Cross-n calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbCalibrationRow:
    """One (n, (a, b)) calibration entry."""

    n: int
    intra: int
    cross: int
    top_predicted_ratio: float
    top_partition: tuple[int, int, int, int]
    proven_ratio: float | None
    gap_to_proven: float | None
    """``top_predicted_ratio - proven_ratio`` (positive ⇒ regime can in
    principle reach the proven max)."""
    plausible: bool
    """``True`` iff ``top_predicted_ratio >= proven_ratio`` (or no proven
    value, in which case ``False``)."""

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "intra": self.intra,
            "cross": self.cross,
            "top_predicted_ratio": self.top_predicted_ratio,
            "top_partition": list(self.top_partition),
            "proven_ratio": self.proven_ratio,
            "gap_to_proven": self.gap_to_proven,
            "plausible": self.plausible,
        }


def calibrate_against_proven(
    n: int,
    proven_ratio: float | None,
    *,
    ab_values: Sequence[int] = DEFAULT_AB_VALUES,
    min_block_size: int = 1,
    enforce_hadamard: bool = True,
    enforce_parity: bool = True,
) -> list[AbCalibrationRow]:
    """For each (a, b) pair, top-prediction-vs-proven ratio at n.

    Returns one row per (a, b) pair, sorted by top-predicted ratio descending.
    Pairs with ``top_predicted_ratio >= proven_ratio`` are flagged as
    *plausible* — these are the (a, b) regimes that in principle CAN contain
    the proven max. Pairs with prediction below the proven max are
    *eliminated* — the proven matrix cannot live in their block-Gram pattern.
    """
    best = best_per_ab_pair(
        n,
        ab_values=ab_values,
        min_block_size=min_block_size,
        enforce_hadamard=enforce_hadamard,
        enforce_parity=enforce_parity,
    )
    rows: list[AbCalibrationRow] = []
    for (a, b), reg in best.items():
        gap = (reg.predicted_ratio - proven_ratio) if proven_ratio is not None else None
        plausible = (
            proven_ratio is not None and reg.predicted_ratio >= proven_ratio
        )
        rows.append(
            AbCalibrationRow(
                n=n,
                intra=a,
                cross=b,
                top_predicted_ratio=reg.predicted_ratio,
                top_partition=reg.partition,
                proven_ratio=proven_ratio,
                gap_to_proven=gap,
                plausible=plausible,
            )
        )
    rows.sort(key=lambda r: -r.top_predicted_ratio)
    return rows


__all__ = [
    "DEFAULT_AB_VALUES",
    "hadamard_ceiling",
    "predicted_block_det",
    "predicted_block_ratio",
    "enumerate_partitions",
    "GeneralizedRegime",
    "enumerate_ab_grid",
    "top_regimes_above",
    "best_per_ab_pair",
    "AbCalibrationRow",
    "calibrate_against_proven",
]
