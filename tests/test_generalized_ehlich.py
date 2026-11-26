"""Tests for :mod:`hadamard_maxdet.generalized_ehlich`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hadamard_maxdet.generalized_ehlich import (
    DEFAULT_AB_VALUES,
    AbCalibrationRow,
    GeneralizedRegime,
    best_per_ab_pair,
    calibrate_against_proven,
    enumerate_ab_grid,
    enumerate_partitions,
    hadamard_ceiling,
    predicted_block_det,
    predicted_block_ratio,
    top_regimes_above,
)
from hadamard_maxdet.orrick_hunt import _block_gram_matrix
from hadamard_maxdet.satellite import predicted_ehlich_det
from hadamard_maxdet.verifier import upper_bound


# ---------------------------------------------------------------------------
# Closed-form vs numerical determinant
# ---------------------------------------------------------------------------


def _numerical_block_det(n, partition, intra, cross):
    G = _block_gram_matrix(tuple(partition), n, intra, cross)
    sign, logd = np.linalg.slogdet(G)
    if sign <= 0:
        return 0.0
    return math.exp(0.5 * logd)


@pytest.mark.parametrize(
    "n,partition,intra,cross",
    [
        (8, (2, 2, 2, 2), 0, 0),       # Hadamard-tight
        (8, (2, 2, 2, 2), 1, -1),
        (11, (2, 3, 3, 3), 1, -1),
        (12, (3, 3, 3, 3), 0, 0),      # Hadamard-tight at n=12
        (19, (4, 5, 5, 5), 3, -1),     # canonical Ehlich
        (19, (4, 5, 5, 5), 5, -1),
        (19, (1, 6, 6, 6), -3, 3),
        (21, (5, 5, 5, 6), 3, -1),
        (21, (1, 6, 7, 7), -5, 1),
        (23, (5, 6, 6, 6), 3, -1),
        (23, (5, 6, 6, 6), 1, -1),
        (23, (1, 1, 1, 20), 1, -1),
        (23, (3, 5, 5, 10), -3, 3),
        (20, (5, 5, 5, 5), 0, 0),      # Hadamard-tight
    ],
)
def test_predicted_block_det_matches_numpy(n, partition, intra, cross):
    """Closed-form det matches numerical slogdet to machine precision."""
    ours = predicted_block_det(n, partition, intra, cross)
    num = _numerical_block_det(n, partition, intra, cross)
    if num == 0.0:
        assert ours == 0.0
        return
    rel = abs(ours - num) / num
    assert rel < 1e-10, f"rel diff {rel} too large: ours={ours}, num={num}"


# ---------------------------------------------------------------------------
# Canonical (3, -1) parity with predicted_ehlich_det
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,partition",
    [
        (19, (4, 5, 5, 5)),
        (19, (1, 6, 6, 6)),
        (21, (5, 5, 5, 6)),
        (21, (3, 6, 6, 6)),
        (23, (5, 6, 6, 6)),
        (23, (4, 5, 7, 7)),
    ],
)
def test_canonical_3_minus1_matches_satellite(n, partition):
    """generalized_ehlich at (intra=3, cross=-1) reproduces satellite.predicted_ehlich_det."""
    ours = predicted_block_det(n, partition, 3, -1)
    canon = predicted_ehlich_det(partition, intra_value=3, cross_value=-1)
    if canon == 0.0 and ours == 0.0:
        return
    assert math.isclose(ours, canon, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# Hadamard-tight regime spot checks
# ---------------------------------------------------------------------------


def test_n8_hadamard_tight_intra_cross_zero():
    """n=8, partition (2,2,2,2), intra=0, cross=0 ⇒ G = 8 I ⇒ det=8^8 ⇒ |det|=8^4=4096."""
    pred = predicted_block_det(8, (2, 2, 2, 2), 0, 0)
    assert math.isclose(pred, 8 ** 4, rel_tol=1e-12)


def test_n20_hadamard_tight_intra_cross_zero():
    """n=20, partition (5,5,5,5), intra=0, cross=0 ⇒ G = 20 I ⇒ |det|=20^10."""
    pred = predicted_block_det(20, (5, 5, 5, 5), 0, 0)
    assert math.isclose(pred, 20 ** 10, rel_tol=1e-12)


def test_hadamard_ceiling_matches_n_pow_n_over_2():
    for n in (4, 8, 12, 16, 20, 19, 21, 23):
        assert math.isclose(hadamard_ceiling(n), float(n) ** (n / 2.0))


# ---------------------------------------------------------------------------
# Permutation invariance
# ---------------------------------------------------------------------------


def test_predicted_block_det_invariant_under_partition_permutation():
    base = predicted_block_det(23, (5, 6, 6, 6), 3, -1)
    for perm in [(6, 5, 6, 6), (6, 6, 5, 6), (6, 6, 6, 5)]:
        assert math.isclose(predicted_block_det(23, perm, 3, -1), base, rel_tol=1e-12)


def test_predicted_block_det_raises_on_wrong_sum():
    with pytest.raises(ValueError):
        predicted_block_det(23, (5, 6, 6, 7), 3, -1)


# ---------------------------------------------------------------------------
# enumerate_partitions
# ---------------------------------------------------------------------------


def test_enumerate_partitions_n23_count_matches_satellite():
    """Same partition count as satellite.enumerate_all_ehlich_partitions."""
    from hadamard_maxdet.satellite import enumerate_all_ehlich_partitions
    parts = enumerate_partitions(23)
    sat = enumerate_all_ehlich_partitions(23)
    assert len(parts) == len(sat)


def test_enumerate_partitions_canonical_sorted():
    for p in enumerate_partitions(23):
        assert list(p) == sorted(p)
        assert sum(p) == 23


# ---------------------------------------------------------------------------
# enumerate_ab_grid
# ---------------------------------------------------------------------------


def test_enumerate_ab_grid_returns_sorted_descending():
    grid = enumerate_ab_grid(19)
    for r1, r2 in zip(grid, grid[1:]):
        assert r1.predicted_ratio >= r2.predicted_ratio


def test_enumerate_ab_grid_distinct_ab_pairs_only():
    grid = enumerate_ab_grid(23)
    for r in grid:
        assert r.intra != r.cross


def test_enumerate_ab_grid_parity_consistent_for_odd_n():
    """At odd n with parity enforcement, all (a, b) entries must be odd."""
    for n in (19, 21, 23):
        grid = enumerate_ab_grid(n)
        for r in grid:
            assert r.intra % 2 == 1, f"intra={r.intra} not odd at n={n}"
            assert r.cross % 2 == 1, f"cross={r.cross} not odd at n={n}"


def test_enumerate_ab_grid_hadamard_bounded():
    """All predicted |det| values are at most n^(n/2) when enforce_hadamard=True."""
    for n in (19, 21, 23):
        ceil = hadamard_ceiling(n)
        grid = enumerate_ab_grid(n)
        for r in grid:
            assert r.predicted_abs_det <= ceil * (1 + 1e-9)


def test_enumerate_ab_grid_unfiltered_can_exceed_hadamard():
    """Without filtering, the formula DOES return triples with predicted > Hadamard
    (these are mathematical artefacts of the planted Gram, not realisable matrices).
    """
    grid = enumerate_ab_grid(19, enforce_hadamard=False, enforce_parity=False)
    ceil = hadamard_ceiling(19)
    n_above = sum(1 for r in grid if r.predicted_abs_det > ceil)
    assert n_above > 0


# ---------------------------------------------------------------------------
# Cross-n calibration
# ---------------------------------------------------------------------------


def test_calibrate_n19_proven_max_has_at_least_one_plausible():
    """At least one (a, b) pair has top prediction ≥ proven 0.9413·Barba."""
    rows = calibrate_against_proven(19, proven_ratio=0.9413)
    plausible = [r for r in rows if r.plausible]
    assert len(plausible) >= 1, (
        "expected at least one (a,b) regime to plausibly contain the H19 proven max; "
        "found 0 — formula or filter is wrong"
    )


def test_calibrate_n21_proven_max_has_at_least_one_plausible():
    rows = calibrate_against_proven(21, proven_ratio=0.9058)
    plausible = [r for r in rows if r.plausible]
    assert len(plausible) >= 1


def test_calibrate_canonical_3_minus1_predicts_below_proven_at_n21():
    """The known-suboptimal canonical (+3, -1) must have top prediction
    strictly below the proven n=21 max ratio (0.9058)."""
    rows = calibrate_against_proven(21, proven_ratio=0.9058)
    by_pair = {(r.intra, r.cross): r for r in rows}
    canonical = by_pair[(3, -1)]
    assert canonical.top_predicted_ratio < 0.9058, (
        "canonical (+3, -1) predicts above proven max — contradicts the "
        "known suboptimality bombshell"
    )


# ---------------------------------------------------------------------------
# top_regimes_above
# ---------------------------------------------------------------------------


def test_top_regimes_above_orrick_at_n23_is_nonempty():
    """There exist (a, b, partition) triples predicting ratio > Orrick 0.7091."""
    above = top_regimes_above(23, threshold_ratio=0.7091)
    assert len(above) > 0


def test_top_regimes_above_returns_sorted_descending():
    above = top_regimes_above(23, threshold_ratio=0.7091)
    for r1, r2 in zip(above, above[1:]):
        assert r1.predicted_ratio >= r2.predicted_ratio


def test_predicted_block_ratio_consistent():
    base = float(upper_bound(23))
    for p in [(5, 6, 6, 6), (4, 5, 7, 7)]:
        for a, b in [(3, -1), (1, -1), (-3, 3)]:
            d = predicted_block_det(23, p, a, b)
            r = predicted_block_ratio(23, p, a, b)
            assert math.isclose(r, d / base, rel_tol=1e-12)
