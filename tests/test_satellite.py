"""Tests for :mod:`hadamard_maxdet.satellite` — the satellite view layer."""

from __future__ import annotations

import math

import pytest

from hadamard_maxdet.orrick_hunt import enumerate_ehlich_partitions
from hadamard_maxdet.satellite import (
    cross_n_partition_summary,
    enumerate_all_ehlich_partitions,
    overlay_basins_on_partitions,
    partition_complexity,
    predicted_ehlich_det,
)
from hadamard_maxdet.verifier import upper_bound


# ---------------------------------------------------------------------------
# 1. Enumeration
# ---------------------------------------------------------------------------


def test_enumerate_at_n8_includes_hadamard_tight_partition():
    """At n=8 with intra=0, cross=0 the block-Gram is nI ⇒ predicted=n^(n/2) (Hadamard).

    This confirms the formula reduces to the Hadamard bound when the planted
    Gram is the identity scaled by n.
    """
    parts = enumerate_all_ehlich_partitions(8, intra_value=0, cross_value=0)
    assert len(parts) > 0
    barba8 = upper_bound(8)
    # Best predicted ratio should be exactly 1.0 (Hadamard bound).
    best_ratio = max(p.predicted_abs_det / barba8 for p in parts)
    assert math.isclose(best_ratio, 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_enumerate_n23_has_at_least_50_partitions():
    parts = enumerate_all_ehlich_partitions(23)
    assert len(parts) >= 50, f"expected ≥ 50 partitions at n=23, got {len(parts)}"


def test_enumerate_n19_n21_n23_counts_reasonable():
    n_to_count = {n: len(enumerate_all_ehlich_partitions(n)) for n in (19, 21, 23)}
    # All three should be in the documented 30-150 band.
    for n, c in n_to_count.items():
        assert 30 <= c <= 200, (
            f"unexpected partition count at n={n}: {c} (expected 30–200)"
        )
    # Counts grow with n.
    assert n_to_count[19] <= n_to_count[21] <= n_to_count[23]


def test_enumerate_returns_canonical_sorted_partitions():
    """Each returned partition should satisfy s_1 ≤ s_2 ≤ s_3 ≤ s_4."""
    for n in (19, 21, 23):
        for p in enumerate_all_ehlich_partitions(n):
            sizes = p.sizes
            assert list(sizes) == sorted(sizes), f"partition not sorted: {sizes}"
            assert sum(sizes) == n
            assert all(s >= 1 for s in sizes)


def test_enumerate_sorted_by_predicted_det_descending():
    parts = enumerate_all_ehlich_partitions(23)
    for a, b in zip(parts, parts[1:]):
        assert a.predicted_abs_det >= b.predicted_abs_det


def test_enumerate_no_duplicate_multisets():
    parts = enumerate_all_ehlich_partitions(23)
    seen = set()
    for p in parts:
        assert p.sizes not in seen
        seen.add(p.sizes)


# ---------------------------------------------------------------------------
# Predicted-bound formula matches orrick_hunt for the canonical (3, -1) ansatz
# ---------------------------------------------------------------------------


def test_predicted_ehlich_det_matches_orrick_hunt_for_5666():
    """``predicted_ehlich_det((6,6,6,5))`` matches ``enumerate_ehlich_partitions``."""
    # orrick_hunt uses the (3, -1) ansatz by default at n=23.
    op = next(
        p for p in enumerate_ehlich_partitions(23)
        if p.sizes == (5, 6, 6, 6)
    )
    pred = predicted_ehlich_det((6, 6, 6, 5))  # any permutation of (5,6,6,6)
    assert math.isclose(pred, op.predicted_abs_det, rel_tol=1e-9)


def test_predicted_ehlich_det_invariant_under_permutation():
    base = predicted_ehlich_det((5, 6, 6, 6))
    for perm in [(6, 5, 6, 6), (6, 6, 5, 6), (6, 6, 6, 5)]:
        assert math.isclose(predicted_ehlich_det(perm), base, rel_tol=1e-12)


def test_predicted_matches_orrick_hunt_top5():
    """Cross-check: top-5 partitions from `enumerate_ehlich_partitions(23)`
    have the same predicted |det| as our `predicted_ehlich_det`."""
    legacy = enumerate_ehlich_partitions(23)[:5]
    for p in legacy:
        ours = predicted_ehlich_det(p.sizes)
        assert math.isclose(ours, p.predicted_abs_det, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 2. partition_complexity
# ---------------------------------------------------------------------------


def test_partition_complexity_zero_for_balanced_n_div_4():
    # n=20, partition (5,5,5,5) is perfectly balanced.
    assert partition_complexity((5, 5, 5, 5)) == 0.0


def test_partition_complexity_strictly_positive_for_unbalanced():
    # n=23 partitions are all unbalanced because n%4 != 0.
    for n in (19, 21, 23):
        for p in enumerate_all_ehlich_partitions(n):
            assert partition_complexity(p.sizes) > 0


def test_partition_complexity_max_skewed_largest():
    # Most-skewed n=23 partition (1,1,1,20) should have larger complexity
    # than any near-balanced one.
    skew = partition_complexity((1, 1, 1, 20))
    balanced = partition_complexity((5, 6, 6, 6))
    assert skew > balanced


# ---------------------------------------------------------------------------
# 3. Cross-n summary
# ---------------------------------------------------------------------------


def test_cross_n_summary_returns_one_row_per_n():
    rows = cross_n_partition_summary([19, 21, 23])
    assert [r.n for r in rows] == [19, 21, 23]


def test_cross_n_summary_n_partitions_matches_enumerate():
    rows = cross_n_partition_summary([19, 21, 23])
    for r in rows:
        assert r.n_partitions == len(enumerate_all_ehlich_partitions(r.n))


def test_cross_n_summary_proven_max_when_known():
    """n=19, 21 are proven optimal — proven_max should be populated."""
    rows = cross_n_partition_summary([19, 21, 23])
    by_n = {r.n: r for r in rows}
    assert by_n[19].proven_max is not None
    assert by_n[21].proven_max is not None
    # n=23 is open: proven_max is None unless we override it.
    assert by_n[23].proven_max is None


def test_cross_n_summary_in_repo_override():
    rows = cross_n_partition_summary([23], in_repo_bests={23: 2_626_567_700_217_856})
    assert rows[0].in_repo_best == 2_626_567_700_217_856
    # Should now have a non-None count of partitions above in_repo_best.
    assert rows[0].n_above_in_repo is not None
    assert rows[0].n_above_in_repo > 0


# ---------------------------------------------------------------------------
# 4. Overlay smoke test (light: 2 fake basins)
# ---------------------------------------------------------------------------


def test_overlay_returns_one_record_per_basin_and_partition():
    from hadamard_maxdet.basin_atlas import BasinEntry
    from hadamard_maxdet.class_tabu import class_signature
    import random

    # Build two trivial random ±1 matrices at n=11 (small to keep test fast).
    n = 11
    rng = random.Random(0)
    mats = []
    for k in range(2):
        m = [[1 if rng.random() < 0.5 else -1 for _ in range(n)] for _ in range(n)]
        mats.append(m)
    basins = [
        BasinEntry(
            label=f"test_{i}",
            matrix=m,
            abs_det=1,  # placeholder; not used by overlay
            signature=class_signature(m),
            source="test",
        )
        for i, m in enumerate(mats)
    ]
    per_basin, per_partition = overlay_basins_on_partitions(
        n, basins, n_restarts=2
    )
    assert len(per_basin) == len(basins)
    assert len(per_partition) == len(enumerate_all_ehlich_partitions(n))
    for ov in per_basin:
        assert 0.0 <= ov.best_score <= 1.0
        assert sum(ov.best_partition) == n
    for lp in per_partition:
        assert 0.0 <= lp.best_score <= 1.0
        assert lp.predicted_ratio > 0.0


# ---------------------------------------------------------------------------
# 5. Plot smoke (only if matplotlib is around)
# ---------------------------------------------------------------------------


def test_plot_satellite_view_writes_file_or_returns_false(tmp_path):
    from hadamard_maxdet.satellite import plot_satellite_view, PartitionLandscape

    parts = enumerate_all_ehlich_partitions(11)
    barba = upper_bound(11)
    landscape = [
        PartitionLandscape(
            partition=p.sizes,
            predicted_abs_det=p.predicted_abs_det,
            predicted_ratio=p.predicted_abs_det / barba,
            complexity=partition_complexity(p.sizes),
            reached=False,
            best_basin_label=None,
            best_score=0.6,
            best_basin_abs_det=0,
        )
        for p in parts
    ]
    out = tmp_path / "satellite_test.png"
    ok = plot_satellite_view(11, landscape, str(out))
    if ok:
        assert out.exists()
        assert out.stat().st_size > 0
    else:
        pytest.skip("matplotlib not available")
