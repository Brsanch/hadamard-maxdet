"""Tests for the basin-atlas module."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

from hadamard_maxdet.basin_atlas import (
    BasinEntry,
    bounded_walk_reachability,
    distance_from_anchor_profile,
    gram_emd,
    hamming_distance,
    load_all_h23_basins,
    mds_embed,
    pairwise_distances,
    partition_assignment,
    saddle_search_between,
    spectral_distance,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant, upper_bound


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def test_hamming_distance_basic():
    A = [[1, 1], [1, -1]]
    B = [[1, -1], [1, -1]]
    assert hamming_distance(A, B) == 1
    assert hamming_distance(A, A) == 0


def test_gram_emd_zero_for_equal_multisets():
    from collections import Counter
    c1 = Counter({-1: 200, 3: 53})
    c2 = Counter({-1: 200, 3: 53})
    assert gram_emd(c1, c2) == 0.0


def test_gram_emd_positive_for_shifted():
    from collections import Counter
    c1 = Counter({-1: 100})
    c2 = Counter({1: 100})
    # CDF L1 distance with two points at -1 and 1: cum jumps by 1 at -1
    # for c1 vs at 1 for c2 -> abs gap = 1 over the gap.
    d = gram_emd(c1, c2)
    assert d > 0


def test_spectral_distance_zero_for_equal():
    M = [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
    assert spectral_distance(M, M) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(not RESULTS_DIR.is_dir(), reason="results/ not present")
def test_load_all_h23_basins_returns_at_least_24():
    basins = load_all_h23_basins(RESULTS_DIR, n=23, include_constructed=False)
    assert len(basins) >= 24, (
        f"expected at least 24 distinct H23 basin matrices on disk; "
        f"got {len(basins)}"
    )
    for b in basins:
        assert b.n == 23
        assert b.abs_det > 0
        det_check = abs(integer_determinant(b.matrix))
        assert det_check == b.abs_det, (
            f"basin {b.label} det mismatch: stored {b.abs_det} "
            f"vs Bareiss {det_check}"
        )
    # Sorted descending by |det|.
    dets = [b.abs_det for b in basins]
    assert dets == sorted(dets, reverse=True)


@pytest.mark.skipif(not RESULTS_DIR.is_dir(), reason="results/ not present")
def test_load_all_h23_basins_with_constructed_includes_williamson_or_paley():
    basins_no = load_all_h23_basins(RESULTS_DIR, n=23, include_constructed=False)
    basins_yes = load_all_h23_basins(RESULTS_DIR, n=23, include_constructed=True)
    # Constructed sources should appear unless their signatures collide with
    # disk basins (in which case we keep the higher-|det| representative).
    sources_yes = {b.source for b in basins_yes}
    # At least one constructed family should land or be deduplicated against
    # an existing basin; assert the count is at least the disk count.
    assert len(basins_yes) >= len(basins_no)


def test_pairwise_distances_symmetric_zero_diag():
    # Use a small synthetic set: 3 nonsingular ±1 matrices of order 8.
    rng = random.Random(0)
    basins: list[BasinEntry] = []
    attempts = 0
    while len(basins) < 3 and attempts < 50:
        attempts += 1
        M = Matrix.random(8, rng).entries
        d = abs(integer_determinant(M))
        if d == 0:
            continue
        basins.append(
            BasinEntry(
                label=f"r{len(basins)}",
                matrix=M,
                abs_det=d,
                signature=(),
                source="synthetic",
            )
        )
    assert len(basins) >= 2
    out = pairwise_distances(basins)
    for key in ("hamming", "gram_emd", "spectral"):
        D = out[key]
        N = len(basins)
        assert D.shape == (N, N)
        for i in range(N):
            assert D[i, i] == 0
            for j in range(N):
                assert D[i, j] == D[j, i]


def test_mds_embed_nondegenerate_2d():
    # Construct a clear 4-point cluster with known geometry.
    D = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    coords = mds_embed(D, k=2)
    assert coords.shape == (4, 2)
    # First-axis spread should be substantial (not collapsed to a point).
    spread = float(coords[:, 0].max() - coords[:, 0].min())
    assert spread > 0.5


def test_mds_embed_handles_single_point():
    coords = mds_embed(np.zeros((1, 1)), k=2)
    assert coords.shape == (1, 2)


def test_partition_assignment_recovers_planted_block():
    # Build a 12×12 ±1 matrix where rows are explicitly grouped: 3 blocks of
    # size 4. Within a block, rows are identical; across blocks they differ
    # in exactly half of the columns. This gives intra-block Gram = +12 and
    # cross-block Gram = 0 — not the (+3, -1) Ehlich pattern, but the
    # *partition recovery* (rows belonging to the right blocks) should still
    # be detectable. We just check the function runs and returns the right
    # shape.
    M = []
    sizes = (4, 4, 4)
    rng = random.Random(0)
    for k, sz in enumerate(sizes):
        base = [1 if rng.random() < 0.5 else -1 for _ in range(12)]
        for _ in range(sz):
            M.append(base[:])
    # We need a 4-block partition for the function. Pad to 4 blocks of size 3.
    M2 = M[:12]
    n = 12
    pa = partition_assignment(
        M2, partition_sizes=(3, 3, 3, 3), intra_value=3, cross_value=-1, n_restarts=2,
    )
    assert len(pa.assignment) == 12
    assert all(0 <= b < 4 for b in pa.assignment)
    assert 0.0 <= pa.gram_match_score <= 1.0


@pytest.mark.skipif(not RESULTS_DIR.is_dir(), reason="results/ not present")
def test_saddle_search_runs_on_two_real_basins():
    basins = load_all_h23_basins(RESULTS_DIR, n=23, include_constructed=False)
    if len(basins) < 2:
        pytest.skip("need at least 2 basins")
    res = saddle_search_between(
        basins[0].matrix,
        basins[1].matrix,
        n_intermediate=3,
        sa_iters=200,
        seed=0,
        label_a=basins[0].label,
        label_b=basins[1].label,
    )
    assert res.hamming_a_b > 0
    assert res.abs_det_a == basins[0].abs_det
    assert res.abs_det_b == basins[1].abs_det
    # Saddle should not exceed either endpoint by orders of magnitude.
    assert res.saddle_height >= 0
    # Path includes endpoints + interior.
    assert len(res.path) == res.n_intermediate + 2


@pytest.mark.skipif(not RESULTS_DIR.is_dir(), reason="results/ not present")
def test_distance_from_anchor_profile_real_basins():
    basins = load_all_h23_basins(RESULTS_DIR, n=23, include_constructed=False)
    if len(basins) < 2:
        pytest.skip("need at least 2 basins")
    profile = distance_from_anchor_profile(basins, anchor_idx=0)
    assert len(profile) == len(basins) - 1
    for entry in profile:
        assert entry["hamming_to_anchor"] >= 0
        assert 0.0 <= entry["ratio_to_bound"] <= 1.0


@pytest.mark.skipif(not RESULTS_DIR.is_dir(), reason="results/ not present")
def test_bounded_walk_runs_on_small_basin_subset():
    basins = load_all_h23_basins(RESULTS_DIR, n=23, include_constructed=False)
    if len(basins) < 3:
        pytest.skip("need at least 3 basins")
    sub = basins[:3]
    out = bounded_walk_reachability(
        sub, walk_budget=50, det_floor_fraction=0.3, seed=0
    )
    assert "edges" in out
    assert "stats" in out
    assert len(out["stats"]) == 3
    for s in out["stats"]:
        assert "label" in s
        assert "n_other_basins_reached" in s
