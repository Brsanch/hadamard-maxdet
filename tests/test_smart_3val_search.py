"""Tests for :mod:`hadamard_maxdet.smart_3val_search`."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.smart_3val_search import (
    AbsClass3State,
    KnownClass3,
    boundary_restart_seeds_3val,
    bridge_excursion_3val,
    count_outside_abs_support_3val,
    enumerate_known_3val_classes,
    gram_abs_class_signature_3val,
    is_in_known_class_3val,
    is_on_manifold_3val,
    manifold_walk_3val,
    _triple_verify_abs_det_3val,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_enumerate_known_3val_classes_loads_basins():
    """Every H23_basin_*.json with |G| ⊆ {1, 3, 5} is loaded; outliers
    (e.g. those containing |G|=7) are skipped."""
    classes = enumerate_known_3val_classes(RESULTS_DIR)
    # We have 24 basins; 21 are exactly 3-valued (1, 18, 22 contain a |G|=7).
    assert len(classes) == 21
    # All names match the basin pattern.
    for c in classes:
        assert c.name.startswith("H23_basin_")


def test_enumerate_known_3val_classes_sorted_by_det():
    classes = enumerate_known_3val_classes(RESULTS_DIR)
    dets = [c.abs_det for c in classes]
    assert dets == sorted(dets, reverse=True)


def test_enumerate_known_3val_classes_signatures_3val_only():
    """Each loaded class has support ⊆ {1, 3, 5}."""
    classes = enumerate_known_3val_classes(RESULTS_DIR)
    for c in classes:
        sigvals = {v for (v, _cnt) in c.abs_class_multiset}
        assert sigvals <= {1, 3, 5}


def test_enumerate_known_3val_classes_abs_det_verifies():
    classes = enumerate_known_3val_classes(RESULTS_DIR)
    from hadamard_maxdet.verifier import integer_determinant
    # Spot-check the first (largest |det|) class against from-scratch det.
    c = classes[0]
    fresh = abs(integer_determinant(c.matrix.entries))
    assert fresh == c.abs_det


# ---------------------------------------------------------------------------
# Signature & manifold predicates
# ---------------------------------------------------------------------------


def test_gram_abs_signature_3val_matches_basin0():
    """basin_0 has counts (1: 196, 3: 53, 5: 4)."""
    data = json.loads((RESULTS_DIR / "H23_basin_0.json").read_text())
    sig = gram_abs_class_signature_3val(data["matrix_entries"])
    assert sig == ((1, 196), (3, 53), (5, 4))


def test_is_on_manifold_3val_basin0_passes():
    data = json.loads((RESULTS_DIR / "H23_basin_0.json").read_text())
    assert is_on_manifold_3val(data["matrix_entries"], allowed_abs=(1, 3, 5))


def test_is_on_manifold_3val_basin1_fails_due_to_7():
    """basin_1 contains a |G|=7 entry, so it is OFF the {1, 3, 5} manifold."""
    data = json.loads((RESULTS_DIR / "H23_basin_1.json").read_text())
    assert not is_on_manifold_3val(data["matrix_entries"], allowed_abs=(1, 3, 5))


def test_is_on_manifold_3val_random_typically_off():
    rng = __import__("random").Random(0)
    M = Matrix.random(23, rng)
    co = count_outside_abs_support_3val(M.entries, allowed_abs=(1, 3, 5))
    # Statistically near-certain at n=23 to have at least one |G|≥7 pair.
    assert co > 0


def test_count_outside_abs_support_3val_consistent():
    rng = __import__("random").Random(42)
    M = Matrix.random(23, rng)
    A = np.asarray(M.entries, dtype=int)
    G = A @ A.T
    iu = np.triu_indices(23, k=1)
    expected = int(np.sum(~np.isin(np.abs(G[iu]), [1, 3, 5])))
    assert count_outside_abs_support_3val(M.entries, allowed_abs=(1, 3, 5)) == expected


# ---------------------------------------------------------------------------
# is_in_known_class_3val
# ---------------------------------------------------------------------------


def test_is_in_known_class_3val_basin0_yes():
    classes = enumerate_known_3val_classes(RESULTS_DIR)
    locked = [c.abs_class_multiset for c in classes]
    data = json.loads((RESULTS_DIR / "H23_basin_0.json").read_text())
    assert is_in_known_class_3val(data["matrix_entries"], locked)


def test_is_in_known_class_3val_perturbed_typically_no():
    classes = enumerate_known_3val_classes(RESULTS_DIR)
    locked = [c.abs_class_multiset for c in classes]
    data = json.loads((RESULTS_DIR / "H23_basin_0.json").read_text())
    rng = __import__("random").Random(7)
    perturbed = [row[:] for row in data["matrix_entries"]]
    for _ in range(150):
        i = rng.randrange(23)
        j = rng.randrange(23)
        perturbed[i][j] = -perturbed[i][j]
    # After 150 random flips the multiset of |G| almost surely no longer
    # matches any of the 21 known basin signatures.
    assert not is_in_known_class_3val(perturbed, locked)


# ---------------------------------------------------------------------------
# AbsClass3State incremental updates
# ---------------------------------------------------------------------------


def test_abs_class3_state_initial_basin0_correct():
    data = json.loads((RESULTS_DIR / "H23_basin_0.json").read_text())
    state = AbsClass3State.from_matrix(data["matrix_entries"])
    assert state.count_abs1 == 196
    assert state.count_abs3 == 53
    assert state.count_abs5 == 4
    assert state.count_out == 0


def test_abs_class3_state_initial_basin1_has_count_out_one():
    """basin_1 has one |G|=7 pair, so on the {1,3,5} manifold count_out=1."""
    data = json.loads((RESULTS_DIR / "H23_basin_1.json").read_text())
    state = AbsClass3State.from_matrix(data["matrix_entries"])
    assert state.count_out == 1


def test_abs_class3_state_apply_after_flip_matches_recompute():
    rng = __import__("random").Random(2)
    M = Matrix.random(23, rng).entries
    state = AbsClass3State.from_matrix(M)
    for _ in range(30):
        i = rng.randrange(23)
        c = rng.randrange(23)
        M[i][c] = -M[i][c]
        state.apply_after_flip(M, i, c)
        fresh = AbsClass3State.from_matrix(M)
        assert state.count_abs1 == fresh.count_abs1
        assert state.count_abs3 == fresh.count_abs3
        assert state.count_abs5 == fresh.count_abs5
        assert state.count_out == fresh.count_out
        assert (state.G == fresh.G).all()


def test_deltas_for_flip_3val_matches_apply():
    rng = __import__("random").Random(3)
    M = Matrix.random(23, rng).entries
    state = AbsClass3State.from_matrix(M)
    for _ in range(20):
        i = rng.randrange(23)
        c = rng.randrange(23)
        M[i][c] = -M[i][c]
        d1, d3, d5, dout = state.deltas_for_flip(M, i, c)
        c1_b = state.count_abs1
        c3_b = state.count_abs3
        c5_b = state.count_abs5
        co_b = state.count_out
        state.apply_after_flip(M, i, c)
        assert state.count_abs1 - c1_b == d1
        assert state.count_abs3 - c3_b == d3
        assert state.count_abs5 - c5_b == d5
        assert state.count_out - co_b == dout


# ---------------------------------------------------------------------------
# manifold_walk_3val
# ---------------------------------------------------------------------------


def test_manifold_walk_3val_respects_manifold_constraint():
    """Run a short walk from basin_0; final matrix must stay on {1,3,5}."""
    classes = enumerate_known_3val_classes(RESULTS_DIR)
    seed_class = classes[0]
    res = manifold_walk_3val(
        seed_class.matrix,
        iterations=400,
        locked_multisets=(),
        rng_seed=11,
        refresh_every=200,
    )
    assert is_on_manifold_3val(res.final_matrix, allowed_abs=(1, 3, 5))
    # Random flips mostly leave {1,3,5}, so off-manifold rejections > 0.
    assert res.rejected_off_manifold > 0


def test_manifold_walk_3val_class_tabu_blocks_all_known():
    """Locking all 21 known signatures should generate non-trivial
    class rejections (or at least some off-manifold rejections)."""
    classes = enumerate_known_3val_classes(RESULTS_DIR)
    seed_class = classes[0]
    sigs_locked = [c.abs_class_multiset for c in classes]
    res = manifold_walk_3val(
        seed_class.matrix,
        iterations=2000,
        locked_multisets=sigs_locked,
        rng_seed=23,
        refresh_every=500,
    )
    assert res.iterations == 2000
    assert res.rejected_off_manifold > 0
    if res.best_feasible_signature is not None:
        assert res.best_feasible_signature not in set(sigs_locked)


def test_manifold_walk_3val_random_seed_safe():
    """A 200-iter walk from a random off-manifold seed must not crash and
    any reported best_feasible matrix must be on the {1,3,5} manifold."""
    rng = __import__("random").Random(5)
    seed = Matrix.random(23, rng)
    res = manifold_walk_3val(
        seed,
        iterations=200,
        locked_multisets=(),
        rng_seed=5,
    )
    if res.best_feasible_entries is not None:
        assert is_on_manifold_3val(
            res.best_feasible_entries, allowed_abs=(1, 3, 5)
        )
        from hadamard_maxdet.verifier import integer_determinant
        actual = abs(integer_determinant(res.best_feasible_entries))
        assert actual == res.best_feasible_abs_det


# ---------------------------------------------------------------------------
# Bridge excursion (3-val)
# ---------------------------------------------------------------------------


def test_bridge_excursion_3val_from_basin_completes():
    classes = enumerate_known_3val_classes(RESULTS_DIR)
    seed_class = classes[0]
    sigs_locked = [c.abs_class_multiset for c in classes]
    res = bridge_excursion_3val(
        seed_class.matrix,
        excursion_steps=4,
        snap_back_iters=4000,
        walk_iters=1000,
        locked_multisets=sigs_locked,
        rng_seed=99,
    )
    assert res.excursion_steps == 4
    if res.snap_back_succeeded:
        assert res.walk is not None
        assert is_on_manifold_3val(res.walk.final_matrix, allowed_abs=(1, 3, 5))


# ---------------------------------------------------------------------------
# Boundary restart seeds (3-val)
# ---------------------------------------------------------------------------


def test_boundary_restart_seeds_3val_returns_only_manifold_neighbours():
    classes = enumerate_known_3val_classes(RESULTS_DIR)
    seed_class = classes[0]
    seeds = boundary_restart_seeds_3val(
        seed_class.matrix, k=2, max_seeds=10, rng_seed=0
    )
    for s in seeds:
        assert is_on_manifold_3val(s.entries, allowed_abs=(1, 3, 5))
    assert len(seeds) <= 10


def test_boundary_restart_seeds_3val_k4_random_sampling():
    classes = enumerate_known_3val_classes(RESULTS_DIR)
    seed_class = classes[0]
    seeds = boundary_restart_seeds_3val(
        seed_class.matrix, k=4, max_seeds=5, rng_seed=10
    )
    for s in seeds:
        assert is_on_manifold_3val(s.entries, allowed_abs=(1, 3, 5))
    assert len(seeds) <= 5


# ---------------------------------------------------------------------------
# Triple verification helper
# ---------------------------------------------------------------------------


def test_triple_verify_abs_det_3val_basin0():
    data = json.loads((RESULTS_DIR / "H23_basin_0.json").read_text())
    d = _triple_verify_abs_det_3val(data["matrix_entries"])
    assert d == int(data["abs_det"])
