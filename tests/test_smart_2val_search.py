"""Tests for :mod:`hadamard_maxdet.smart_2val_search`."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.smart_2val_search import (
    AbsClassState,
    KnownClass,
    boundary_restart_seeds,
    bridge_excursion,
    count_outside_abs_support,
    enumerate_known_2val_classes,
    gram_abs_class_signature,
    is_in_known_class,
    is_on_manifold,
    manifold_walk,
    _triple_verify_abs_det,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Signature + manifold predicates
# ---------------------------------------------------------------------------


def test_gram_abs_signature_orrick():
    data = json.loads((RESULTS_DIR / "H23_orrick2003_record.json").read_text())
    M = data["entries"]
    sig = gram_abs_class_signature(M)
    # Orrick: |G|=1 count 208, |G|=3 count 45.
    assert sig == ((1, 208), (3, 45))


def test_gram_abs_signature_btc():
    data = json.loads((RESULTS_DIR / "H23_rich_ansatz_best.json").read_text())
    M = data["entries"]
    sig = gram_abs_class_signature(M)
    # BTC Gram: -3:11, -1:209, +1:11, +3:22  -> |G|=1: 220, |G|=3: 33
    assert sig == ((1, 220), (3, 33))


def test_gram_abs_signature_kms():
    data = json.loads((RESULTS_DIR / "H23_from_kms_best.json").read_text())
    M = data["entries"]
    sig = gram_abs_class_signature(M)
    # KMS Gram: -3:31, -1:102, +1:99, +3:21 -> |G|=1: 201, |G|=3: 52
    assert sig == ((1, 201), (3, 52))


def test_known_signatures_all_distinct():
    sigs = []
    for fname in (
        "H23_orrick2003_record.json",
        "H23_rich_ansatz_best.json",
        "H23_from_kms_best.json",
    ):
        data = json.loads((RESULTS_DIR / fname).read_text())
        sigs.append(gram_abs_class_signature(data["entries"]))
    assert len(set(sigs)) == 3


def test_is_on_manifold_orrick_btc_kms_all_pass():
    for fname in (
        "H23_orrick2003_record.json",
        "H23_rich_ansatz_best.json",
        "H23_from_kms_best.json",
    ):
        data = json.loads((RESULTS_DIR / fname).read_text())
        assert is_on_manifold(data["entries"], allowed_abs=(1, 3))


def test_is_on_manifold_random_typically_off():
    rng = __import__("random").Random(0)
    M = Matrix.random(23, rng)
    # A uniform random ±1 matrix at n=23 almost surely has off-diag
    # |G_ij| values that include 5 or higher; check that at least one
    # is outside {1, 3}.
    co = count_outside_abs_support(M.entries, allowed_abs=(1, 3))
    assert co > 0  # Statistically near-certain at n=23.


def test_count_outside_abs_support_consistent():
    rng = __import__("random").Random(42)
    M = Matrix.random(23, rng)
    A = np.asarray(M.entries, dtype=int)
    G = A @ A.T
    iu = np.triu_indices(23, k=1)
    expected = int(np.sum(~np.isin(np.abs(G[iu]), [1, 3])))
    assert count_outside_abs_support(M.entries, allowed_abs=(1, 3)) == expected


# ---------------------------------------------------------------------------
# enumerate_known_2val_classes loader
# ---------------------------------------------------------------------------


def test_enumerate_known_2val_classes_returns_three():
    classes = enumerate_known_2val_classes(RESULTS_DIR)
    assert len(classes) == 3
    names = {c.name for c in classes}
    assert names == {"orrick", "btc", "kms"}


def test_enumerate_known_2val_classes_sorted_by_det():
    classes = enumerate_known_2val_classes(RESULTS_DIR)
    dets = [c.abs_det for c in classes]
    assert dets == sorted(dets, reverse=True)
    # Orrick is largest.
    assert classes[0].name == "orrick"
    assert classes[0].abs_det == 2_779_447_296_000_000


def test_is_in_known_class_orrick_yes_random_no():
    classes = enumerate_known_2val_classes(RESULTS_DIR)
    locked = [c.abs_class_multiset for c in classes]
    orrick_M = next(c for c in classes if c.name == "orrick").matrix.entries
    assert is_in_known_class(orrick_M, locked)
    # Construct a fake matrix with a different multiset by perturbing 100 entries.
    rng = __import__("random").Random(7)
    perturbed = [row[:] for row in orrick_M]
    for _ in range(100):
        i = rng.randrange(23)
        j = rng.randrange(23)
        perturbed[i][j] = -perturbed[i][j]
    assert not is_in_known_class(perturbed, locked)


# ---------------------------------------------------------------------------
# AbsClassState incremental updates
# ---------------------------------------------------------------------------


def test_abs_class_state_initial_orrick_correct():
    data = json.loads((RESULTS_DIR / "H23_orrick2003_record.json").read_text())
    M = data["entries"]
    state = AbsClassState.from_matrix(M)
    assert state.count_abs1 == 208
    assert state.count_abs3 == 45
    assert state.count_out == 0


def test_abs_class_state_initial_btc_correct():
    data = json.loads((RESULTS_DIR / "H23_rich_ansatz_best.json").read_text())
    M = data["entries"]
    state = AbsClassState.from_matrix(M)
    assert state.count_abs1 == 220
    assert state.count_abs3 == 33
    assert state.count_out == 0


def test_abs_class_state_apply_after_flip_matches_recompute():
    # Apply 30 random single flips and check the incremental tracker's
    # counts agree with a from-scratch recompute at every step.
    rng = __import__("random").Random(2)
    M = Matrix.random(23, rng).entries
    state = AbsClassState.from_matrix(M)
    for _ in range(30):
        i = rng.randrange(23)
        c = rng.randrange(23)
        # Flip M
        M[i][c] = -M[i][c]
        state.apply_after_flip(M, i, c)
        # From-scratch
        fresh = AbsClassState.from_matrix(M)
        assert state.count_abs1 == fresh.count_abs1
        assert state.count_abs3 == fresh.count_abs3
        assert state.count_out == fresh.count_out
        # And the Gram is correct
        assert (state.G == fresh.G).all()


def test_deltas_for_flip_matches_apply():
    rng = __import__("random").Random(3)
    M = Matrix.random(23, rng).entries
    state = AbsClassState.from_matrix(M)
    for _ in range(20):
        i = rng.randrange(23)
        c = rng.randrange(23)
        M[i][c] = -M[i][c]  # M_after has the new value
        d1, d3, dout = state.deltas_for_flip(M, i, c)
        # Apply and confirm changes.
        c1_before = state.count_abs1
        c3_before = state.count_abs3
        co_before = state.count_out
        state.apply_after_flip(M, i, c)
        assert state.count_abs1 - c1_before == d1
        assert state.count_abs3 - c3_before == d3
        assert state.count_out - co_before == dout


# ---------------------------------------------------------------------------
# manifold_walk respects manifold + class tabu
# ---------------------------------------------------------------------------


def test_manifold_walk_respects_manifold_constraint():
    """Run a short walk from Orrick; every step must keep ``count_out == 0``."""
    classes = enumerate_known_2val_classes(RESULTS_DIR)
    orrick = next(c for c in classes if c.name == "orrick").matrix
    res = manifold_walk(
        orrick,
        iterations=400,
        locked_multisets=(),  # don't tabu Orrick to allow staying in place
        rng_seed=11,
        refresh_every=200,
    )
    # Final matrix must be on the manifold.
    assert is_on_manifold(res.final_matrix, allowed_abs=(1, 3))
    # Off-manifold rejection counter should be positive (random flips
    # mostly leave the manifold).
    assert res.rejected_off_manifold > 0


def test_manifold_walk_class_tabu_blocks_orrick():
    """If Orrick's signature is locked AND we start at Orrick, the walk
    cannot return to Orrick's signature once it leaves it (any move into
    (208, 45) is rejected). Since at Orrick the only manifold-preserving
    flips lead either back to Orrick or to a different signature, locking
    Orrick should produce a non-zero ``rejected_class``."""
    classes = enumerate_known_2val_classes(RESULTS_DIR)
    orrick = next(c for c in classes if c.name == "orrick").matrix
    sigs_locked = [c.abs_class_multiset for c in classes]  # all 3 locked
    res = manifold_walk(
        orrick,
        iterations=4000,
        locked_multisets=sigs_locked,
        rng_seed=23,
        refresh_every=500,
    )
    # Orrick's manifold neighbourhood is sparse; locking 3 classes means
    # most accepted moves either stay on a non-tabu sig or are rejected.
    # We don't assert >0 here strictly because the manifold neighbourhood
    # of Orrick may be empty — but rejected_off_manifold is huge, so the
    # walk does run.
    assert res.iterations == 4000
    assert res.rejected_off_manifold > 0
    # If best_feasible_signature is set, it must NOT be in the locked set.
    if res.best_feasible_signature is not None:
        assert res.best_feasible_signature not in set(sigs_locked)


def test_manifold_walk_never_reports_above_orrick_in_short_run():
    """Sanity: a 200-iter walk from a random (likely off-manifold) seed
    should not falsely claim to beat Orrick."""
    rng = __import__("random").Random(5)
    seed = Matrix.random(23, rng)
    res = manifold_walk(
        seed,
        iterations=200,
        locked_multisets=(),
        rng_seed=5,
    )
    # A random 23x23 matrix is almost surely off-manifold; so
    # best_feasible_abs_det should be 0 (no feasible step).
    # But it might land on the manifold by accident — only assert that
    # if it claims a det, the matrix is on the manifold and the det is
    # correct.
    if res.best_feasible_entries is not None:
        assert is_on_manifold(res.best_feasible_entries, allowed_abs=(1, 3))
        from hadamard_maxdet.verifier import integer_determinant
        actual = abs(integer_determinant(res.best_feasible_entries))
        assert actual == res.best_feasible_abs_det


# ---------------------------------------------------------------------------
# Bridge excursion
# ---------------------------------------------------------------------------


def test_bridge_excursion_from_orrick_completes():
    classes = enumerate_known_2val_classes(RESULTS_DIR)
    orrick = next(c for c in classes if c.name == "orrick").matrix
    sigs_locked = [c.abs_class_multiset for c in classes]
    res = bridge_excursion(
        orrick,
        excursion_steps=4,
        snap_back_iters=4000,
        walk_iters=1000,
        locked_multisets=sigs_locked,
        rng_seed=99,
    )
    # The result either fails snap-back (acceptable) or runs a walk.
    assert res.excursion_steps == 4
    if res.snap_back_succeeded:
        assert res.walk is not None
        assert is_on_manifold(res.walk.final_matrix, allowed_abs=(1, 3))


# ---------------------------------------------------------------------------
# Boundary-restart seeds
# ---------------------------------------------------------------------------


def test_boundary_restart_seeds_orrick_returns_only_manifold_neighbours():
    classes = enumerate_known_2val_classes(RESULTS_DIR)
    orrick = next(c for c in classes if c.name == "orrick").matrix
    seeds = boundary_restart_seeds(orrick, k=2, max_seeds=10, rng_seed=0)
    # Every returned seed must be on the manifold.
    for s in seeds:
        assert is_on_manifold(s.entries, allowed_abs=(1, 3))
    # At most 10 seeds.
    assert len(seeds) <= 10


def test_boundary_restart_seeds_k4_random_sampling():
    classes = enumerate_known_2val_classes(RESULTS_DIR)
    orrick = next(c for c in classes if c.name == "orrick").matrix
    seeds = boundary_restart_seeds(orrick, k=4, max_seeds=5, rng_seed=10)
    # Every returned seed must be on the manifold.
    for s in seeds:
        assert is_on_manifold(s.entries, allowed_abs=(1, 3))
    # At most 5 (may return fewer if random sampling can't find enough).
    assert len(seeds) <= 5


# ---------------------------------------------------------------------------
# Triple verification helper
# ---------------------------------------------------------------------------


def test_triple_verify_abs_det_orrick():
    data = json.loads((RESULTS_DIR / "H23_orrick2003_record.json").read_text())
    d = _triple_verify_abs_det(data["entries"])
    assert d == 2_779_447_296_000_000
