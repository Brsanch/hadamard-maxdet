"""Tests for :mod:`hadamard_maxdet.targeted_bridge`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hadamard_maxdet.smart_2val_search import (
    enumerate_known_2val_classes,
    gram_abs_class_signature,
    is_on_manifold,
)
from hadamard_maxdet.smart_3val_search import (
    enumerate_known_3val_classes,
    is_on_manifold_3val,
)
from hadamard_maxdet.targeted_bridge import (
    BridgeAggregate,
    BridgeMove,
    NearTwoValMatrix,
    ORRICK_DET,
    cross_reference,
    enumerate_bridges,
    find_snap_to_2val_moves,
    load_known_2val_signatures,
    load_near_2val_matrices_from_3val_run,
    triple_verify_abs_det,
)
from hadamard_maxdet.verifier import integer_determinant


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# find_snap_to_2val_moves on a synthetic case
# ---------------------------------------------------------------------------


def _make_synthetic_near_2val_from_orrick() -> tuple[list[list[int]], tuple[int, int]]:
    """Take Orrick's matrix (2-val), flip a single entry that pushes a few
    Gram pairs to |G|=5, and return the resulting near-2-val matrix together
    with the flip position. By construction, the *reverse* flip snaps it
    back to 2-val.
    """
    canonical = enumerate_known_2val_classes(RESULTS_DIR)
    orrick = next(k for k in canonical if k.name == "orrick")
    M0 = [row[:] for row in orrick.matrix.entries]
    n = len(M0)
    # Try positions until we find one that creates >=1 |G|=5 pair without
    # destroying the manifold beyond {1, 3, 5}.
    for i in range(n):
        for c in range(n):
            M_perturbed = [row[:] for row in M0]
            M_perturbed[i][c] = -M_perturbed[i][c]
            if is_on_manifold_3val(M_perturbed, allowed_abs=(1, 3, 5)):
                # Was 2-val. Flipped → check it's now genuinely 3-val (count_5 > 0).
                from hadamard_maxdet.smart_3val_search import AbsClass3State
                st = AbsClass3State.from_matrix(M_perturbed)
                if st.count_abs5 >= 1:
                    return M_perturbed, (i, c)
    pytest.fail("Could not construct a synthetic near-2-val from Orrick.")


def test_find_snap_to_2val_moves_finds_inverse_flip_on_orrick_perturbation():
    """If we perturbed Orrick by one flip and ended up near-2-val, then
    flipping the same entry back must be a valid snap. The function should
    discover that single-entry flip as one of its solutions."""
    M_perturbed, flipped_pos = _make_synthetic_near_2val_from_orrick()
    moves = find_snap_to_2val_moves(M_perturbed, max_k=1, n_max_solutions=200)
    # At least one single-entry flip must work — at minimum, the inverse flip.
    assert len(moves) >= 1, "Expected at least one snap-to-2-val flip"
    flip_sets = {fs for fs, _sig in moves}
    assert (flipped_pos,) in flip_sets, (
        f"Inverse flip {flipped_pos} not among {len(flip_sets)} found single-flip snaps"
    )


def test_find_snap_to_2val_moves_returns_2val_destinations():
    """Every flip-set returned by find_snap_to_2val_moves must produce a
    matrix that lies on the |G| ⊆ {1, 3} manifold (i.e. is 2-val)."""
    M_perturbed, _ = _make_synthetic_near_2val_from_orrick()
    moves = find_snap_to_2val_moves(M_perturbed, max_k=2, n_max_solutions=20)
    assert moves, "Expected at least one snap-to-2-val flip-set"
    for flip_positions, dest_sig in moves:
        Mp = [row[:] for row in M_perturbed]
        for i, c in flip_positions:
            Mp[i][c] = -Mp[i][c]
        assert is_on_manifold(Mp, allowed_abs=(1, 3)), (
            f"flip-set {flip_positions} does NOT land on |G| ⊆ {{1, 3}}"
        )
        # Sig matches what's reported.
        actual_sig = gram_abs_class_signature(Mp)
        assert actual_sig == dest_sig, (
            f"Reported dest sig {dest_sig} != actual {actual_sig}"
        )


def test_find_snap_to_2val_moves_returns_empty_when_already_2val():
    """If M is already on the |G| ⊆ {1, 3} submanifold (count_5 = 0),
    the function returns the trivial empty flip-set as the unique solution."""
    canonical = enumerate_known_2val_classes(RESULTS_DIR)
    orrick = next(k for k in canonical if k.name == "orrick")
    moves = find_snap_to_2val_moves(orrick.matrix.entries, max_k=1)
    assert len(moves) == 1
    flip_set, sig = moves[0]
    assert flip_set == ()
    assert sig == orrick.abs_class_multiset


def test_find_snap_to_2val_moves_off_manifold_source_returns_empty():
    """A matrix not even on the 3-val manifold (has |G|=7 entry) cannot be
    snapped — the function returns an empty list."""
    # basin_1 has |G|=7. Load directly.
    with (RESULTS_DIR / "H23_basin_1.json").open() as f:
        d = json.load(f)
    M = [list(r) for r in d["matrix_entries"]]
    assert not is_on_manifold_3val(M)
    moves = find_snap_to_2val_moves(M, max_k=1)
    assert moves == []


# ---------------------------------------------------------------------------
# Loader sanity
# ---------------------------------------------------------------------------


def test_load_near_2val_returns_well_formed_records():
    """Loader returns NearTwoValMatrix records whose own signature really
    has count_5 in the requested range."""
    json_path = RESULTS_DIR / "smart_3val_search.json"
    if not json_path.exists():
        pytest.skip("smart_3val_search.json not present in this checkout")
    near = load_near_2val_matrices_from_3val_run(json_path, count_5_max=2)
    assert near, "Expected at least one near-2-val matrix in the 3-val run"
    for rec in near:
        assert rec.count_5 in (1, 2)
        # signature really has the claimed count_5
        sig_dict = {p[0]: p[1] for p in rec.signature}
        assert sig_dict.get(5, 0) == rec.count_5
        # matrix is 23×23 ±1
        assert len(rec.matrix) == 23
        for row in rec.matrix:
            assert len(row) == 23
            for v in row:
                assert v in (-1, 1)
        # abs_det matches
        actual = abs(integer_determinant(rec.matrix))
        assert actual == rec.abs_det, (
            f"loader abs_det {rec.abs_det} != recomputed {actual}"
        )


# ---------------------------------------------------------------------------
# enumerate_bridges + cross_reference smoke
# ---------------------------------------------------------------------------


def test_enumerate_bridges_on_synthetic_orrick_perturbation():
    """Pack a NearTwoValMatrix from a single Orrick perturbation, and check
    enumerate_bridges discovers ≥1 bridge whose destination is Orrick's
    own (208, 45) signature."""
    M_perturbed, flipped_pos = _make_synthetic_near_2val_from_orrick()
    canonical = enumerate_known_2val_classes(RESULTS_DIR)
    orrick = next(k for k in canonical if k.name == "orrick")

    from hadamard_maxdet.smart_3val_search import gram_abs_class_signature_3val
    src_sig = gram_abs_class_signature_3val(M_perturbed)
    sig_dict = {p[0]: p[1] for p in src_sig}
    src = NearTwoValMatrix(
        job_name="synthetic.orrick.perturbed",
        signature=src_sig,
        abs_det=abs(integer_determinant(M_perturbed)),
        count_5=int(sig_dict.get(5, 0)),
        matrix=M_perturbed,
    )
    bridges, by_dest = enumerate_bridges([src], max_k=1, n_max_solutions_per_source=200)
    assert bridges, "Expected at least one bridge from synthetic source"
    # Orrick's signature must be among the destinations.
    assert orrick.abs_class_multiset in by_dest
    agg = by_dest[orrick.abs_class_multiset]
    # The Orrick destination must hit Orrick's exact |det|.
    assert agg.best_abs_det == orrick.abs_det, (
        f"agg.best_abs_det {agg.best_abs_det} != orrick.abs_det {orrick.abs_det}"
    )


def test_cross_reference_splits_known_and_new():
    """Crafted check: a destination matching a canonical signature goes to
    'already', any other goes to 'new'."""
    canonical = enumerate_known_2val_classes(RESULTS_DIR)
    orrick = next(k for k in canonical if k.name == "orrick")
    fake_new_sig = ((1, 999), (3, 999))  # impossible at H23, just a placeholder
    by_dest = {
        orrick.abs_class_multiset: BridgeAggregate(
            destination_signature=orrick.abs_class_multiset,
            best_abs_det=orrick.abs_det,
            best_matrix=[row[:] for row in orrick.matrix.entries],
            n_bridges=1,
            sample_source_jobs=["x"],
        ),
        fake_new_sig: BridgeAggregate(
            destination_signature=fake_new_sig,
            best_abs_det=42,
            best_matrix=[],
            n_bridges=1,
            sample_source_jobs=["y"],
        ),
    }
    known = load_known_2val_signatures(
        results_dir=RESULTS_DIR,
        smart_2val_path=RESULTS_DIR / "smart_2val_search.json",
    )
    already, new = cross_reference(by_dest, known)
    assert orrick.abs_class_multiset in already
    assert fake_new_sig in new


def test_load_known_2val_signatures_contains_canonicals():
    """Sanity: the canonical Orrick / BTC / KMS signatures are in the
    known set."""
    canonical = enumerate_known_2val_classes(RESULTS_DIR)
    known = load_known_2val_signatures(
        results_dir=RESULTS_DIR,
        smart_2val_path=RESULTS_DIR / "smart_2val_search.json",
    )
    for kc in canonical:
        assert kc.abs_class_multiset in known


def test_triple_verify_orrick_matches_record():
    """triple_verify_abs_det on Orrick's matrix returns the published record."""
    canonical = enumerate_known_2val_classes(RESULTS_DIR)
    orrick = next(k for k in canonical if k.name == "orrick")
    d = triple_verify_abs_det(orrick.matrix.entries)
    assert d == ORRICK_DET == orrick.abs_det
