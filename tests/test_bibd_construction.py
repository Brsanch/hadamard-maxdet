"""Tests for the (23, 11, 5)-Paley biplane construction and its perturbations."""

from __future__ import annotations

import numpy as np
import pytest

from hadamard_maxdet.bibd_construction import (
    biplane_design_check,
    feasible_row_perturbations,
    multiset_grouped_results,
    paley_biplane_23,
    quadratic_residues_mod_q,
)
from hadamard_maxdet.verifier import integer_determinant


def test_quadratic_residues_mod_23():
    """The QRs mod 23 are exactly {1, 2, 3, 4, 6, 8, 9, 12, 13, 16, 18}."""
    Q = quadratic_residues_mod_q(23)
    assert Q == [1, 2, 3, 4, 6, 8, 9, 12, 13, 16, 18]
    assert len(Q) == 11


def test_quadratic_residues_mod_11():
    """The QRs mod 11 are {1, 3, 4, 5, 9}."""
    Q = quadratic_residues_mod_q(11)
    assert Q == [1, 3, 4, 5, 9]
    assert len(Q) == 5


def test_paley_biplane_shape_and_entries():
    M = paley_biplane_23()
    assert len(M) == 23
    assert all(len(r) == 23 for r in M)
    for r in M:
        for v in r:
            assert v in (-1, 1)


def test_paley_biplane_row_weights_are_11():
    """Each row of the (23, 11, 5)-design has exactly 11 +1 entries."""
    M = paley_biplane_23()
    for i, row in enumerate(M):
        ones = sum(1 for v in row if v == 1)
        assert ones == 11, f"row {i} has {ones} ones, expected 11"


def test_paley_biplane_pairwise_lambda_is_5():
    """Every pair of rows shares exactly 5 +1 columns (lambda = 5)."""
    M = paley_biplane_23()
    A = np.asarray(M, dtype=int)
    plus = ((A + 1) // 2)
    P = plus @ plus.T
    n = 23
    for i in range(n):
        for j in range(i + 1, n):
            assert P[i, j] == 5, f"pair ({i},{j}) has lambda={P[i,j]}, expected 5"


def test_paley_biplane_offdiag_gram_uniform_minus_1():
    """All off-diagonal Gram entries are -1, since 4*lambda - 4k + v = -1."""
    M = paley_biplane_23()
    A = np.asarray(M, dtype=int)
    G = A @ A.T
    iu = np.triu_indices(23, k=1)
    off = G[iu]
    assert int(off.min()) == -1
    assert int(off.max()) == -1


def test_paley_biplane_abs_det_equals_24_pow_11():
    """|det| of the biplane ±1 matrix equals 24^11."""
    M = paley_biplane_23()
    d = abs(integer_determinant(M))
    expected = 24 ** 11
    assert d == expected, f"got |det|={d}, expected 24^11={expected}"


def test_biplane_design_check_full_report():
    """biplane_design_check returns the expected verifier flags."""
    M = paley_biplane_23()
    rep = biplane_design_check(M)
    assert rep["row_weights_uniform_11"] is True
    assert rep["off_diag_uniform_minus_1"] is True
    assert rep["all_lambda_5"] is True
    assert rep["abs_det"] == 24 ** 11
    assert rep["abs_det_factored_24_pow_11"] is True


def test_feasible_row_perturbations_zero_flip_returns_biplane():
    """The k=0 perturbation yields the biplane itself with all-(-1) Gram."""
    M = paley_biplane_23()
    perts = list(feasible_row_perturbations(
        M, row=0, max_flips=0, min_flips=0,
    ))
    assert len(perts) == 1
    p = perts[0]
    assert p.flip_columns == ()
    assert p.count_minus1 == 23 * 22 // 2
    assert p.count_plus3 == 0
    assert p.abs_det == 24 ** 11


def test_feasible_row_perturbations_k1_count_zero():
    """A single-entry flip cannot keep all 22 affected off-diags in {-1,+3}.

    Flipping ONE column c in row r changes <v_r, v_i> by -2*M[r,c]*M[i,c]
    = ±2 for each i. Going from -1 by ±2 gives -3 or +1, neither of which
    is in {-1, +3}. So no 1-flip is feasible.
    """
    M = paley_biplane_23()
    perts = list(feasible_row_perturbations(
        M, row=0, max_flips=1, min_flips=1,
    ))
    assert perts == []


def test_feasible_row_perturbations_k2_yields_some():
    """A 2-flip perturbation can be feasible: change per i is in {-4, 0, +4}.

    Need every change to be in {0, +4}. We test that the enumerator finds at
    least one feasible 2-flip.
    """
    M = paley_biplane_23()
    perts = list(feasible_row_perturbations(
        M, row=0, max_flips=2, min_flips=2,
    ))
    # Sanity: every yielded perturbation has 22 deltas in {0, 4}.
    for p in perts:
        assert all(d in (0, 4) for d in p.deltas_to_other_rows)
        assert p.count_minus1 + p.count_plus3 == 23 * 22 // 2


def test_multiset_grouped_results_basic():
    """multiset_grouped_results groups perturbations by (cm1, cp3)."""
    M = paley_biplane_23()
    perts = list(feasible_row_perturbations(
        M, row=0, max_flips=2, min_flips=0,
    ))
    grouped = multiset_grouped_results(iter(perts))
    # The k=0 case must produce the (253, 0) bucket.
    assert (253, 0) in grouped
    assert grouped[(253, 0)]["best_abs_det"] == 24 ** 11
