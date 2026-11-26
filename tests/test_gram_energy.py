"""Tests for the Gram-asymmetry energy module."""

from __future__ import annotations

import random

import pytest

from hadamard_maxdet.gram_energy import (
    apply_flip_update_gram,
    gram_energy,
    gram_energy_delta,
    gram_energy_from_G,
    gram_offdiag_matrix,
    sa_gram_energy,
    sa_targeted,
    targeted_plus3_flip,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


def _random_matrix(n: int, rng: random.Random) -> list[list[int]]:
    return [[rng.choice([-1, 1]) for _ in range(n)] for _ in range(n)]


# ----------------------------------------------------------------------------
# gram_energy correctness
# ----------------------------------------------------------------------------


def test_gram_energy_naive_vs_from_G():
    rng = random.Random(0)
    for n in (3, 4, 6, 11, 19):
        for trial in range(5):
            M = _random_matrix(n, rng)
            e_direct = gram_energy(M)
            G = gram_offdiag_matrix(M)
            e_via_G = gram_energy_from_G(G)
            assert e_direct == e_via_G, (
                f"n={n} trial={trial}: direct={e_direct} vs from_G={e_via_G}"
            )


def test_gram_energy_known_zero_on_hadamard():
    """Sylvester H_4 has all off-diagonals at 0; E_gram counts max(0, 0+1)^2 = 1
    for each pair, so for n=4 we get C(4,2)=6 pairs of value 0 contributing 1
    each, total 6. Wait — Hadamard rows are orthogonal, <v_i,v_j>=0 ⟹ max(0,1)=1.
    So E_gram(Hadamard)=C(n,2). This is a sanity check on the formula."""
    H = Matrix.sylvester(4)
    e = gram_energy(H.entries)
    n = 4
    assert e == n * (n - 1) // 2  # 6


def test_gram_offdiag_matrix_symmetric():
    rng = random.Random(7)
    n = 8
    M = _random_matrix(n, rng)
    G = gram_offdiag_matrix(M)
    for i in range(n):
        assert G[i][i] == 0
        for j in range(n):
            assert G[i][j] == G[j][i]
    # Diagonal values would be n; we deliberately set them to 0.


# ----------------------------------------------------------------------------
# gram_energy_delta correctness — load-bearing optimisation.
# ----------------------------------------------------------------------------


def test_gram_energy_delta_matches_recompute_small():
    rng = random.Random(1)
    for n in (3, 4, 6, 11, 19):
        M = _random_matrix(n, rng)
        G = gram_offdiag_matrix(M)
        before = gram_energy(M)
        for trial in range(20):
            i = rng.randrange(n)
            k = rng.randrange(n)
            predicted_delta = gram_energy_delta(M, G, i, k)
            # Apply, recompute, compare.
            apply_flip_update_gram(M, G, i, k)
            after = gram_energy(M)
            actual_delta = after - before
            assert predicted_delta == actual_delta, (
                f"n={n} trial={trial} flip ({i},{k}): "
                f"predicted {predicted_delta} ≠ actual {actual_delta}"
            )
            # Verify the cached Gram matches a fresh recompute.
            G_fresh = gram_offdiag_matrix(M)
            for a in range(n):
                for b in range(n):
                    assert G[a][b] == G_fresh[a][b], (
                        f"Gram cache stale at ({a},{b}) after flip ({i},{k})"
                    )
            before = after


def test_gram_energy_delta_long_walk_n11():
    """1000 random flips on n=11, delta exactly tracks the energy each step."""
    rng = random.Random(42)
    n = 11
    M = _random_matrix(n, rng)
    G = gram_offdiag_matrix(M)
    e = gram_energy(M)
    for step in range(1000):
        i = rng.randrange(n)
        k = rng.randrange(n)
        d = gram_energy_delta(M, G, i, k)
        apply_flip_update_gram(M, G, i, k)
        e += d
        if step % 100 == 0:
            e_fresh = gram_energy(M)
            assert e == e_fresh, f"step {step}: tracked {e} ≠ fresh {e_fresh}"
    assert e == gram_energy(M)


# ----------------------------------------------------------------------------
# sa_gram_energy sanity (small n, just check it reduces E_gram).
# ----------------------------------------------------------------------------


def test_sa_gram_energy_reduces_energy_n8():
    rng = random.Random(123)
    n = 8
    M = Matrix(n=n, entries=_random_matrix(n, rng))
    starting_e = gram_energy(M.entries)
    res = sa_gram_energy(
        initial=M,
        iterations=5000,
        t_start=20.0,
        t_end=0.1,
        seed=0,
    )
    assert res.best_e_gram <= starting_e, (
        f"SA failed to reduce E_gram (started {starting_e}, ended {res.best_e_gram})"
    )
    # Best matrix |det| is computed by exact Bareiss; ensure it matches.
    expected_det = abs(integer_determinant(res.matrix.entries))
    assert res.abs_det == expected_det


# ----------------------------------------------------------------------------
# targeted_plus3_flip correctness
# ----------------------------------------------------------------------------


def test_targeted_plus3_flip_picks_worst_pair_agree_column():
    """The proposer must pick (i, k) such that:
    (a) (i, j) for some j is at the max Gram value, and
    (b) M[i][k] == M[j][k] (so flipping reduces <v_i, v_j>)."""
    rng = random.Random(0)
    n = 11
    M = _random_matrix(n, rng)
    G = gram_offdiag_matrix(M)
    # Compute the max off-diagonal value to know the target.
    max_g = max(G[i][j] for i in range(n) for j in range(i + 1, n))

    for trial in range(50):
        pick = targeted_plus3_flip(M, G, rng)
        assert pick is not None, "targeted_plus3_flip returned None on n=11"
        i, k = pick
        # There must be some j != i with G[i][j] == max_g and M[i][k] == M[j][k].
        ok = False
        for j in range(n):
            if j == i:
                continue
            if G[i][j] == max_g and M[i][k] == M[j][k]:
                ok = True
                break
        assert ok, (
            f"trial {trial}: pick ({i},{k}) does not target a worst pair "
            f"with an agree column (max_g={max_g})"
        )


def test_targeted_proposer_actually_reduces_worst_pair_inner_product():
    """Sanity: applying the targeted flip lowers some max-pair inner product."""
    rng = random.Random(1)
    n = 11
    M = _random_matrix(n, rng)
    G = gram_offdiag_matrix(M)
    max_g_before = max(G[i][j] for i in range(n) for j in range(i + 1, n))
    pick = targeted_plus3_flip(M, G, rng)
    assert pick is not None
    i, k = pick
    # Identify some j at max_g that agrees with row i at column k.
    j_target = next(
        j for j in range(n) if j != i and G[i][j] == max_g_before and M[i][k] == M[j][k]
    )
    apply_flip_update_gram(M, G, i, k)
    # After the flip, <v_i, v_{j_target}> drops by exactly 2.
    assert G[i][j_target] == max_g_before - 2


# ----------------------------------------------------------------------------
# sa_targeted smoke test (does not improve guaranteed; just runs cleanly).
# ----------------------------------------------------------------------------


def test_sa_targeted_runs_and_verifies_n8():
    rng = random.Random(7)
    n = 8
    # Build a non-singular start.
    while True:
        M_entries = _random_matrix(n, rng)
        if integer_determinant(M_entries) != 0:
            break
    M = Matrix(n=n, entries=M_entries)
    res = sa_targeted(
        initial=M,
        iterations=2000,
        t_start=1.0,
        t_end=0.01,
        seed=0,
        targeted_fraction=0.7,
    )
    # |det| reported must equal exact Bareiss on the returned matrix.
    expected = abs(integer_determinant(res.matrix.entries))
    assert res.abs_det == expected
    assert res.targeted_proposed > 0
    assert res.uniform_proposed > 0
