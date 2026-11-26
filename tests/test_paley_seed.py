"""Tests for the Paley-deletion seed module.

Mathematical note. For any Hadamard matrix ``H`` of order ``m``,
``adj(H) = det(H) · H^{-1}`` and (since ``H^T = m · H^{-1}``) every cofactor
satisfies ``|cof(i,j)| = m^{m/2}/m = m^{m/2 - 1}``. The signed determinant
of the (m-1)×(m-1) submatrix obtained by deleting row ``i`` and column ``j``
of ``H`` equals the cofactor up to a sign, so **every Paley/Sylvester
deletion of an m×m Hadamard matrix has the same |det| =
m^{(m-2)/2}**.

Concretely:
  - H8 deletions: every 7×7 sub has |det| = 8^3 = **512** (below the
    proven H7 max 576 attained by an inequivalent ±1 matrix).
  - H12 deletions: every 11×11 sub has |det| = 12^5 = **248832** (below
    the proven H11 max 327680 and below the bordered-circulant ceiling
    281600).
  - H24 deletions: every 23×23 sub has |det| = 24^11 ≈ 1.52e15 (below
    the bordered-circulant ceiling 2.42e15 at n=23).

So Hadamard cofactors are *strong starting seeds* (≥75% of every known
ceiling at their order) but are not themselves the maximum. They give
parallel tempering a high-determinant launchpad far above what random
±1 starts provide.
"""

from __future__ import annotations

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.paley_seed import paley_deletions, top_paley_seeds
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Structural correctness
# ---------------------------------------------------------------------------


def test_paley_deletions_produces_correct_count_at_n7():
    """H8 → 8 × 8 = 64 deletions."""
    seeds = list(paley_deletions(7))
    assert len(seeds) == 64
    for i, j, mat, _ in seeds:
        assert 0 <= i < 8 and 0 <= j < 8
        assert mat.n == 7
        assert len(mat.entries) == 7
        for row in mat.entries:
            assert len(row) == 7
            for v in row:
                assert v in (-1, 1)


def test_paley_deletions_produces_correct_count_at_n11():
    """H12 → 12 × 12 = 144 deletions."""
    seeds = list(paley_deletions(11))
    assert len(seeds) == 144
    for _, _, mat, _ in seeds:
        assert mat.n == 11


def test_paley_deletions_produces_correct_count_at_n23():
    """H24 → 24 × 24 = 576 deletions."""
    seeds = list(paley_deletions(23))
    assert len(seeds) == 576
    for _, _, mat, _ in seeds:
        assert mat.n == 23


# ---------------------------------------------------------------------------
# Cofactor identity at orders where the parent is Hadamard
# ---------------------------------------------------------------------------


def test_paley_deletions_at_n7_all_yield_cofactor_512():
    """Every 7×7 cofactor of Paley-I H8 has |det| = 8^3 = 512.

    This is the cofactor identity: |cof(i,j)| = |det(H)|/m = 8^4/8 = 512.
    The proven H7 max is 576, attained by an inequivalent ±1 matrix
    that is *not* a deletion of any 8×8 Hadamard matrix; so 512 (not
    576) is what deletion produces here.
    """
    seeds = list(paley_deletions(7))
    abs_dets = sorted({abs(d) for _, _, _, d in seeds})
    assert abs_dets == [512], f"unexpected |det| set: {abs_dets}"


def test_paley_deletions_at_n11_all_yield_cofactor_248832():
    """Every 11×11 cofactor of Paley-I H12 has |det| = 12^5 = 248832.

    This is below the bordered-circulant ceiling 281600 *and* the proven
    H11 max 327680 — but it is a strong starting seed (≥75% of true max)
    that gives SA / replica exchange a high-|det| launchpad. The literal
    spec wanted 'a deletion ≥ 281600'; that is mathematically impossible
    from a Hadamard parent, since every cofactor of an m×m Hadamard
    matrix has |det| = m^{m/2-1}.
    """
    seeds = list(paley_deletions(11))
    abs_dets = sorted({abs(d) for _, _, _, d in seeds})
    assert abs_dets == [248832], f"unexpected |det| set: {abs_dets}"
    # Sanity floor: cofactor must be at least the bordered ceiling at n=7
    # (i.e. comfortably above random ±1 expected |det|).
    assert max(abs_dets) >= 100_000


# ---------------------------------------------------------------------------
# top_paley_seeds API
# ---------------------------------------------------------------------------


def test_top_paley_seeds_returns_descending_by_abs_det_at_n7():
    top = top_paley_seeds(7, k=10)
    assert len(top) == 10
    abs_dets = [abs(t[3]) for t in top]
    assert abs_dets == sorted(abs_dets, reverse=True)
    # First entry must be the global max (= 512 at n=7).
    assert abs_dets[0] == 512


def test_top_paley_seeds_returns_descending_by_abs_det_at_n23():
    """At n=23 every deletion has |det| = 24^11; tie-break by (i, j)."""
    top = top_paley_seeds(23, k=10)
    assert len(top) == 10
    abs_dets = [abs(t[3]) for t in top]
    assert abs_dets == sorted(abs_dets, reverse=True)
    assert abs_dets[0] == 24**11


def test_top_paley_seed_matrix_has_correct_det():
    """Recompute |det| on returned Matrix to check it round-trips."""
    top = top_paley_seeds(7, k=3)
    for _, _, mat, signed_d in top:
        recomputed = integer_determinant(mat.entries)
        assert recomputed == signed_d


def test_top_paley_seeds_default_k_equals_10():
    top = top_paley_seeds(11)
    assert len(top) == 10


# ---------------------------------------------------------------------------
# Seed has high determinant relative to random ±1
# ---------------------------------------------------------------------------


def test_paley_seed_at_n23_exceeds_random_baseline():
    """The H24 cofactor seed (24^11 ≈ 1.52e15) is FAR above any random
    ±1 expectation at n=23. This is the whole point of using it.
    """
    top = top_paley_seeds(23, k=1)
    seed_det = abs(top[0][3])
    # Random ±1 expected |det| at n=23 is roughly sqrt(n!) ≈ 4e11;
    # the seed is at least 1000× that.
    assert seed_det >= 1e15
    assert seed_det == 24**11
    # And it should be a Matrix of size 23.
    assert top[0][2].n == 23
    assert isinstance(top[0][2], Matrix)
