"""Tests for winner_decode: structural fingerprints of winning matrices."""
from __future__ import annotations

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant
from hadamard_maxdet.winner_decode import (
    cross_n_winner_table,
    fingerprint_matrix,
    winner_fingerprint,
)


def test_fingerprint_hadamard_n8():
    """Sylvester H_8 has Gram = 8·I, so off-diagonals are all 0
    (single-valued support {0}) and abs_det = 8^4."""
    M = Matrix.sylvester(8)
    fp = fingerprint_matrix(M.entries, source="sylvester")
    assert fp.n == 8
    # Sylvester H_8 has |det| = 8^4 = 4096.
    assert fp.abs_det_verified == 8 ** 4
    # Hadamard ⇒ Gram is 8·I ⇒ off-diagonals all zero.
    assert fp.gram_offdiag_support == [0]
    assert fp.gram_offdiag_cardinality == 1
    # All singular values equal sqrt(8).
    assert fp.sv_distinct_count == 1
    # Hadamard ratio_to_upper_bound == 1.
    assert abs(fp.ratio_to_upper_bound - 1.0) < 1e-12


def test_winner_fingerprint_n11_proven():
    """At n=11 the proven max is 327680 and SA multi-start finds it
    routinely. Fingerprint should record proven_max and ratio 1.0."""
    fp = winner_fingerprint(
        11,
        n_starts=4,
        iterations=40_000,
        structured_seconds=0.0,   # cheap path
        base_seed=7,
    )
    assert fp.n == 11
    assert fp.proven_max == 327680
    assert fp.abs_det_verified == 327680, (
        f"expected proven 327680, got {fp.abs_det_verified}"
    )
    assert fp.ratio_to_proven == 1.0
    # n ≡ 3 mod 4: Gram off-diagonals are odd. Support cardinality
    # must be at least 1; record it.
    assert fp.gram_offdiag_cardinality >= 1
    # All off-diag values should be odd at n=11 (since rows have weight w
    # and <v_i,v_j> ≡ n mod 2 = 1 mod 2).
    for v in fp.gram_offdiag_support:
        assert v % 2 == 1, f"expected odd off-diag at n=11, got {v}"
    # Verified determinant matches.
    assert fp.abs_det_verified == abs(integer_determinant(fp.matrix))


def test_winner_fingerprint_n13_proven():
    """At n=13 the proven max is 14929920. SA multi-start saturates."""
    fp = winner_fingerprint(
        13,
        n_starts=6,
        iterations=80_000,
        structured_seconds=0.0,
        base_seed=11,
    )
    assert fp.n == 13
    assert fp.proven_max == 14929920
    assert fp.abs_det_verified == 14929920
    assert fp.ratio_to_proven == 1.0
    # n=13 is Barba-tight ⇒ Gram should have a clean 2-valued support
    # ((+5, -3) or similar after row sign normalisation; we report
    # whatever the actual signed off-diagonals are).
    # Cardinality should be small (1 or 2) for the Barba-tight case.
    assert fp.gram_offdiag_cardinality <= 4
    assert fp.abs_det_verified == abs(integer_determinant(fp.matrix))


def test_cross_n_winner_table_smoke():
    """End-to-end: build winner table for two small n values."""
    rows = cross_n_winner_table(
        [8, 11],
        n_starts=2,
        iterations=20_000,
        structured_seconds=0.0,
        base_seed=23,
    )
    assert len(rows) == 2
    assert rows[0].n == 8
    assert rows[1].n == 11
    # JSON serialisation works.
    for r in rows:
        j = r.to_json(include_matrix=False)
        assert j["n"] == r.n
        assert "gram_offdiag_support" in j
        assert "sv_multiplicities" in j
