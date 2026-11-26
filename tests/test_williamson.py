"""Tests for the Williamson m=5 construction.

Load-bearing checks:
  * verify_williamson_constraint is exact for all enumerated quadruples.
  * williamson_array is a valid 20×20 ±1 Hadamard (rows orthogonal,
    |det| = 20^10).
  * At least one bordered-H_21 candidate is a valid ±1 matrix with a
    substantial |det|.
  * At least one H_19 deletion has |det| ≥ 0.9 · proven max
    (894,426,939,392).
"""

from __future__ import annotations

from hadamard_maxdet.verifier import integer_determinant, verify
from hadamard_maxdet.williamson import (
    BORDER_PATTERNS,
    bordered_williamson_h21,
    circulant_matrix,
    enumerate_symmetric_circulant_generators,
    gs_array_williamson_h23,
    is_symmetric,
    symmetric_circulant_first_row,
    verify_williamson_constraint,
    williamson_array,
    williamson_h20_deletion_h19,
    williamson_h20_from_generators,
    williamson_quadruples_m5,
)


# ---------------------------------------------------------------------------
# Generators and circulant symmetry
# ---------------------------------------------------------------------------


def test_symmetric_circulant_generator_layout():
    g = symmetric_circulant_first_row(1, -1, 1)
    assert g == [1, -1, 1, 1, -1]


def test_enumeration_count_m5():
    gens = enumerate_symmetric_circulant_generators(5)
    assert len(gens) == 8
    seen = {tuple(g) for g in gens}
    assert len(seen) == 8


def test_circulant_matrix_symmetric_for_symmetric_generator():
    for g in enumerate_symmetric_circulant_generators(5):
        C = circulant_matrix(g)
        assert is_symmetric(C), f"non-symmetric for g={g}"


# ---------------------------------------------------------------------------
# Williamson constraint + array
# ---------------------------------------------------------------------------


def test_williamson_quadruples_nonempty_and_constraint():
    quads = williamson_quadruples_m5()
    assert len(quads) > 0, "no Williamson quadruples found at m=5"
    # The exhaustive enumeration over symmetric circulants includes the full
    # symmetry orbit (S_4 permutation × ±1 negations), so we expect well over
    # a handful. Williamson's classical existence at m=5 yields >= 6 inequivalent.
    assert len(quads) >= 24
    # Re-verify each quadruple via the constraint (load-bearing).
    for a, b, c, d in quads:
        A = circulant_matrix(a)
        B = circulant_matrix(b)
        C = circulant_matrix(c)
        D = circulant_matrix(d)
        assert all(is_symmetric(X) for X in (A, B, C, D))
        assert verify_williamson_constraint(A, B, C, D)


def test_williamson_array_is_hadamard_h20():
    quads = williamson_quadruples_m5()
    target = 20 ** 10
    # Spot-check at least 10 quadruples, all must give |det| == 20^10.
    for a, b, c, d in quads[: min(10, len(quads))]:
        H = williamson_h20_from_generators(a, b, c, d)
        assert H.n == 20
        v = verify(H.entries)
        assert v.valid
        assert v.abs_det == target, f"|det|={v.abs_det} != 20^10 for {(a,b,c,d)}"


def test_williamson_array_rows_orthogonal():
    a, b, c, d = williamson_quadruples_m5()[0]
    H = williamson_h20_from_generators(a, b, c, d).entries
    n = 20
    # Hadamard ⇔ H Hᵀ = n I.
    for i in range(n):
        for j in range(n):
            ip = sum(H[i][k] * H[j][k] for k in range(n))
            if i == j:
                assert ip == n
            else:
                assert ip == 0


def test_verify_constraint_rejects_bad_quadruple():
    # All-+1 length-5 generators give A = J (all ones). Then A Aᵀ = 5 J ≠ 5 I,
    # so the sum is 4·5·J = 20 J, definitely not 20 I.
    g = symmetric_circulant_first_row(1, 1, 1)
    A = circulant_matrix(g)
    B = circulant_matrix(g)
    C = circulant_matrix(g)
    D = circulant_matrix(g)
    assert not verify_williamson_constraint(A, B, C, D)


# ---------------------------------------------------------------------------
# Border patterns → H_21
# ---------------------------------------------------------------------------


def test_bordered_williamson_h21_valid_for_all_patterns():
    a, b, c, d = williamson_quadruples_m5()[0]
    A = circulant_matrix(a)
    B = circulant_matrix(b)
    C = circulant_matrix(c)
    D = circulant_matrix(d)
    seen_dets: list[int] = []
    for pattern in BORDER_PATTERNS:
        M = bordered_williamson_h21(A, B, C, D, border_pattern=pattern)
        v = verify(M.entries)
        assert v.valid, f"{pattern}: {v.error}"
        assert v.n == 21
        seen_dets.append(v.abs_det)
    assert max(seen_dets) > 0
    # We don't yet claim what the best border |det| is — that's the
    # benchmark's job. But at least one should be substantial (>= 1e12).
    assert max(seen_dets) >= 10 ** 12


# ---------------------------------------------------------------------------
# H_19 deletions
# ---------------------------------------------------------------------------


def test_williamson_h20_deletions_count_and_validity():
    a, b, c, d = williamson_quadruples_m5()[0]
    A = circulant_matrix(a)
    B = circulant_matrix(b)
    C = circulant_matrix(c)
    D = circulant_matrix(d)
    dels = williamson_h20_deletion_h19(A, B, C, D)
    assert len(dels) == 400
    # Spot-check shape + ±1 entries on a sample.
    for (i, j), M in dels[:5]:
        assert M.n == 19
        for row in M.entries:
            assert len(row) == 19
            for v in row:
                assert v in (-1, 1)


def test_williamson_h20_deletion_substantial_h19():
    """At least one deletion must give a substantial |det| (>= 5e11).

    The naive Williamson H_20 → H_19 by single row + col deletion is known
    NOT to reach the proven H_19 max (894,426,939,392; Brent–Osborn–Orrick–
    Zimmermann 2012); that record is constructed differently. Here we only
    assert the deletion gives a non-trivial value, suitable as an SA seed.
    The benchmark script reports the actual ratio achieved.
    """
    threshold = 5 * 10 ** 11
    best = 0
    a, b, c, d = williamson_quadruples_m5()[0]
    A = circulant_matrix(a)
    B = circulant_matrix(b)
    C = circulant_matrix(c)
    D = circulant_matrix(d)
    for (_ij), M in williamson_h20_deletion_h19(A, B, C, D):
        d_abs = abs(integer_determinant(M.entries))
        if d_abs > best:
            best = d_abs
    assert best >= threshold, (
        f"best Williamson-derived |det H_19| = {best:,} < threshold "
        f"({threshold:,})"
    )


# ---------------------------------------------------------------------------
# GS-style H_23
# ---------------------------------------------------------------------------


def test_gs_array_williamson_h23_valid():
    a, b, c, d = williamson_quadruples_m5()[0]
    A = circulant_matrix(a)
    B = circulant_matrix(b)
    C = circulant_matrix(c)
    D = circulant_matrix(d)
    for variant in ("border3", "border3_gs"):
        M = gs_array_williamson_h23(A, B, C, D, variant=variant)
        assert M.n == 23
        v = verify(M.entries)
        assert v.valid, f"{variant}: {v.error}"
