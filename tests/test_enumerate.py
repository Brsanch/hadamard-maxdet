"""Tests for exhaustive enumeration over structured-ansatz cores."""

from __future__ import annotations

import itertools

import pytest

from hadamard_maxdet.enumerate import (
    EnumResult,
    _bits_from_index,
    _canonical_under_rotation,
    _index_from_bits,
    enumerate_bordered_circulant,
    enumerate_skew_circulant,
    negation_preserves_abs_det,
)
from hadamard_maxdet.structured import (
    bordered_circulant,
    skew_generator_from_free_bits,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# Bit / index round-trip
# ---------------------------------------------------------------------------


def test_bits_index_roundtrip():
    for m in (1, 2, 5, 10):
        for idx in range(1 << m):
            assert _index_from_bits(_bits_from_index(idx, m)) == idx


def test_canonical_under_rotation_is_orbit_minimum():
    # For length-4 orbit {[1,1,-1,-1], [-1,1,1,-1], [-1,-1,1,1], [1,-1,-1,1]}
    bits_set = [
        [1, 1, -1, -1],
        [-1, 1, 1, -1],
        [-1, -1, 1, 1],
        [1, -1, -1, 1],
    ]
    canon_set = {
        _canonical_under_rotation(_index_from_bits(b), 4) for b in bits_set
    }
    assert len(canon_set) == 1


# ---------------------------------------------------------------------------
# Negation symmetry: empirical claim from spec is FALSE.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 7, 8, 9, 11, 13])
def test_negation_does_NOT_preserve_abs_det(n):
    # Spec's "negating the whole core doesn't change |det|" hypothesis is
    # incorrect for bordered-circulant. We document it as a negative
    # finding here so the test fails loudly if anyone re-introduces a
    # speedup based on it.
    assert negation_preserves_abs_det(n, samples=64) is False


# ---------------------------------------------------------------------------
# Bordered-circulant enumeration agrees with brute force at small n.
# ---------------------------------------------------------------------------


def _brute_force_bordered_max(n: int) -> int:
    m = n - 1
    best = 0
    for bits in itertools.product([-1, 1], repeat=m):
        d = abs(integer_determinant(bordered_circulant(list(bits)).entries))
        if d > best:
            best = d
    return best


@pytest.mark.parametrize("n,expected", [(5, 48), (7, 576), (8, 4096)])
def test_bordered_enum_matches_known_small(n, expected):
    """Bordered-circulant ansatz reaches the known maximum at n=5,7,8."""
    r = enumerate_bordered_circulant(n, use_multiprocessing=False)
    assert r.best_det == expected
    # Cross-check against brute force.
    assert r.best_det == _brute_force_bordered_max(n)


def test_bordered_enum_n11_ansatz_ceiling():
    # Existing test_structured asserts SA reaches 281,600 at n=11; confirm
    # that's actually the ansatz max via exhaustive enum.
    r = enumerate_bordered_circulant(11, use_multiprocessing=False)
    assert r.best_det == 281_600


def test_bordered_enum_n13_below_proven_max():
    # H13 proven max is 14,929,920. Bordered-circulant ansatz cannot reach
    # it; we record its actual ceiling to lock the value in.
    r = enumerate_bordered_circulant(13, use_multiprocessing=False)
    assert r.best_det <= 14_929_920
    assert r.best_det == 9_175_040  # observed ansatz ceiling


# ---------------------------------------------------------------------------
# Equivalence-class reduction preserves the maximum at n=8.
# ---------------------------------------------------------------------------


def test_equivalence_reduction_preserves_max_n8():
    """At n=8, full enum and reduced (rotation-quotiented) enum agree."""
    full = _brute_force_bordered_max(8)
    reduced = enumerate_bordered_circulant(8, use_multiprocessing=False)
    assert reduced.best_det == full
    # Reduction actually shrinks the work: 128 -> ~20 reps.
    assert reduced.visited < reduced.total_indices
    assert reduced.visited <= 128 // 6  # each orbit has up to 7 rotations


def test_equivalence_reduction_preserves_max_n9():
    full = _brute_force_bordered_max(9)
    reduced = enumerate_bordered_circulant(9, use_multiprocessing=False)
    assert reduced.best_det == full


# ---------------------------------------------------------------------------
# Negation-class flag is honest about the symmetry.
# ---------------------------------------------------------------------------


def test_include_negation_class_does_not_drop_max():
    # Even when we ask for the negation quotient, since the symmetry
    # doesn't hold the enumerator must NOT skip orbits and must report
    # negation_symmetric == False. The result must still equal the
    # rotation-only max.
    r_plain = enumerate_bordered_circulant(7, use_multiprocessing=False)
    r_neg = enumerate_bordered_circulant(
        7, use_multiprocessing=False, include_negation_class=True
    )
    assert r_neg.negation_symmetric is False
    assert r_neg.best_det == r_plain.best_det


# ---------------------------------------------------------------------------
# Limit parameter caps the number of representatives visited.
# ---------------------------------------------------------------------------


def test_limit_caps_visited():
    r = enumerate_bordered_circulant(11, limit=10, use_multiprocessing=False)
    assert r.visited == 10


# ---------------------------------------------------------------------------
# Skew-circulant enumeration.
# ---------------------------------------------------------------------------


def test_skew_enum_n7():
    # 3 free bits = 8 cores; brute-force max is the |det| ceiling.
    half = 3
    best = 0
    for bits in itertools.product([-1, 1], repeat=half):
        gen = skew_generator_from_free_bits(7, list(bits))
        d = abs(integer_determinant(Matrix.circulant(gen).entries))
        if d > best:
            best = d
    r = enumerate_skew_circulant(7, use_multiprocessing=False)
    assert r.best_det == best


def test_skew_enum_rejects_even():
    with pytest.raises(ValueError):
        enumerate_skew_circulant(8)


# ---------------------------------------------------------------------------
# EnumResult dataclass shape.
# ---------------------------------------------------------------------------


def test_enum_result_fields():
    r = enumerate_bordered_circulant(5, use_multiprocessing=False)
    assert isinstance(r, EnumResult)
    assert r.n == 5
    assert r.ansatz == "bordered_circulant"
    assert r.total_indices == 16
    assert len(r.best_core) == 4
    assert all(b in (-1, 1) for b in r.best_core)
