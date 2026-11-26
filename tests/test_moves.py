import random

from hadamard_maxdet import Matrix, verify
from hadamard_maxdet.moves import (
    col_negate,
    col_swap,
    entry_flip,
    random_move,
    row_negate,
    row_swap,
    two_row_xor,
)
from hadamard_maxdet.verifier import integer_determinant


def _fresh(n: int, seed: int) -> list[list[int]]:
    return Matrix.random(n, random.Random(seed)).entries


def test_entry_flip_changes_one_entry():
    M = _fresh(5, 0)
    N = entry_flip(M, random.Random(1))
    diff = sum(M[i][j] != N[i][j] for i in range(5) for j in range(5))
    assert diff == 1
    assert verify(N).valid


def test_row_negate_preserves_abs_det():
    M = _fresh(6, 2)
    d0 = abs(integer_determinant(M))
    for _ in range(5):
        M = row_negate(M, random.Random(_))
    d1 = abs(integer_determinant(M))
    assert d0 == d1


def test_col_negate_preserves_abs_det():
    M = _fresh(6, 3)
    d0 = abs(integer_determinant(M))
    for _ in range(5):
        M = col_negate(M, random.Random(_))
    d1 = abs(integer_determinant(M))
    assert d0 == d1


def test_row_swap_preserves_abs_det():
    M = _fresh(6, 4)
    d0 = abs(integer_determinant(M))
    N = row_swap(M, random.Random(0))
    assert N is not None
    d1 = abs(integer_determinant(N))
    assert d0 == d1


def test_col_swap_preserves_abs_det():
    M = _fresh(6, 5)
    d0 = abs(integer_determinant(M))
    N = col_swap(M, random.Random(0))
    assert N is not None
    d1 = abs(integer_determinant(N))
    assert d0 == d1


def test_two_row_xor_is_valid_pm1():
    M = _fresh(6, 6)
    N = two_row_xor(M, random.Random(0))
    assert N is not None
    assert verify(N).valid


def test_random_move_always_valid():
    rng = random.Random(42)
    M = _fresh(7, 7)
    for _ in range(200):
        p = random_move(M, rng)
        if p is None:
            continue
        assert verify(p).valid
        M = p


def test_moves_do_not_mutate_input():
    M = _fresh(5, 8)
    snap = [row[:] for row in M]
    entry_flip(M, random.Random(0))
    row_negate(M, random.Random(0))
    two_row_xor(M, random.Random(0))
    assert M == snap
