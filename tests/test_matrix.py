import random

import pytest

from hadamard_maxdet import Matrix, verify


def test_random_matrix_is_valid_pm1():
    rng = random.Random(0)
    M = Matrix.random(5, rng)
    v = verify(M.entries)
    assert v.valid


def test_sylvester_requires_power_of_two():
    Matrix.sylvester(1)
    Matrix.sylvester(2)
    Matrix.sylvester(4)
    with pytest.raises(ValueError):
        Matrix.sylvester(3)
    with pytest.raises(ValueError):
        Matrix.sylvester(6)


def test_sylvester_is_hadamard():
    # Rows pairwise orthogonal ⇔ H H^T = n I.
    for n in (1, 2, 4, 8, 16):
        H = Matrix.sylvester(n).entries
        for i in range(n):
            for j in range(i + 1, n):
                s = sum(H[i][k] * H[j][k] for k in range(n))
                assert s == 0, f"rows {i},{j} not orthogonal at n={n}"


def test_paley_type_i_order_12():
    # q=11, q ≡ 3 (mod 4), q prime → Paley I of order 12.
    H = Matrix.paley_type_i(12).entries
    v = verify(H)
    assert v.valid
    assert v.abs_det == 12 ** 6


def test_paley_type_i_order_8():
    # q=7, q ≡ 3 (mod 4), q prime → Paley I of order 8.
    H = Matrix.paley_type_i(8).entries
    v = verify(H)
    assert v.valid
    assert v.abs_det == 8 ** 4


def test_paley_rejects_bad_order():
    with pytest.raises(ValueError):
        Matrix.paley_type_i(6)  # q=5 ≡ 1 (mod 4)
    with pytest.raises(ValueError):
        Matrix.paley_type_i(10)  # q=9 not prime


def test_circulant_is_valid():
    rng = random.Random(1)
    M = Matrix.random_circulant(7, rng)
    v = verify(M.entries)
    assert v.valid
    # First row equals row_0.
    first = M.entries[0]
    for i in range(7):
        for j in range(7):
            assert M.entries[i][j] == first[(j - i) % 7]


def test_circulant_rejects_non_pm1():
    with pytest.raises(ValueError):
        Matrix.circulant([1, 0, 1])


def test_copy_is_independent():
    rng = random.Random(2)
    M = Matrix.random(4, rng)
    N = M.copy()
    N.entries[0][0] = -N.entries[0][0]
    assert N.entries[0][0] != M.entries[0][0]
