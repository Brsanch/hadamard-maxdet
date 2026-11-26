from hadamard_maxdet import verify
from hadamard_maxdet.verifier import integer_determinant, upper_bound


def test_verify_rejects_non_pm1():
    v = verify([[1, 0], [1, 1]])
    assert not v.valid
    assert "only -1 and +1" in v.error


def test_verify_rejects_non_square():
    v = verify([[1, 1, 1], [1, -1, 1]])
    assert not v.valid
    assert "length" in v.error


def test_verify_empty():
    v = verify([])
    assert not v.valid


def test_sylvester_h4_has_abs_det_16():
    from hadamard_maxdet import Matrix
    H = Matrix.sylvester(4).entries
    v = verify(H)
    assert v.valid
    assert v.abs_det == 16


def test_sylvester_h8_has_abs_det_4096():
    from hadamard_maxdet import Matrix
    H = Matrix.sylvester(8).entries
    v = verify(H)
    assert v.valid
    assert v.abs_det == 4096  # 8^4


def test_integer_determinant_small_cases():
    # 1x1
    assert integer_determinant([[1]]) == 1
    assert integer_determinant([[-1]]) == -1
    # 2x2: det = ad - bc
    assert integer_determinant([[1, 1], [1, -1]]) == -2
    assert integer_determinant([[1, -1], [1, 1]]) == 2


def test_integer_determinant_exact_no_float():
    # H16 Sylvester: |det| = 16^8 = 4294967296 — beyond float53 mantissa precision is fine here
    # but the point is we get the exact integer.
    from hadamard_maxdet import Matrix
    H = Matrix.sylvester(16).entries
    d = integer_determinant(H)
    assert abs(d) == 16 ** 8


def test_integer_determinant_singular():
    # Rows identical → det 0.
    assert integer_determinant([[1, 1], [1, 1]]) == 0


def test_upper_bound_hadamard():
    assert upper_bound(4) == 16
    assert upper_bound(8) == 4096
    assert upper_bound(16) == 16 ** 8


def test_integer_determinant_numpy_int64_no_overflow_at_n23():
    """Regression: numpy int64 silently overflows during Bareiss elimination
    at n >= 23 for ±1 matrices, because intermediate products
    M[i][j] * pivot - M[i][k] * M[k][j] can exceed 2**63.

    Discovered 2026-04-25 from the H23 Orrick record matrix:
    pure-Python integer_determinant returned ±2,779,447,296,000,000 (correct),
    while passing the same matrix as a numpy.ndarray(dtype=int64) returned
    a small wrong integer (4,541,561) with a RuntimeWarning emitted by numpy.

    The fix coerces every scalar to a native Python int before elimination.
    """
    import json
    import os

    import numpy as np

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    record_path = os.path.join(repo_root, "results", "H23_orrick2003_record.json")
    with open(record_path) as fh:
        record = json.load(fh)

    entries = record["entries"]
    expected_abs = 2_779_447_296_000_000

    # Pure-Python list-of-lists baseline.
    d_python = integer_determinant(entries)
    assert abs(d_python) == expected_abs, (
        f"baseline list[list[int]] returned {d_python}, expected ±{expected_abs}"
    )

    # numpy int64 input must agree with the baseline (this was the bug).
    arr = np.asarray(entries, dtype=np.int64)
    d_numpy = integer_determinant(arr)
    assert d_numpy == d_python, (
        f"numpy int64 input returned {d_numpy} but pure-Python returned {d_python}; "
        "this is the silent overflow regression — see verifier.py "
        "_to_python_int_rows for the fix."
    )

    # Other fixed-width dtypes that would also overflow without coercion.
    for dtype in (np.int32, np.int16, np.int8):
        arr_d = np.asarray(entries, dtype=dtype)
        d = integer_determinant(arr_d)
        assert d == d_python, (
            f"dtype={dtype} returned {d} but pure-Python returned {d_python}"
        )


def test_integer_determinant_accepts_tuples_and_iterables():
    # Coercion should accept tuple-of-tuples and produce the same answer.
    m = ((1, 1, 1), (1, -1, 1), (1, 1, -1))
    assert integer_determinant(m) == integer_determinant([list(r) for r in m])


def test_upper_bound_other_classes_are_upper():
    # Barba bound at n=7 is > 576 (the true max); check it's at least 576.
    from hadamard_maxdet import BENCHMARKS
    for oid, o in BENCHMARKS.items():
        if o.best_known is None:
            continue
        assert o.best_known <= upper_bound(o.n), (
            f"{oid}: best_known {o.best_known} exceeds bound {upper_bound(o.n)}"
        )
