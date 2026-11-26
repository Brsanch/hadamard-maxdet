"""Verify the 0.9499·Orrick plateau-neighbor matrix is correct."""
import json
from pathlib import Path
from hadamard_maxdet.verifier import integer_determinant

ROOT = Path(__file__).resolve().parents[1]


def test_plateau_neighbor_0p95_det():
    payload = json.loads((ROOT / "results" / "H23_plateau_neighbor_0p95.json").read_text())
    M = payload["entries"]
    assert len(M) == 23 and all(len(r) == 23 for r in M)
    assert all(v in (1, -1) for r in M for v in r)
    det = abs(integer_determinant(M))
    assert det == payload["abs_det"] == 2_640_248_832_000_000
    # 22 years of Orrick is 2,779,447,296,000,000; this matrix is 0.9499 of that
    assert 0.949 < det / 2_779_447_296_000_000 < 0.951


def test_plateau_neighbor_0p95_is_2flip_of_orrick():
    """The plateau-neighbor must differ from Orrick by exactly 2 sign flips."""
    orrick = json.loads((ROOT / "results" / "H23_orrick2003_record.json").read_text())["entries"]
    M = json.loads((ROOT / "results" / "H23_plateau_neighbor_0p95.json").read_text())["entries"]
    diffs = [(i, j) for i in range(23) for j in range(23) if M[i][j] != orrick[i][j]]
    assert len(diffs) == 2
    # The two flipped cells must be in the same row (this is a same-row 2-flip)
    assert diffs[0][0] == diffs[1][0]


def test_plateau_neighbor_0p95_gram_signature():
    """Confirm the 3-val signature with count_5=1 (actually count_7=1 in the
    raw signed sums — the signed Gram off-diag has a single +7 entry)."""
    M = json.loads((ROOT / "results" / "H23_plateau_neighbor_0p95.json").read_text())["entries"]
    n = 23
    G = [[sum(M[i][k] * M[j][k] for k in range(n)) for j in range(n)] for i in range(n)]
    od = [G[i][j] for i in range(n) for j in range(i + 1, n)]
    abs_counts = {}
    for v in od:
        abs_counts[abs(v)] = abs_counts.get(abs(v), 0) + 1
    assert abs_counts == {1: 207, 3: 45, 7: 1}
