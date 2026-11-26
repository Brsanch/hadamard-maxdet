"""Verify the 0.9576·Orrick band-anchor matrix."""
import json
from pathlib import Path
from hadamard_maxdet.verifier import integer_determinant

ROOT = Path(__file__).resolve().parents[1]


def test_band_020_040_anchor_det():
    p = json.loads((ROOT / "results" / "H23_anchor_band_020_040.json").read_text())
    M = p["entries"]
    assert len(M) == 23 and all(len(r) == 23 for r in M)
    assert all(v in (1, -1) for r in M for v in r)
    det = abs(integer_determinant(M))
    assert det == p["abs_det"] == 2_661_482_496_000_000
    assert 0.957 < det / 2_779_447_296_000_000 < 0.958


def test_band_020_040_anchor_is_2flip_of_orrick():
    orrick = json.loads((ROOT / "results" / "H23_orrick2003_record.json").read_text())["entries"]
    M = json.loads((ROOT / "results" / "H23_anchor_band_020_040.json").read_text())["entries"]
    diffs = [(i, j) for i in range(23) for j in range(23) if M[i][j] != orrick[i][j]]
    assert len(diffs) == 2  # k=2 from Orrick


def test_band_020_040_anchor_gram():
    M = json.loads((ROOT / "results" / "H23_anchor_band_020_040.json").read_text())["entries"]
    n = 23
    G = [[sum(M[i][k] * M[j][k] for k in range(n)) for j in range(n)] for i in range(n)]
    od = [G[i][j] for i in range(n) for j in range(i + 1, n)]
    abs_counts = {}
    for v in od:
        abs_counts[abs(v)] = abs_counts.get(abs(v), 0) + 1
    assert abs_counts == {1: 207, 3: 45, 5: 1}
