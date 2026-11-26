"""Verify the 0.9810·Orrick matrix found in Round 3 — third orbit in (202, 51) class."""
import json
from pathlib import Path
from hadamard_maxdet.verifier import integer_determinant

ROOT = Path(__file__).resolve().parents[1]
ORRICK = 2_779_447_296_000_000


def test_round3_0p9810_det():
    p = json.loads((ROOT / "results" / "H23_round3_record_00_r0.9810.json").read_text())
    M = p["entries"]
    assert len(M) == 23 and all(len(r) == 23 for r in M)
    assert all(v in (1, -1) for r in M for v in r)
    det = abs(integer_determinant(M))
    assert det == p["best_det"] == 2_726_756_352_000_000
    # 2^22 * 3^3 * 5^6 * 23 * 67
    assert det == (2**22) * (3**3) * (5**6) * 23 * 67
    assert 0.980 < det / ORRICK < 0.982


def test_round3_0p9810_2val_202_51():
    p = json.loads((ROOT / "results" / "H23_round3_record_00_r0.9810.json").read_text())
    assert p["gram_sig"] == {"1": 202, "3": 51}  # 2-val signature


def test_three_distinct_orbits_in_202_51_class():
    """The (202, 51) count-class contains 3 distinct |det| sign-orbits."""
    a = json.loads((ROOT / "results" / "H23_anchor_band_0005_002_smart2val.json").read_text())
    b = json.loads((ROOT / "results" / "H23_recurse_record_0p9617.json").read_text())
    c = json.loads((ROOT / "results" / "H23_round3_record_00_r0.9810.json").read_text())
    a_det = a["abs_det"]
    b_det = b["abs_det"]
    c_det = c["best_det"]
    # All three different
    assert len({a_det, b_det, c_det}) == 3
    # All have the (202, 51) count signature
    assert a["abs_gram_multiset"] == {"1": 202, "3": 51}
    assert b["abs_gram_multiset"] == {"1": 202, "3": 51}
    assert c["gram_sig"] == {"1": 202, "3": 51}
    # Strict ordering
    assert a_det < b_det < c_det
