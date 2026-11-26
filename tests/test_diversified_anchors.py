"""Verify the 5 new matrices found via the anchor-diversification campaign."""
import json
from pathlib import Path
from hadamard_maxdet.verifier import integer_determinant

ROOT = Path(__file__).resolve().parents[1]
ORRICK = 2_779_447_296_000_000


def _load(path):
    return json.loads((ROOT / "results" / path).read_text())


def test_band_011_020_k2_orrick_neighbor():
    """k=2 neighbor of Orrick at ratio 0.9605 — NEW 2-val class (205, 48)."""
    p = _load("H23_anchor_band_011_020_k2.json")
    M = p["entries"]
    assert len(M) == 23 and all(len(r) == 23 for r in M)
    det = abs(integer_determinant(M))
    assert det == p["abs_det"]
    assert 0.960 < det / ORRICK < 0.961
    assert p["abs_gram_multiset"] == {"1": 205, "3": 48}  # 2-val, count_5/7 = 0


def test_band_0005_002_smart2val_neighbor():
    """k=2 of smart_2val in formerly-empty band [0.0005, 0.002) — 2-val (202, 51)."""
    p = _load("H23_anchor_band_0005_002_smart2val.json")
    det = abs(integer_determinant(p["entries"]))
    assert det == p["abs_det"]
    assert 0.945 < det / ORRICK < 0.947
    assert p["abs_gram_multiset"] == {"1": 202, "3": 51}


def test_band_04_08_smart2val_neighbor():
    """k=2 of smart_2val in band [0.04, 0.08) — 3-val (207, 45, 1)."""
    p = _load("H23_anchor_band_04_08_smart2val.json")
    det = abs(integer_determinant(p["entries"]))
    assert det == p["abs_det"]
    assert 0.949 < det / ORRICK < 0.950
    assert p["abs_gram_multiset"] == {"1": 207, "3": 45, "5": 1}


def test_band_08_14_btc_neighbor():
    """k=2 of BTC in band [0.08, 0.14) — 2-val (213, 40)."""
    p = _load("H23_anchor_band_08_14_btc.json")
    det = abs(integer_determinant(p["entries"]))
    assert det == p["abs_det"]
    assert 0.913 < det / ORRICK < 0.915
    assert p["abs_gram_multiset"] == {"1": 213, "3": 40}


def test_band_002_005_plateaunbr_neighbor():
    """k=2 of plateau_nbr_095 in band [0.002, 0.005) — 3-val (205, 47, 1)."""
    p = _load("H23_anchor_band_002_005_plateaunbr.json")
    det = abs(integer_determinant(p["entries"]))
    assert det == p["abs_det"]
    assert 0.930 < det / ORRICK < 0.931
    assert p["abs_gram_multiset"] == {"1": 205, "3": 47, "7": 1}
