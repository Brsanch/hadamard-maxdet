"""Verify the 0.9617·Orrick matrix found via recursive anchor enumeration.

Provenance: smart_2val → k=2 → band_0005_002_smart2val → k=2 → this.
"""
import json
from pathlib import Path
from hadamard_maxdet.verifier import integer_determinant

ROOT = Path(__file__).resolve().parents[1]
ORRICK = 2_779_447_296_000_000


def test_recurse_record_0p9617_det():
    p = json.loads((ROOT / "results" / "H23_recurse_record_0p9617.json").read_text())
    M = p["entries"]
    assert len(M) == 23 and all(len(r) == 23 for r in M)
    assert all(v in (1, -1) for r in M for v in r)
    det = abs(integer_determinant(M))
    assert det == p["abs_det"] == 2_672_885_760_000_000
    assert 0.961 < det / ORRICK < 0.962


def test_recurse_record_0p9617_2val_class_202_51():
    p = json.loads((ROOT / "results" / "H23_recurse_record_0p9617.json").read_text())
    assert p["abs_gram_multiset"] == {"1": 202, "3": 51}  # 2-val signature


def test_recurse_record_0p9617_distinct_from_band_0005_002():
    """Same |G| count signature as band_0005_002_smart2val, but distinct
    sign-equivalence orbit (different |det| ⇒ not sign-equivalent)."""
    parent = json.loads((ROOT / "results" / "H23_anchor_band_0005_002_smart2val.json").read_text())
    new = json.loads((ROOT / "results" / "H23_recurse_record_0p9617.json").read_text())
    assert parent["abs_gram_multiset"] == new["abs_gram_multiset"]  # same count
    assert parent["abs_det"] != new["abs_det"]  # different orbit
    assert new["abs_det"] > parent["abs_det"]  # the new one is better


def test_recurse_record_0p9617_distance_from_smart_2val():
    """Should be Hamming distance 4 from smart_2val (k=2 of k=2)."""
    sm = json.loads((ROOT / "results" / "H23_smart_2val_best_new_class.json").read_text())["entries"]
    new = json.loads((ROOT / "results" / "H23_recurse_record_0p9617.json").read_text())["entries"]
    diffs = sum(1 for i in range(23) for j in range(23) if sm[i][j] != new[i][j])
    assert diffs == 4
