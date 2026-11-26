"""H61 and H113 closure at EW bound via Koukouvinos SBIBD lookup.

Same template as cycle 51 (H41): for n ≡ 1 mod 4 with 2n−1 a perfect
square AND n = q² + (q+1)² for prime power q, Theorem 16 of Browne-Egan-
Hegarty-Ó Catháin 2021 (generalising Brouwer-Whiteman) gives an EW-tight
matrix via an SBIBD with parameters (n, k, λ) where k − λ = (n−1)/4 and
the wrap M = 2N − J satisfies M·M^T = (n−1)·I + J.

Koukouvinos publishes explicit (v, k, λ) SBIBDs at:
    http://www.math.ntua.gr/~ckoukouv/{v}_{k}_{lam}.htm

Available designs in this family (q ≤ 7):
    q=1: (5, 1, 0)        — trivial
    q=2: (13, 4, 1)       — projective plane PG(2, 3)
    q=3: (25, 9, 3)       — closed via cycle 50 (own search)
    q=4: (41, 16, 6)      — closed via cycle 51 (Koukouvinos lookup)
    q=5: (61, 25, 10)     — THIS, q=5 SBIB Design
    q=7: (113, 49, 21)    — THIS, q=7 SBIB Design

For q=6 (n=85) and q=8 (n=145), q is not in the simple-prime case;
Koukouvinos does not have these designs published.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hadamard_maxdet.verifier import integer_determinant


def fetch_koukouvinos_sbibd(v: int, k: int, lam: int) -> list[list[int]]:
    """Fetch and parse the Koukouvinos (v, k, λ) SBIBD page.

    Returns the v × v {0, 1} incidence matrix as nested lists.
    """
    url = f"http://www.math.ntua.gr/~ckoukouv/{v}_{k}_{lam}.htm"
    with urllib.request.urlopen(url, timeout=30) as f:
        raw = f.read().decode("latin-1")
    rows = []
    for ln in raw.replace("<br>", "\n").replace("&nbsp;", " ").split("\n"):
        s = "".join(c for c in ln if c in "01")
        if len(s) == v:
            rows.append([int(c) for c in s])
    if len(rows) < v:
        raise RuntimeError(f"Could not parse {v} rows from {url}; got {len(rows)}")
    return rows[:v]


def verify_and_save(N: list[list[int]], n: int, k: int, lam: int, label: str) -> dict:
    A = np.array(N, dtype=np.int64)
    AAt = A @ A.T
    expected = (k - lam) * np.eye(n, dtype=np.int64) + lam * np.ones((n, n), dtype=np.int64)
    if not np.array_equal(AAt, expected):
        raise RuntimeError(f"{label}: SBIBD verification failed")
    M = 2 * A - np.ones((n, n), dtype=np.int64)
    det = abs(integer_determinant(M.tolist()))
    EW = 0
    # EW(n) for n ≡ 1 mod 4 with 2n−1 a perfect square: sqrt(2n-1) · (n-1)^((n-1)/2)
    sqrt_2n_1 = int(round((2 * n - 1) ** 0.5))
    assert sqrt_2n_1 * sqrt_2n_1 == 2 * n - 1
    EW = sqrt_2n_1 * (n - 1) ** ((n - 1) // 2)
    print(f"{label}:")
    print(f"  |det(M)| = {det}")
    print(f"  EW({n})  = {EW}")
    print(f"  ratio    = {det / EW}")
    print(f"  EW-tight = {det == EW}")

    # Triple-verify with sympy
    import sympy as sp
    det_sp = abs(int(sp.Matrix(M.tolist()).det()))
    print(f"  sympy match: {det_sp == det}")
    assert det == det_sp == EW, "ALL three must match exactly"

    out = {
        "order_id": f"H{n}", "n": n, "k": k, "lam": lam,
        "source": f"({n}, {k}, {lam}) SBIBD from Koukouvinos {f'http://www.math.ntua.gr/~ckoukouv/{n}_{k}_{lam}.htm'}",
        "survey_reference": "Browne-Egan-Hegarty-Ó Catháin 2021 Theorem 16 (Brouwer-Whiteman family at q=q²+(q+1)²)",
        "abs_det": det,
        f"ew_{n}": EW,
        "ratio_to_ew": det / EW,
        "ew_tight": True,
        "verified_by_bareiss_sympy": True,
        "incidence_N": A.tolist(),
    }
    out_path = REPO / "results" / f"H{n}_cycle52_sbibd.json"
    out_path.write_text(json.dumps(out, indent=2))
    np.save(REPO / "results" / f"H{n}_cycle52_sbibd.npy", M.astype(np.int8))
    print(f"  Saved: {out_path}")
    return out


def main() -> None:
    print("Fetching (61, 25, 10) SBIBD from Koukouvinos...")
    N61 = fetch_koukouvinos_sbibd(61, 25, 10)
    verify_and_save(N61, 61, 25, 10, "H61")
    print()
    print("Fetching (113, 49, 21) SBIBD from Koukouvinos...")
    N113 = fetch_koukouvinos_sbibd(113, 49, 21)
    verify_and_save(N113, 113, 49, 21, "H113")


if __name__ == "__main__":
    main()
