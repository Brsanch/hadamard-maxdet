"""H41 cycle 51 v4: C-extension-driven SBIBD search.

Same algorithm as cycle50 / cycle51 v1; same canonical (B1, B2). Hot path
runs in compiled C (sbibd_search.so) — no Python interpreter overhead in
the inner DFS. Expected ~50-100x speedup over pure Python.

Verified correct on (25, 9, 3): produces identical block list to cycle 50.
"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np

from hadamard_maxdet.verifier import integer_determinant


REPO = Path(__file__).resolve().parents[1]
SO = REPO / "scripts" / "sbibd_search.so"
RESULTS = REPO / "results"

_lib = ctypes.CDLL(str(SO))
_lib.sbibd_search.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_long, ctypes.c_int,
    ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_double),
]
_lib.sbibd_search.restype = ctypes.c_int


def ew_bound(n: int) -> int:
    s2 = 2 * n - 1
    s = int(round(s2 ** 0.5))
    assert s * s == s2
    return s * (n - 1) ** ((n - 1) // 2)


def search_sbibd_c(v: int, k: int, lam: int, *,
                   n_initial: int = 1,
                   progress_every: int = 1_000_000,
                   verbose: bool = True) -> list[list[int]] | None:
    """Search for an SBIBD via C extension.

    n_initial=1: pass only B1 = {0..k-1}. The C enumerates B2 candidates
                 lex-asc starting from {0..lam-1, k..2k-lam-1}; if that
                 fails, it tries the next B2 candidate, and so on. This
                 widens the search to all B2 forms with |B2 ∩ B1| = lam,
                 not just the canonical lex-min.
    n_initial=2: original canonical (B1, B2). Faster but assumes canonical
                 form extends.
    """
    B1 = list(range(k))
    B2 = list(range(lam)) + list(range(k, 2 * k - lam))
    if n_initial == 1:
        initial = (ctypes.c_int * (1 * k))(*B1)
    else:
        initial = (ctypes.c_int * (2 * k))(*B1, *B2)
    out = (ctypes.c_int * (v * k))()
    nodes = ctypes.c_long(0)
    elapsed = ctypes.c_double(0.0)
    rc = _lib.sbibd_search(
        v, k, lam,
        initial, n_initial,
        out,
        progress_every, 1 if verbose else 0,
        ctypes.byref(nodes), ctypes.byref(elapsed),
    )
    print(f"[c] returncode={rc}  nodes={nodes.value:_}  elapsed={elapsed.value:.3f}s", flush=True)
    if rc != 1:
        return None
    blocks = []
    for t in range(v):
        blocks.append([out[t * k + i] for i in range(k)])
    return blocks


def verify_sbibd(N, k: int, lam: int) -> bool:
    A = np.array(N, dtype=np.int64)
    v = A.shape[0]
    AAt = A @ A.T
    expected = (k - lam) * np.eye(v, dtype=np.int64) + lam * np.ones((v, v), dtype=np.int64)
    return np.array_equal(AAt, expected)


def main(v: int = 41, k: int = 16, lam: int = 6, n_initial: int = 1) -> None:
    EW = ew_bound(v)
    print(f"[c] EW({v}) = {EW} ({EW:.4e})")
    print(f"[c] Searching for ({v}, {k}, {lam}) SBIBD via C extension (n_initial={n_initial})...")

    blocks = search_sbibd_c(v, k, lam, n_initial=n_initial, progress_every=10_000_000, verbose=True)
    if blocks is None:
        print("[c] EXHAUSTED — no SBIBD found from canonical (B1, B2).")
        return

    N = [[0] * v for _ in range(v)]
    for j, B in enumerate(blocks):
        for p in B:
            N[p][j] = 1
    if not verify_sbibd(N, k, lam):
        print("[c] SBIBD verify FAILED")
        return
    print(f"[c] N N^T = {k - lam} I + {lam} J ✓")

    M = [[2 * N[i][j] - 1 for j in range(v)] for i in range(v)]
    det = abs(integer_determinant(M))
    ratio = det / EW
    print(f"[c] |det| = {det}")
    print(f"[c] EW    = {EW}")
    print(f"[c] ratio = {ratio}")
    print(f"[c] EW-tight = {det == EW}")

    out = {
        "n": v, "k": k, "lam": lam, "ew": EW, "det": det, "ratio": ratio,
        "ew_tight": det == EW, "blocks": blocks,
        "construction": f"({v}, {k}, {lam}) SBIBD via C-extension point-wise backtracking",
    }
    suffix = f"H{v}_cycle51_sbibd"
    (RESULTS / f"{suffix}.json").write_text(json.dumps(out, indent=2))
    np.save(RESULTS / f"{suffix}.npy", np.array(M, dtype=np.int8))
    print(f"\n[c] Saved.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        v, k, lam = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        n_initial = int(sys.argv[4]) if len(sys.argv) >= 5 else 1
        main(v, k, lam, n_initial=n_initial)
    else:
        main()
