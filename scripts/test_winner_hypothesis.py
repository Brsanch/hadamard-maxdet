"""Test three structural hypotheses for what distinguishes proven-winner
matrices from our SA-found bests at H19+.

For each of 5 matrices (proven n=17, proven n=15, our H19 best, our H23
BTC, random ±1 at n=23), compute the three diagnostics from
:mod:`hadamard_maxdet.structural_test` and print a comparison table.
Save full results to ``results/structural_test.json``.

Usage:
    python scripts/test_winner_hypothesis.py [--budget 90]

The proven n=15 / n=17 matrices are not stored verbatim in
``winner_decode.json`` (only fingerprints are), so we re-derive them
via ``winner_fingerprint`` (SA multistart, deterministic seed). This
is fast for n <= 17 (a few seconds each at default budget).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.structural_test import compare_matrices
from hadamard_maxdet.verifier import integer_determinant
from hadamard_maxdet.winner_decode import winner_fingerprint


def _load_proven(n: int, budget_seconds: float) -> tuple[list[list[int]], int]:
    """Re-derive a proven-max matrix at order ``n`` via SA multistart."""
    print(f"  re-deriving proven n={n} via winner_fingerprint ...", flush=True)
    t0 = time.monotonic()
    fp = winner_fingerprint(
        n,
        n_starts=8,
        iterations=80_000,
        structured_seconds=min(budget_seconds, 30.0),
        base_seed=12345,
    )
    elapsed = time.monotonic() - t0
    abs_det = abs(integer_determinant(fp.matrix))
    print(
        f"  -> n={n} reached |det|={abs_det:,d} "
        f"(target {fp.proven_max:,d}, ratio {abs_det / fp.proven_max:.4f}) "
        f"in {elapsed:.1f}s",
        flush=True,
    )
    return fp.matrix, abs_det


def _load_h19_best() -> tuple[list[list[int]], int]:
    path = REPO / "results" / "H19_diagnostic_sa_best.json"
    with path.open() as f:
        d = json.load(f)
    return d["matrix"], int(d["abs_det"])


def _load_h23_btc() -> tuple[list[list[int]], int]:
    path = REPO / "results" / "H23_rich_ansatz_best.json"
    with path.open() as f:
        d = json.load(f)
    return d["entries"], int(d["abs_det"])


def _random_h23(seed: int = 42) -> tuple[list[list[int]], int]:
    rng = random.Random(seed)
    n = 23
    M = [[1 if rng.random() < 0.5 else -1 for _ in range(n)] for _ in range(n)]
    abs_det = abs(integer_determinant(M))
    return M, abs_det


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=60.0,
                        help="Per-proven-n re-derivation budget (seconds)")
    parser.add_argument("--out", type=str,
                        default=str(REPO / "results" / "structural_test.json"))
    args = parser.parse_args()

    print("[1/5] Loading proven n=17 ...", flush=True)
    M17, det17 = _load_proven(17, args.budget)

    print("[2/5] Loading proven n=15 ...", flush=True)
    M15, det15 = _load_proven(15, args.budget)

    print("[3/5] Loading H19 best (diagnostic SA) ...", flush=True)
    M19, det19 = _load_h19_best()
    print(f"  H19 |det|={det19:,d}", flush=True)

    print("[4/5] Loading H23 BTC ...", flush=True)
    M23btc, det23btc = _load_h23_btc()
    print(f"  H23 BTC |det|={det23btc:,d}", flush=True)

    print("[5/5] Generating random H23 control ...", flush=True)
    M23rnd, det23rnd = _random_h23(seed=42)
    print(f"  H23 random |det|={det23rnd:,d}", flush=True)

    matrices = [M17, M15, M19, M23btc, M23rnd]
    labels = [
        "proven n=17 (1.00)",
        "proven n=15 (1.00)",
        "our H19 (0.92)",
        "BTC H23 (1.00 BTC)",
        "random H23 (~?)",
    ]
    abs_dets = [det17, det15, det19, det23btc, det23rnd]

    print("\n" + "=" * 110)
    print("Structural diagnostic comparison")
    print("=" * 110)
    rows = compare_matrices(matrices, labels, seed=0, offset=1)

    # Augment with abs_det.
    for r, d in zip(rows, abs_dets):
        r["abs_det"] = int(d)

    # Verdict.
    print("\n" + "=" * 110)
    print("Verdict per hypothesis")
    print("=" * 110)
    proven_idx = [0, 1]
    near_idx = [2, 3]
    rand_idx = [4]

    def _summary(values, name):
        vp = [values[i] for i in proven_idx]
        vn = [values[i] for i in near_idx]
        vr = [values[i] for i in rand_idx]
        return (
            f"  {name:30s}  proven={vp}  near={vn}  random={vr}"
        )

    h1_vals = [r["h1_cluster_score"] for r in rows]
    h2_vals = [r["h2_modularity"] for r in rows]
    h3_vals = [r["h3_score"] for r in rows]
    print(_summary([f"{v:+.3f}" for v in h1_vals], "H1 cluster score (lower=better)"))
    print(_summary([f"{v:.4f}" for v in h2_vals], "H2 modularity   (higher=better)"))
    print(_summary([f"{v:.4f}" for v in h3_vals], "H3 symmetry     (higher=better)"))

    # Crude separation rule: discriminator if proven > near > random
    # (or the reverse for H1 where lower is better).
    def _discriminates(values, lower_is_better: bool) -> tuple[bool, str]:
        vp_min = min(values[i] for i in proven_idx)
        vp_max = max(values[i] for i in proven_idx)
        vn_max = max(values[i] for i in near_idx)
        vn_min = min(values[i] for i in near_idx)
        vr = values[rand_idx[0]]
        if lower_is_better:
            sep_pn = vp_max < vn_min
            sep_nr = vn_max < vr
        else:
            sep_pn = vp_min > vn_max
            sep_nr = vn_min > vr
        ok = sep_pn and sep_nr
        return ok, f"proven_vs_near={'OK' if sep_pn else 'NO'}, near_vs_random={'OK' if sep_nr else 'NO'}"

    print()
    for name, values, lower in [
        ("H1 cluster", h1_vals, True),
        ("H2 modularity", h2_vals, False),
        ("H3 symmetry", h3_vals, False),
    ]:
        ok, expl = _discriminates(values, lower_is_better=lower)
        verdict = "DISCRIMINATES" if ok else "does NOT cleanly discriminate"
        print(f"  {name:18s} -> {verdict}  ({expl})")

    # Save JSON.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "comparison": rows,
        "labels": labels,
        "abs_dets": abs_dets,
        "discriminates": {
            "h1_cluster": _discriminates(h1_vals, True)[0],
            "h2_modularity": _discriminates(h2_vals, False)[0],
            "h3_symmetry": _discriminates(h3_vals, False)[0],
        },
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
