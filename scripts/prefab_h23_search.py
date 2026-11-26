"""H23 prefab block-assembly search.

For each of three (a, b) decompositions of n = 23, run prefab simulated
annealing over the off-diagonal "glue" blocks while holding the diagonal
blocks at proven-max ±1 matrices:

    decomposition       a   b
    -----------        --  --
    12 + 11            12  11   (Paley I H12, proven H11 winner)
    16 +  7            16   7   (Sylvester H16, proven H7 winner)
    17 +  6            17   6   (proven H17 winner, proven H6 winner)

Reports per decomposition:
  best |det|, ratio to BTC (2.626e15) and Orrick (2.778e15), and the
  Gram fingerprint of the best assembled matrix (does it land in the
  high-SV-multiplicity regime that records share?).

Saves the best matrix overall to results/H23_prefab_best.json.

Usage:
  python scripts/prefab_h23_search.py [--seconds 300] [--starts 6]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running from repo root or scripts/.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from hadamard_maxdet.prefab import (  # noqa: E402
    load_proven_matrix,
    prefab_sa,
)
from hadamard_maxdet.verifier import integer_determinant  # noqa: E402
from hadamard_maxdet.winner_decode import fingerprint_matrix  # noqa: E402


BTC_BASELINE = 2_626_567_700_217_856   # in-repo bordered_two_circulant SA
ORRICK_BEST = 2_778_001_640_357_888    # 0.7091 · Barba H23 (Browne et al. 2021)


DECOMPOSITIONS: list[tuple[int, int, str]] = [
    (12, 11, "Paley H12 + proven H11"),
    (16,  7, "Sylvester H16 + proven H7"),
    (17,  6, "proven H17 + proven H6"),
]


def run_one_decomposition(
    a: int,
    b: int,
    label: str,
    *,
    wall_seconds: float,
    n_starts: int,
    iters_per_start: int,
    base_seed: int,
) -> dict:
    """Run a budgeted multi-start prefab SA on (a, b)."""
    print(f"\n=== prefab decomposition a={a}, b={b}  ({label}) ===")
    t_load = time.monotonic()
    A = load_proven_matrix(a)
    D = load_proven_matrix(b)
    print(
        f"  loaded A ({a}×{a}, |det|={abs(integer_determinant(A))}) "
        f"and D ({b}×{b}, |det|={abs(integer_determinant(D))}) "
        f"in {time.monotonic() - t_load:.1f}s"
    )

    deadline = time.monotonic() + wall_seconds
    best_det = 0
    best_full: list[list[int]] = []
    best_B: list[list[int]] = []
    best_C: list[list[int]] = []

    s = 0
    runs = 0
    while time.monotonic() < deadline and s < n_starts * 4:
        seed = base_seed + s * 7919
        t0 = time.monotonic()
        r = prefab_sa(
            a, b, A=A, D=D,
            iterations=iters_per_start,
            t_start=1.5, t_end=0.002,
            seed=seed,
        )
        runs += 1
        if r.abs_det > best_det:
            best_det = r.abs_det
            best_full = r.full_matrix
            best_B = r.B
            best_C = r.C
            tag = "  ↑"
        else:
            tag = "   "
        print(
            f"  start {s:2d} seed={seed:8d} "
            f"|det|={r.abs_det:.6e} (best {best_det:.6e}) "
            f"acc={r.accepted}/{r.iterations} "
            f"in {time.monotonic() - t0:.1f}s {tag}"
        )
        s += 1

    ratio_btc = best_det / BTC_BASELINE if BTC_BASELINE else float("nan")
    ratio_orrick = best_det / ORRICK_BEST if ORRICK_BEST else float("nan")
    print(
        f"  → best |det| = {best_det:.6e}  "
        f"({ratio_btc * 100:.2f}% of BTC, {ratio_orrick * 100:.2f}% of Orrick)"
    )

    # Fingerprint diagnostic.
    fp = fingerprint_matrix(best_full, source=f"prefab_{a}+{b}")
    sv_mults = sorted(fp.sv_multiplicities, key=lambda kv: -kv[1])
    top_mult = sv_mults[0][1] if sv_mults else 0
    print(
        f"  fingerprint: gram_offdiag_support={fp.gram_offdiag_support}  "
        f"sv_distinct={fp.sv_distinct_count}  top_sv_mult={top_mult}"
    )

    return {
        "a": a, "b": b, "label": label,
        "abs_det": best_det,
        "ratio_btc": ratio_btc,
        "ratio_orrick": ratio_orrick,
        "runs": runs,
        "fingerprint": fp.to_json(include_matrix=False),
        "matrix": best_full,
        "B": best_B,
        "C": best_C,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=300.0,
                        help="Wall budget per decomposition (default 300).")
    parser.add_argument("--starts", type=int, default=6,
                        help="Max number of multi-starts per decomposition.")
    parser.add_argument("--iters", type=int, default=80_000,
                        help="SA iterations per start.")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="PRNG seed offset.")
    parser.add_argument("--out", type=str,
                        default=str(ROOT / "results" / "H23_prefab_best.json"))
    args = parser.parse_args()

    print(f"BTC baseline:    {BTC_BASELINE:.6e}")
    print(f"Orrick:          {ORRICK_BEST:.6e}")
    print(f"Wall per decomp: {args.seconds}s")

    results = []
    for a, b, label in DECOMPOSITIONS:
        res = run_one_decomposition(
            a, b, label,
            wall_seconds=args.seconds,
            n_starts=args.starts,
            iters_per_start=args.iters,
            base_seed=args.base_seed + 1000 * a,
        )
        results.append(res)

    print("\n=== SUMMARY ===")
    print(f"{'split':<10} {'best |det|':>16} {'%BTC':>8} {'%Orrick':>10} {'top_sv_mult':>12}")
    for r in results:
        top_mult = r["fingerprint"]["sv_multiplicities"][0][1] if r["fingerprint"]["sv_multiplicities"] else 0
        print(
            f"{r['a']:>2}+{r['b']:<6} "
            f"{r['abs_det']:>16.6e} "
            f"{r['ratio_btc'] * 100:>7.2f}% "
            f"{r['ratio_orrick'] * 100:>9.2f}% "
            f"{top_mult:>12d}"
        )

    best_r = max(results, key=lambda r: r["abs_det"])
    payload = {
        "best": best_r,
        "all": [
            {k: v for k, v in r.items() if k not in ("matrix", "B", "C")}
            for r in results
        ],
        "btc_baseline": BTC_BASELINE,
        "orrick_best": ORRICK_BEST,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nbest matrix saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
