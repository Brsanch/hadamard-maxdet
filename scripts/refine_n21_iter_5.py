"""Iteration 5: bordered "negacyclic" two-block ansatz at n=21.

Hypothesis: The bordered_two_circulant family contains BTC at 51.2T,
but its inner [[A,B],[B^T,-A^T]] block is built from ordinary
circulant A, B. Replacing them with NEGAcirculants changes the
inner-block orthogonality structure (group algebra over x^m + 1
vs x^m - 1). Different basin of attraction; potentially different
ceiling.

For n=21=2m+1 with m=10. Free bits = 2m + 4m + 1 = 41 with full border.

We run:
  Part A: SA over full-border bordered_negacyclic_two, multi-start,
          90s budget.
  Part B: 4-replica parallel-tempering-light over the same space,
          30s. (Lightweight: 4 chains at log-spaced T, swap every 200
          steps.)
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from hadamard_maxdet.n21_negacyclic import (
    bordered_negacyclic_two,
    sa_bordered_negacyclic_two,
)
from hadamard_maxdet.verifier import integer_determinant


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "refine_n21_iter_5.json"

    print("=== Iteration 5: bordered negacyclic two-block, n=21 ===")
    print()

    # Part A: SA multi-start
    print("Part A: SA multi-start over 41-bit space (90s).")
    t0 = time.monotonic()
    sa_best = 0
    sa_best_M = None
    n_starts = 0
    seed_off = 0
    while time.monotonic() - t0 < 90.0:
        d, _tele, M = sa_bordered_negacyclic_two(
            n=21,
            iterations=8000,
            t_start=1.0,
            t_end=0.005,
            seed=seed_off,
            full_border=True,
        )
        if d > sa_best:
            sa_best = d
            sa_best_M = M
        n_starts += 1
        seed_off += 1
    sa_wall = time.monotonic() - t0
    print(f"  best |det| = {sa_best:,}  walltime = {sa_wall:.1f}s  starts = {n_starts}")
    if sa_best_M is not None:
        assert abs(integer_determinant(sa_best_M.entries)) == sa_best

    # Part B: long single SA with very-cold tail
    print()
    print("Part B: long single SA with very-cold tail (30s).")
    t0 = time.monotonic()
    long_best = 0
    long_best_M = None
    seed_off = 1000
    while time.monotonic() - t0 < 30.0:
        d, _tele, M = sa_bordered_negacyclic_two(
            n=21,
            iterations=30_000,
            t_start=2.0,
            t_end=0.001,
            seed=seed_off,
            full_border=True,
        )
        if d > long_best:
            long_best = d
            long_best_M = M
        seed_off += 1
    long_wall = time.monotonic() - t0
    print(f"  best |det| = {long_best:,}  walltime = {long_wall:.1f}s")

    # Part C: lift the negacyclic best to unconstrained ±1 and run sa_incremental
    print()
    print("Part C: lift best to unconstrained ±1 lattice + sa_incremental (30s).")
    from hadamard_maxdet.incremental import sa_incremental
    from hadamard_maxdet.matrix import Matrix

    overall_negacyc_best = max(sa_best, long_best)
    overall_negacyc_M = sa_best_M if sa_best >= long_best else long_best_M

    t0 = time.monotonic()
    lift_best = overall_negacyc_best
    lift_best_M = overall_negacyc_M
    seed_off = 2000
    while time.monotonic() - t0 < 30.0:
        r = sa_incremental(
            21,
            iterations=20_000,
            t_start=0.05,
            t_end=0.001,
            seed=seed_off,
            initial=Matrix(n=21, entries=[row[:] for row in lift_best_M.entries]),
            refresh_every=2000,
        )
        if r.abs_det > lift_best:
            lift_best = r.abs_det
            lift_best_M = r.matrix
        seed_off += 1
    lift_wall = time.monotonic() - t0
    print(f"  best |det| = {lift_best:,}  walltime = {lift_wall:.1f}s")

    overall = max(sa_best, long_best, lift_best)
    if overall == lift_best and lift_best > overall_negacyc_best:
        winner_M = lift_best_M
        winner = "negacyclic_lifted_to_pm1_via_sa_incremental"
    elif overall == sa_best:
        winner_M = sa_best_M
        winner = "sa_negacyclic_multistart"
    else:
        winner_M = long_best_M
        winner = "sa_negacyclic_long_cold_tail"

    overall_verified = abs(integer_determinant(winner_M.entries))
    assert overall_verified == overall

    proven = 59_392_000_000_000
    ratio = overall / proven
    print()
    print(f"=== Iteration 5 summary ===")
    print(f"  SA neg-multi (90s)     = {sa_best:,}")
    print(f"  SA neg-long-cold (30s) = {long_best:,}")
    print(f"  lifted via sa_incr     = {lift_best:,}")
    print(f"  overall                = {overall:,} ({winner})")
    print(f"  proven n=21 max        = {proven:,}")
    print(f"  ratio                  = {ratio:.4f}")
    print(f"  prior pipeline best    = 51,200,000,000,000  (0.8621)")

    payload = {
        "iteration": 5,
        "hypothesis": "bordered negacyclic two-block (negacirculant A, B blocks)",
        "n": 21,
        "proven_max": proven,
        "sa_negacyc_multistart_det": sa_best,
        "sa_negacyc_multistart_walltime_s": sa_wall,
        "sa_negacyc_multistart_n_starts": n_starts,
        "sa_negacyc_long_det": long_best,
        "sa_negacyc_long_walltime_s": long_wall,
        "lifted_via_sa_incremental_det": lift_best,
        "lifted_walltime_s": lift_wall,
        "overall_best_det": overall,
        "overall_best_label": winner,
        "overall_ratio": ratio,
        "winner_matrix": winner_M.entries,
    }
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
