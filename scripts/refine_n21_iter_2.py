"""Iteration 2: GS array (Goethals–Seidel) base-sequence search at n=4m+1=21.

Hypothesis: GS arrays at m=5 use 4 length-5 generators. The bordered
GS at order 4m+1 = 21 is a structurally rich ansatz that has produced
the proven optimum at several n=4m+1 orders. The 4-quad enum at m=5
covers 2^20 ≈ 1M configurations with default (all-+1) border —
tractable in a few minutes parallel.

Then SA over (4m + 8m + 1) = 61 free bits (full border) seeded from
enum-best.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from hadamard_maxdet.n21_gs_array import (
    bordered_gs,
    enumerate_gs_quads,
    sa_bordered_gs,
)
from hadamard_maxdet.verifier import integer_determinant


def _bits_m(idx: int, m: int) -> list[int]:
    return [1 if (idx >> k) & 1 else -1 for k in range(m)]


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "refine_n21_iter_2.json"

    print("=== Iteration 2: GS array, n=21 (m=5) ===")
    print()

    # Part A: exhaustive enum of all 4-quad generators (default border)
    m = 5
    print(f"Part A: enum of (2^{m})^4 = 2^20 ≈ 1M quads with default border.")
    t0 = time.monotonic()
    enum_best, enum_idx = enumerate_gs_quads(m=m)
    enum_wall = time.monotonic() - t0
    a, b, c, d = (_bits_m(i, m) for i in enum_idx)
    M_enum = bordered_gs(a, b, c, d)
    enum_verified = abs(integer_determinant(M_enum.entries))
    assert enum_verified == enum_best
    print(f"  enum best |det| = {enum_best:,}  walltime = {enum_wall:.1f}s")
    print(f"  best (a,b,c,d idx) = {enum_idx}")

    # Part B: SA over the full (61-bit) bordered_gs space, multi-start, 60s
    print()
    print("Part B: SA over 61-bit (4m + 8m + 1) bordered_gs space, 60s.")
    t0 = time.monotonic()
    sa_best = 0
    sa_best_M = None
    n_starts = 0
    seed_off = 0
    while time.monotonic() - t0 < 60.0:
        d_, _tele, M_ = sa_bordered_gs(
            m=m,
            iterations=8_000,
            t_start=1.0,
            t_end=0.005,
            seed=seed_off,
            full_border=True,
        )
        if d_ > sa_best:
            sa_best = d_
            sa_best_M = M_
        n_starts += 1
        seed_off += 1
    sa_wall = time.monotonic() - t0
    print(f"  SA best |det| = {sa_best:,}  walltime = {sa_wall:.1f}s  starts = {n_starts}")
    if sa_best_M is not None:
        assert abs(integer_determinant(sa_best_M.entries)) == sa_best

    # Part C: SA seeded from enum-best (full border)
    print()
    print("Part C: SA seeded from enum-best, full-border, 30s.")
    t0 = time.monotonic()
    seed_best = enum_best
    seed_best_M = M_enum
    seed_off = 1000
    while time.monotonic() - t0 < 30.0:
        d_, _tele, M_ = sa_bordered_gs(
            m=m,
            iterations=8_000,
            t_start=0.05,
            t_end=0.001,
            seed=seed_off,
            full_border=True,
            initial_quad=(a, b, c, d),
        )
        if d_ > seed_best:
            seed_best = d_
            seed_best_M = M_
        seed_off += 1
    seed_wall = time.monotonic() - t0
    print(f"  Seeded SA best |det| = {seed_best:,}  walltime = {seed_wall:.1f}s")

    # Overall
    overall = max(enum_best, sa_best, seed_best)
    if overall == enum_best:
        winner_M, winner = M_enum, "enum_default_border"
    elif overall == seed_best:
        winner_M, winner = seed_best_M, "sa_seeded_full_border"
    else:
        winner_M, winner = sa_best_M, "sa_random_full_border"

    overall_verified = abs(integer_determinant(winner_M.entries))
    assert overall_verified == overall

    proven = 59_392_000_000_000
    ratio = overall / proven
    print()
    print(f"=== Iteration 2 summary ===")
    print(f"  enum default-border         = {enum_best:,}")
    print(f"  SA random full-border       = {sa_best:,}")
    print(f"  SA seeded from enum, full-b = {seed_best:,}")
    print(f"  overall best                = {overall:,} ({winner})")
    print(f"  proven n=21 max             = {proven:,}")
    print(f"  ratio                       = {ratio:.4f}")
    print(f"  prior pipeline best         = 51,200,000,000,000  (0.8621)")
    print(f"  iter-1 (block-cyclic) best  = 41,248,865,910,784  (0.6945)")

    payload = {
        "iteration": 2,
        "hypothesis": "Goethals–Seidel array at m=5; bordered GS = order 21",
        "n": 21,
        "proven_max": proven,
        "enum_best_det": enum_best,
        "enum_best_idx": list(enum_idx),
        "enum_walltime_s": enum_wall,
        "sa_random_best_det": sa_best,
        "sa_random_walltime_s": sa_wall,
        "sa_random_n_starts": n_starts,
        "sa_seeded_best_det": seed_best,
        "sa_seeded_walltime_s": seed_wall,
        "overall_best_det": overall,
        "overall_winner": winner,
        "overall_ratio": ratio,
        "winner_matrix": winner_M.entries,
    }
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
