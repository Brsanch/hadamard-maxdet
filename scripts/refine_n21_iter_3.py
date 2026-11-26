"""Iteration 3: k-flip random sampling around the n=21 BTC seed.

The H23 diagnostic deep-dive established the BTC seed at n=23 is a strict
local maximum under Hamming-1, -2, AND -3 entry flips. We replicate the
test at n=21 first (cheap), then go to k=4 and k=5 sampling. The energy
profile around the seed at d=1,2,3 is recorded; if there exists ANY
4-flip improving neighbour, we found a new basin entry.

For n=21, single-flip neighbour count = 21*21 = 441; sample 5000 each
of d=2 and d=3 (random); sample 50_000 of d=4 in 60s budget.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.rich_ansatz import sa_bordered_two_circulant
from hadamard_maxdet.verifier import integer_determinant


def random_k_flip(M: list[list[int]], rng: random.Random, k: int) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """Return a copy of M with k *distinct* random entry flips applied."""
    n = len(M)
    positions: set[tuple[int, int]] = set()
    while len(positions) < k:
        positions.add((rng.randrange(n), rng.randrange(n)))
    pos_list = list(positions)
    out = [row[:] for row in M]
    for (i, j) in pos_list:
        out[i][j] = -out[i][j]
    return out, pos_list


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "refine_n21_iter_3.json"

    print("=== Iteration 3: k-flip energy profile around n=21 BTC seed ===")
    print()

    # Get the BTC seed (51.2T)
    print("Re-deriving BTC n=21 seed (51.2T)...")
    t0 = time.monotonic()
    r_btc = sa_bordered_two_circulant(21, iterations=15_000, t_start=1.0, t_end=0.005, seed=0, full_border=True)
    seed_M = r_btc.matrix.entries
    seed_det = abs(integer_determinant(seed_M))
    print(f"  seed |det| = {seed_det:,}  walltime = {time.monotonic()-t0:.1f}s")

    # k=1 exhaustive
    n = 21
    print()
    print("k=1 exhaustive (441 neighbours)...")
    t0 = time.monotonic()
    above_1 = 0
    eq_1 = 0
    below_1 = 0
    max_above_1 = 0
    min_d1 = float("inf")
    max_d1 = 0
    best_above_pos: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            seed_M[i][j] = -seed_M[i][j]
            d = abs(integer_determinant(seed_M))
            seed_M[i][j] = -seed_M[i][j]
            if d > seed_det:
                above_1 += 1
                if d > max_above_1:
                    max_above_1 = d
                    best_above_pos = [(i, j)]
            elif d == seed_det:
                eq_1 += 1
            else:
                below_1 += 1
            min_d1 = min(min_d1, d)
            max_d1 = max(max_d1, d)
    print(f"  d=1: above={above_1} eq={eq_1} below={below_1}  max={max_d1:,}  min={min_d1:,}")
    print(f"  d=1 walltime = {time.monotonic()-t0:.2f}s")

    # k=2,3,4,5 random sampling
    rng = random.Random(0)
    sample_results = {}
    for k, n_samples, budget_s in [(2, 5000, 30), (3, 5000, 30), (4, 50000, 60), (5, 50000, 60)]:
        print()
        print(f"k={k} random sampling, target {n_samples} samples, budget {budget_s}s...")
        t0 = time.monotonic()
        above = 0
        eq = 0
        below = 0
        max_above = 0
        max_above_M: list[list[int]] | None = None
        max_above_pos: list[tuple[int, int]] | None = None
        max_v = 0
        min_v = float("inf")
        actual = 0
        while actual < n_samples and time.monotonic() - t0 < budget_s:
            cand, pos = random_k_flip(seed_M, rng, k)
            d = abs(integer_determinant(cand))
            actual += 1
            if d > seed_det:
                above += 1
                if d > max_above:
                    max_above = d
                    max_above_M = cand
                    max_above_pos = pos
            elif d == seed_det:
                eq += 1
            else:
                below += 1
            max_v = max(max_v, d)
            min_v = min(min_v, d)
        elapsed = time.monotonic() - t0
        print(f"  k={k}: actual_samples={actual}  above={above}  eq={eq}  below={below}")
        print(f"         max={max_v:,}  min={min_v:,}  walltime={elapsed:.1f}s")
        if max_above > 0:
            print(f"  ***** k={k}: FOUND IMPROVING NEIGHBOUR |det|={max_above:,} at {max_above_pos} *****")
        sample_results[k] = {
            "samples": actual,
            "above": above,
            "eq": eq,
            "below": below,
            "max_neighbour": max_v,
            "min_neighbour": min_v,
            "max_above": max_above,
            "max_above_pos": max_above_pos,
            "walltime_s": elapsed,
        }

    # Verdict
    overall_best = seed_det
    best_M = seed_M
    for k, res in sample_results.items():
        if res["max_above"] > overall_best:
            overall_best = res["max_above"]
            # Reconstruct best
            cand = [row[:] for row in seed_M]
            for (i, j) in res["max_above_pos"]:
                cand[i][j] = -cand[i][j]
            best_M = cand

    if overall_best > seed_det:
        verified = abs(integer_determinant(best_M))
        assert verified == overall_best
        print()
        print(f"*** NEW BEST via k-flip sampling: |det|={overall_best:,} ***")

    proven = 59_392_000_000_000
    ratio = overall_best / proven
    print()
    print(f"=== Iteration 3 summary ===")
    print(f"  seed (BTC) |det|        = {seed_det:,}")
    print(f"  k=1 best neighbour      = {max_d1:,}  (above={above_1})")
    for k in (2, 3, 4, 5):
        r = sample_results[k]
        print(f"  k={k} max neighbour       = {r['max_neighbour']:,}  (above={r['above']})")
    print(f"  overall best            = {overall_best:,}")
    print(f"  proven n=21 max         = {proven:,}")
    print(f"  ratio                   = {ratio:.4f}")

    payload = {
        "iteration": 3,
        "hypothesis": "k-flip energy profile (k∈{1,2,3,4,5}) around n=21 BTC seed",
        "n": 21,
        "proven_max": proven,
        "seed_det": seed_det,
        "k1": {
            "neighbours": n * n,
            "above": above_1,
            "eq": eq_1,
            "below": below_1,
            "max_neighbour": max_d1,
            "min_neighbour": min_d1,
            "max_above": max_above_1,
            "max_above_pos": best_above_pos,
        },
        "k_samples": sample_results,
        "overall_best_det": overall_best,
        "overall_ratio": ratio,
    }
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
