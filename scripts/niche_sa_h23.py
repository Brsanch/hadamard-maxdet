"""Competitive Exclusion Niche SA experiment for H23.

Ecology domain analogy: Competitive Exclusion Principle (Gause 1934) +
Island Biogeography (MacArthur-Wilson 1967).

Usage:
  python scripts/niche_sa_h23.py [--budget 600] [--sa-iters 80000] [--seed 20260426]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from hadamard_maxdet.niche_sa import (
    ORRICK_ABS_DET_H23,
    niche_sa_search,
    triple_verify,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Niche SA H23 experiment")
    parser.add_argument("--budget", type=float, default=600.0, help="Wall-clock budget in seconds")
    parser.add_argument("--sa-iters", type=int, default=80_000, help="SA iterations per chain")
    parser.add_argument("--greedy-iters", type=int, default=1, help="Greedy passes per chain")
    parser.add_argument("--exploit-prob", type=float, default=0.30, help="Exploitation probability")
    parser.add_argument("--island-exp", type=float, default=0.25, help="Island-area exponent")
    parser.add_argument("--seed", type=int, default=20260426, help="Random seed")
    parser.add_argument("--out", type=str, default="results/niche_sa_h23.json", help="Output JSON path")
    args = parser.parse_args()

    print(f"=== Competitive Exclusion Niche SA — H23 ===")
    print(f"Budget:          {args.budget:.0f} s")
    print(f"SA iters/chain:  {args.sa_iters:,}")
    print(f"Greedy passes:   {args.greedy_iters}")
    print(f"Exploit prob:    {args.exploit_prob}")
    print(f"Island exponent: {args.island_exp}")
    print(f"Seed:            {args.seed}")
    print(f"Orrick record:   {ORRICK_ABS_DET_H23:,}")
    print()

    t0 = time.monotonic()
    result = niche_sa_search(
        n=23,
        budget_seconds=args.budget,
        sa_iters=args.sa_iters,
        greedy_iters=args.greedy_iters,
        exploit_prob=args.exploit_prob,
        island_exp=args.island_exp,
        seed=args.seed,
    )
    elapsed = time.monotonic() - t0

    print(f"=== Results ===")
    print(f"Elapsed:           {elapsed:.1f} s")
    print(f"Total chains:      {result.total_chains:,}")
    print(f"  Exploration:     {result.exploration_chains:,}")
    print(f"  Exploitation:    {result.exploitation_chains:,}")
    print(f"Community size:    {result.community_size:,} distinct niches")
    print(f"New niches found:  {result.new_niches:,}")
    print(f"Collisions:        {result.collisions:,}")
    print(f"Niche discovery:   {result.new_niches / max(1, result.total_chains):.3f} (new niches / chain)")
    print(f"Collision rate:    {result.collisions / max(1, result.total_chains):.3f}")
    print()
    print(f"Best |det|:        {result.best_abs_det:,}")
    print(f"Ratio to Orrick:   {result.ratio_to_orrick:.6f}")
    print(f"Above Orrick:      {result.above_orrick}")
    print()
    print(f"Community det histogram (buckets [0,0.1) ... [0.9,1.0]):")
    for i, cnt in enumerate(result.niche_det_histogram):
        if cnt:
            lo = i * 0.1
            hi = (i + 1) * 0.1
            print(f"  [{lo:.1f}, {hi:.1f}): {cnt:4d} niches")

    # Triple verification
    print()
    print("=== Triple verification ===")
    tv = triple_verify(result.best_matrix)
    print(f"det(M)   = {tv['det']:,}")
    print(f"det(M^T) = {tv['det_transpose']:,}")
    print(f"det(copy)= {tv['det_copy']:,}")
    print(f"Consistent: {tv['consistent']}")

    if result.above_orrick:
        print()
        print("!!! NEW RECORD — |det| > ORRICK 2,779,447,296,000,000 !!!")

    # Save results
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_abs_det": result.best_abs_det,
        "ratio_to_orrick": result.ratio_to_orrick,
        "above_orrick": result.above_orrick,
        "orrick_record": ORRICK_ABS_DET_H23,
        "total_chains": result.total_chains,
        "exploration_chains": result.exploration_chains,
        "exploitation_chains": result.exploitation_chains,
        "community_size": result.community_size,
        "new_niches": result.new_niches,
        "collisions": result.collisions,
        "niche_discovery_rate": result.new_niches / max(1, result.total_chains),
        "collision_rate": result.collisions / max(1, result.total_chains),
        "elapsed_seconds": result.elapsed_seconds,
        "triple_verify": tv,
        "best_matrix": result.best_matrix,
        "params": {
            "budget_seconds": args.budget,
            "sa_iters": args.sa_iters,
            "greedy_iters": args.greedy_iters,
            "exploit_prob": args.exploit_prob,
            "island_exp": args.island_exp,
            "seed": args.seed,
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
