"""Run parallel tempering search for H23 maxdet and save results."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hadamard_maxdet.parallel_tempering import (
    ORRICK_ABS_DET_H23,
    DEFAULT_TEMPS,
    parallel_tempering,
    triple_verify,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel tempering (Replica Exchange MC) search for H23 maxdet"
    )
    parser.add_argument("--n-replicas", type=int, default=5)
    parser.add_argument(
        "--temps",
        type=float,
        nargs="+",
        default=None,
        help="Temperature list high-to-low; overrides --n-replicas default ladder",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--wall-cap", type=float, default=600.0, help="Wall cap in seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default=os.path.join(RESULTS_DIR, "parallel_tempering_h23.json"),
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--verbose-interval", type=float, default=30.0)
    args = parser.parse_args()

    temps = args.temps
    if temps is None:
        if args.n_replicas == 5:
            temps = list(DEFAULT_TEMPS)
        else:
            import math
            t_high, t_low = 2.0, 0.003
            r = args.n_replicas
            temps = [
                t_high * (t_low / t_high) ** (k / max(r - 1, 1))
                for k in range(r)
            ]

    print(f"Parallel Tempering — H23 maxdet search")
    print(f"  Replicas: {len(temps)}, temperatures: {temps}")
    print(f"  Steps/epoch: {args.steps_per_epoch}")
    print(f"  Wall cap: {args.wall_cap:.0f}s")
    print(f"  Seed: {args.seed}")
    print(f"  Orrick record: {ORRICK_ABS_DET_H23:,}")
    print()

    result = parallel_tempering(
        n=23,
        temps=temps,
        steps_per_epoch=args.steps_per_epoch,
        wall_cap_s=args.wall_cap,
        seed=args.seed,
        verbose=args.verbose,
        verbose_interval_s=args.verbose_interval,
    )

    print()
    print("=" * 70)
    print(f"Run complete: {result.walltime_seconds:.1f}s")
    print(f"  Epochs completed:    {result.epochs_completed:,}")
    print(f"  Total SA steps:      {result.total_sa_steps:,}")
    print(f"  SA throughput:       {result.total_sa_steps / result.walltime_seconds:.0f} steps/s")
    print(f"  Swap attempts:       {result.total_swap_attempts:,}")
    print(f"  Swaps accepted:      {result.swaps_accepted:,}")
    print(f"  Overall swap rate:   {result.swaps_accepted / max(1, result.total_swap_attempts):.3f}")
    for k, rate in enumerate(result.swap_rates_by_pair):
        print(f"    Pair T={temps[k]:.3f}↔T={temps[k+1]:.3f}: {rate:.3f}")
    print()
    print(f"  Best |det|:  {result.best_abs_det:,}")
    print(f"  Ratio:       {result.best_ratio_to_orrick:.6f} × Orrick")
    print(f"  Per-replica: {result.per_replica_best}")

    # Triple verify best matrix
    if result.best_matrix:
        det_M, det_MT, det_copy = triple_verify(result.best_matrix)
        print()
        print(f"  Triple verify: det(M)={det_M:,}  det(M^T)={det_MT:,}  "
              f"det(copy)={det_copy:,}")
        consistent = abs(det_M) == abs(det_MT) == abs(det_copy) == result.best_abs_det
        print(f"  Consistent: {consistent}")

        if result.best_abs_det > ORRICK_ABS_DET_H23:
            print()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("  *** NEW RECORD: |det| > Orrick (2003) ***")
            print(f"  |det| = {result.best_abs_det:,}")
            print(f"  Ratio = {result.best_ratio_to_orrick:.8f}")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "algorithm": "parallel_tempering",
        "domain_analogy": "thermodynamics",
        "n": 23,
        "temps_used": result.temps_used,
        "steps_per_epoch": args.steps_per_epoch,
        "seed": args.seed,
        "wall_cap_s": args.wall_cap,
        "best_abs_det": result.best_abs_det,
        "best_ratio_to_orrick": result.best_ratio_to_orrick,
        "orrick_abs_det": ORRICK_ABS_DET_H23,
        "above_orrick": result.best_abs_det > ORRICK_ABS_DET_H23,
        "epochs_completed": result.epochs_completed,
        "total_sa_steps": result.total_sa_steps,
        "total_swap_attempts": result.total_swap_attempts,
        "swaps_accepted": result.swaps_accepted,
        "swap_rates_by_pair": result.swap_rates_by_pair,
        "per_replica_best": result.per_replica_best,
        "walltime_seconds": result.walltime_seconds,
        "best_matrix": result.best_matrix if result.best_matrix else [],
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
