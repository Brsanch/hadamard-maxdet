"""Experiment driver: Quasispecies error-threshold search at H23.

Domain analogy: evolutionary_biology (Eigen quasispecies model, 1971).

Usage
-----
  python scripts/quasispecies_h23.py [--wall SECONDS] [--seed SEED]
                                     [--pop POP_SIZE] [--alpha ALPHA]
                                     [--mu0 MU0] [--muf MU_FINAL]
                                     [--restart-every N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hadamard_maxdet.quasispecies import (
    ORRICK_ABS_DET_H23,
    QSResult,
    quasispecies_search,
    triple_verify,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quasispecies H23 maxdet search")
    parser.add_argument("--wall", type=float, default=600.0, help="Wall-clock budget (s)")
    parser.add_argument("--seed", type=int, default=20260426, help="RNG seed")
    parser.add_argument("--pop", type=int, default=20, help="Population size")
    parser.add_argument("--alpha", type=float, default=3.0, help="Selection pressure")
    parser.add_argument("--mu0", type=float, default=0.25, help="Initial mutation rate")
    parser.add_argument("--muf", type=float, default=0.001, help="Final mutation rate")
    parser.add_argument("--restart-every", type=int, default=500,
                        help="Restart population every N generations")
    args = parser.parse_args()

    print("=" * 70)
    print("Quasispecies Error-Threshold Search — H23 Maxdet")
    print("Domain: evolutionary_biology (Eigen 1971)")
    print("=" * 70)
    print()

    t0 = time.time()
    result: QSResult = quasispecies_search(
        n=23,
        pop_size=args.pop,
        mu_0=args.mu0,
        mu_final=args.muf,
        alpha=args.alpha,
        restart_every=args.restart_every,
        wall_time=args.wall,
        orrick=ORRICK_ABS_DET_H23,
        seed=args.seed,
        verbose=True,
    )
    wall = time.time() - t0

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total generations    : {result.n_generations:,}")
    print(f"Total restarts       : {result.n_restarts}")
    print(f"Wall time            : {wall:.1f} s")
    print(f"Best |det|           : {result.best_abs_det:,}")
    print(f"Orrick record        : {ORRICK_ABS_DET_H23:,}")
    print(f"Ratio to Orrick      : {result.best_ratio_to_orrick:.6f}")
    print(f"Above Orrick         : {result.above_orrick}")

    # Triple verification
    print()
    print("Triple verification ...")
    d1, d2, d3 = triple_verify(result.best_matrix)
    consistent = (d1 == d2 == d3)
    print(f"  det(M)      = {d1}")
    print(f"  det(M^T)    = {d2}")
    print(f"  det(copy)   = {d3}")
    print(f"  Consistent  : {consistent}")
    if not consistent:
        print("  WARNING: triple verify FAILED!")

    if result.above_orrick:
        print()
        print("!" * 70)
        print("!!! NEW RECORD — |det| > ORRICK 2003 !!!")
        print("!" * 70)

    # Save to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "quasispecies_h23.json"
    payload = {
        "algorithm": "quasispecies_error_threshold",
        "domain": "evolutionary_biology",
        "n": 23,
        "best_abs_det": result.best_abs_det,
        "orrick": ORRICK_ABS_DET_H23,
        "ratio_to_orrick": result.best_ratio_to_orrick,
        "above_orrick": result.above_orrick,
        "n_generations": result.n_generations,
        "n_restarts": result.n_restarts,
        "walltime_seconds": wall,
        "triple_verify": {"d1": d1, "d2": d2, "d3": d3, "consistent": consistent},
        "best_matrix": result.best_matrix,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
