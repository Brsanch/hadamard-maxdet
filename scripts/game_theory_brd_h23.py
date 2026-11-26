#!/usr/bin/env python3
"""
BRD game-theory search for H23 maxdet.

Runs Best Response Dynamics from many random ±1 starts.  Each start
converges (or cycle-detects with perturbation) to a local Nash equilibrium
of the cooperative |det| game.  Reports best |det| found and a histogram
of outcomes.

Wall-clock cap: 780 s (13 min).  Usage:
    python scripts/game_theory_brd_h23.py [--mode sync|async] [--seed N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hadamard_maxdet.game_theory_brd import (
    ORRICK_ABS_DET_H23,
    N23,
    brd_multistart_search,
    triple_verify,
)
from hadamard_maxdet.verifier import integer_determinant

RESULTS_DIR = Path(__file__).parent.parent / "results"
TIME_LIMIT = 780  # 13 minutes wall-clock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sync", "async", "greedy_sm"], default="greedy_sm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time", type=float, default=TIME_LIMIT)
    args = parser.parse_args()

    print(f"=== BRD game-theory search for H23 maxdet ===")
    print(f"mode={args.mode}  seed={args.seed}  time_limit={args.time:.0f}s")
    print(f"Orrick record: {ORRICK_ABS_DET_H23}")
    print()

    t0 = time.time()
    best_M, best_det, stats = brd_multistart_search(
        n=N23,
        time_limit_s=args.time,
        seed=args.seed,
        mode=args.mode,
        verbose=True,
    )
    elapsed = time.time() - t0

    print()
    print("=== RESULTS ===")
    print(f"Elapsed:        {elapsed:.1f} s")
    print(f"Trials:         {stats['n_trials']}")
    print(f"Trials/second:  {stats['trials_per_second']:.1f}")
    perturb_rate = stats.get("n_perturbs_total", 0) / max(1, stats["n_trials"])
    print(f"Perturb rate:   {perturb_rate:.3f} per trial")
    print(f"Best |det|:     {best_det}")
    print(f"Orrick ratio:   {best_det / ORRICK_ABS_DET_H23:.6f}")

    if best_det > ORRICK_ABS_DET_H23:
        print()
        print("*** ABOVE ORRICK — TRIPLE VERIFYING ***")
        det_M, det_MT, match = triple_verify(best_M)
        print(f"  det(M)   = {det_M}  |det| = {abs(det_M)}")
        print(f"  det(M^T) = {det_MT}  |det| = {abs(det_MT)}")
        print(f"  match: {match}")
        if not match:
            print("  WARNING: det(M) != det(M^T) — matrix may be invalid!")
        else:
            print("  TRIPLE VERIFY PASSED: new record confirmed!")
    else:
        print(f"Did not beat Orrick ({best_det/ORRICK_ABS_DET_H23:.4f}×)")

    print()
    print("Outcome histogram (fraction of Orrick):")
    hist = stats["det_histogram"]
    for bucket in sorted(hist):
        frac = bucket / 10.0
        count = hist[bucket]
        print(f"  [{frac:.1f}, {frac+0.1:.1f})×Orrick: {count} ({100*count/stats['n_trials']:.1f}%)")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "experiment": "game_theory_brd_h23",
        "mode": args.mode,
        "seed": args.seed,
        "best_abs_det": best_det,
        "orrick_abs_det": ORRICK_ABS_DET_H23,
        "ratio_to_orrick": best_det / ORRICK_ABS_DET_H23,
        "above_orrick": best_det > ORRICK_ABS_DET_H23,
        "stats": stats,
    }
    if best_M is not None:
        out["best_matrix"] = best_M
        # Re-verify for the JSON
        out["best_abs_det_verified"] = abs(integer_determinant(best_M))

    out_path = RESULTS_DIR / "game_theory_brd_h23.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
