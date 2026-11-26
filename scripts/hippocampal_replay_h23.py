"""Hippocampal Replay experiment driver for H23 maxdet.

Neuroscience domain analogy:
  Hippocampal replay (Wilson & McNaughton 1994; Skaggs & McNaughton 1996):
  during sleep, sharp-wave ripples replay episodes from the experience buffer.
  Row crossover implements "pattern completion" — reconstructing a new
  candidate from two stored high-quality matrices.

Usage:
    python scripts/hippocampal_replay_h23.py [--budget 600] [--seed 20260426]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hadamard_maxdet.hippocampal_replay import (
    ORRICK_ABS_DET_H23,
    hippocampal_replay,
    triple_verify,
)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=600.0,
                        help="Wall-clock budget in seconds")
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--buffer-capacity", type=int, default=20)
    parser.add_argument("--explore-iters", type=int, default=50_000)
    parser.add_argument("--consolidate-iters", type=int, default=40_000)
    parser.add_argument("--consolidation-fraction", type=float, default=0.40)
    parser.add_argument("--crossover-mode", choices=["row", "col"], default="row")
    args = parser.parse_args()

    print("=" * 70)
    print("Hippocampal Replay — H23 Maxdet — Neuroscience Domain Analogy")
    print("=" * 70)
    print(f"  Budget:                {args.budget:.0f} s")
    print(f"  Seed:                  {args.seed}")
    print(f"  Buffer capacity:       {args.buffer_capacity}")
    print(f"  Explore iters/chain:   {args.explore_iters:,}")
    print(f"  Consolidate iters:     {args.consolidate_iters:,}")
    print(f"  Consolidation frac:    {args.consolidation_fraction:.0%}")
    print(f"  Crossover mode:        {args.crossover_mode}")
    print(f"  Orrick record:         {ORRICK_ABS_DET_H23:,}")
    print()

    t0 = time.monotonic()
    result = hippocampal_replay(
        n=23,
        wall_budget_seconds=args.budget,
        buffer_capacity=args.buffer_capacity,
        explore_iters=args.explore_iters,
        consolidate_iters=args.consolidate_iters,
        consolidate_t_start=0.40,
        consolidate_t_end=0.003,
        explore_t_start=2.0,
        explore_t_end=0.003,
        min_buffer_for_consolidation=3,
        consolidation_fraction=args.consolidation_fraction,
        crossover_mode=args.crossover_mode,
        seed=args.seed,
    )
    wall = time.monotonic() - t0

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Wall time:             {wall:.1f} s")
    print(f"  Total SA steps:        {result.total_sa_steps:,}")
    print(f"  Exploration chains:    {result.n_exploration_chains}")
    print(f"  Consolidation chains:  {result.n_consolidation_chains}")
    print(f"  Buffer size (final):   {result.buffer_size_final}")
    print(f"  Best |det| found:      {result.best_abs_det:,}")
    print(f"  Orrick record:         {ORRICK_ABS_DET_H23:,}")
    print(f"  Ratio to Orrick:       {result.best_ratio_to_orrick:.4f}")
    print(f"  Above Orrick:          {result.above_orrick}")
    print()

    if result.best_abs_det > 0:
        print("Triple-verifying best matrix...")
        det_M, det_MT, det_copy = triple_verify(result.best_matrix)
        abs_dets = [abs(det_M), abs(det_MT), abs(det_copy)]
        consistent = (abs_dets[0] == abs_dets[1] == abs_dets[2])
        print(f"  det(M)         = {det_M:,}")
        print(f"  det(M^T)       = {det_MT:,}")
        print(f"  det(copy)      = {det_copy:,}")
        print(f"  Consistent:    {consistent}")
        if not consistent:
            print("  *** WARNING: triple verification inconsistency! ***")
        verified_abs = abs_dets[0]
    else:
        print("  No valid matrix found.")
        verified_abs = 0
        consistent = False

    if result.above_orrick:
        print()
        print("*" * 70)
        print("*** POTENTIAL NEW RECORD — TRIPLE VERIFICATION REQUIRED ***")
        print(f"*** |det| = {verified_abs:,}")
        print(f"*** Ratio = {verified_abs / ORRICK_ABS_DET_H23:.6f}")
        print("*" * 70)

    # Save result
    out = {
        "domain": "neuroscience",
        "technique": "hippocampal_replay_row_crossover",
        "best_abs_det": verified_abs,
        "orrick_record": ORRICK_ABS_DET_H23,
        "ratio_to_orrick": verified_abs / ORRICK_ABS_DET_H23 if ORRICK_ABS_DET_H23 else 0.0,
        "above_orrick": verified_abs > ORRICK_ABS_DET_H23,
        "triple_verified_consistent": consistent,
        "total_sa_steps": result.total_sa_steps,
        "n_exploration_chains": result.n_exploration_chains,
        "n_consolidation_chains": result.n_consolidation_chains,
        "buffer_size_final": result.buffer_size_final,
        "walltime_seconds": wall,
        "params": {
            "seed": args.seed,
            "buffer_capacity": args.buffer_capacity,
            "explore_iters": args.explore_iters,
            "consolidate_iters": args.consolidate_iters,
            "consolidation_fraction": args.consolidation_fraction,
            "crossover_mode": args.crossover_mode,
        },
        "best_matrix": result.best_matrix,
    }
    out_path = RESULTS_DIR / "hippocampal_replay_h23.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResult saved to: {out_path}")


if __name__ == "__main__":
    main()
