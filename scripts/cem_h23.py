"""Cross-Entropy Method search for H23 Hadamard maxdet.

Domain: Information Theory
Technique: CEM — maintain a per-entry Bernoulli distribution over ±1 matrix
entries, iteratively update it via maximum-likelihood on elite (high-|det|)
SA-refined samples, minimising the KL divergence between the sampling
distribution and the unknown distribution concentrated on the optimum.

Usage::

    python scripts/cem_h23.py

Writes ``results/cem_h23.json``.  Wall-clock cap: 14.5 minutes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hadamard_maxdet.cross_entropy import ORRICK_DET_H23, run_cem
from hadamard_maxdet.verifier import integer_determinant

RESULTS_PATH = REPO_ROOT / "results" / "cem_h23.json"


def main() -> int:
    t0 = time.monotonic()

    print("=" * 65)
    print("Cross-Entropy Method (CEM) — H23 Hadamard maxdet search")
    print(f"Orrick record: {ORRICK_DET_H23:,d}")
    print("Domain: Information Theory")
    print(f"Wall cap: 860 s  |  SA iters per sample: 500,000")
    print("=" * 65)

    result = run_cem(
        n=23,
        n_init=6,
        n_samples=5,
        n_elite=3,
        n_generations=4,
        sa_iterations=500_000,
        alpha=0.7,
        p_min=0.1,
        t_sa_start=2.0,
        t_sa_end=0.002,
        wall_cap=860.0,
        rng_seed=2026,
        verbose=True,
    )

    print("\n" + "=" * 65)
    print("Final result:")
    print(f"  Best |det|      = {result.best_abs_det:,d}")
    print(f"  Ratio to Orrick = {result.ratio_to_orrick:.6f}")
    print(f"  Above Orrick?   = {result.above_orrick}")
    print(f"  SA runs total   = {result.n_sa_total}")
    print(f"  Wall time       = {result.walltime_seconds:.1f} s")
    print(f"  Initial entropy = {result.initial_entropy:.4f} nats/entry")
    print(f"  Final entropy   = {result.final_entropy:.4f} nats/entry")
    print(f"  Entropy drop    = {result.initial_entropy - result.final_entropy:.4f} nats/entry")

    if result.above_orrick:
        print("\n*** |det| ABOVE ORRICK — TRIPLE-VERIFYING ***")
        M = result.best_matrix
        v1 = abs(integer_determinant(M))
        MT = [list(col) for col in zip(*M)]
        v2 = abs(integer_determinant(MT))
        v3 = abs(integer_determinant(M))
        print(f"  Verify M   : {v1:,d}")
        print(f"  Verify M^T : {v2:,d}")
        print(f"  Verify M   : {v3:,d}")
        if v1 == v2 == v3 and v1 > ORRICK_DET_H23:
            print("  TRIPLE-VERIFIED: |det| > Orrick record!")
        else:
            print("  WARNING: triple-verification inconsistent — check carefully")

    # Persist results
    payload = {
        "method": "cross_entropy_method",
        "domain": "information_theory",
        "n": 23,
        "best_abs_det": result.best_abs_det,
        "orrick_det": ORRICK_DET_H23,
        "ratio_to_orrick": result.ratio_to_orrick,
        "above_orrick": result.above_orrick,
        "initial_entropy_nats": result.initial_entropy,
        "final_entropy_nats": result.final_entropy,
        "entropy_drop": result.initial_entropy - result.final_entropy,
        "n_sa_total": result.n_sa_total,
        "walltime_seconds": result.walltime_seconds,
        "generation_log": result.generation_log,
        "best_matrix": result.best_matrix,
    }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nResults written to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
